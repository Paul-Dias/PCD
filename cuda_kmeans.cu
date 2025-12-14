#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// Fully Parallelized CUDA K-means 1D - NO LOOPS IN GPU THREADS!
// Usage: cuda_kmeans data.csv centroids.csv [max_iter=50] [eps=1e-4] [update=gpu] [block=256] [assign_out] [centroids_out]

#define CHECK(call)                                                                                    \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                                                   \
        }                                                                                              \
    } while (0)

static int count_rows(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s\n", path);
        exit(1);
    }
    int rows = 0;
    char line[8192];
    while (fgets(line, sizeof(line), f))
    {
        int only_ws = 1;
        for (char *p = line; *p; p++)
        {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
            {
                only_ws = 0;
                break;
            }
        }
        if (!only_ws)
            rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out)
{
    int R = count_rows(path);
    if (R <= 0)
    {
        fprintf(stderr, "Arquivo vazio: %s\n", path);
        exit(1);
    }
    double *A = (double *)malloc((size_t)R * sizeof(double));
    if (!A)
    {
        fprintf(stderr, "Sem memoria para %d linhas\n", R);
        exit(1);
    }

    FILE *f = fopen(path, "r");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s\n", path);
        free(A);
        exit(1);
    }

    char line[8192];
    int r = 0;
    while (fgets(line, sizeof(line), f))
    {
        int only_ws = 1;
        for (char *p = line; *p; p++)
        {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
            {
                only_ws = 0;
                break;
            }
        }
        if (only_ws)
            continue;

        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if (!tok)
        {
            fprintf(stderr, "Linha %d sem valor em %s\n", r + 1, path);
            free(A);
            fclose(f);
            exit(1);
        }
        A[r] = atof(tok);
        r++;
        if (r > R)
            break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N)
{
    if (!path)
        return;
    FILE *f = fopen(path, "w");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s para escrita\n", path);
        return;
    }
    for (int i = 0; i < N; i++)
        fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K)
{
    if (!path)
        return;
    FILE *f = fopen(path, "w");
    if (!f)
    {
        fprintf(stderr, "Erro ao abrir %s para escrita\n", path);
        return;
    }
    for (int c = 0; c < K; c++)
        fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

// atomicAdd for double on devices where native atomicAdd(double) is not available
static __device__ double atomicAdd_double(double *address, double val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    double old_d;
    do
    {
        assumed = old;
        old_d = __longlong_as_double(assumed);
        double new_d = old_d + val;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_d));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

// ============================================================================
// KERNEL 1: Compute ALL distances in parallel (N x K threads)
// NO LOOPS! Each thread computes ONE distance
// ============================================================================
__global__ void compute_distances_kernel(const double *d_X, const double *d_C,
                                         double *d_distances, int N, int K)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // point index
    int c = blockIdx.x * blockDim.x + threadIdx.x; // centroid index

    if (i >= N || c >= K)
        return;

    // Each thread computes ONE distance - FULLY PARALLEL!
    double diff = d_X[i] - d_C[c];
    d_distances[i * K + c] = diff * diff;
}

// ============================================================================
// KERNEL 2: Find minimum distance for each point using parallel reduction
// Each block processes ONE point, threads reduce over K centroids
// ============================================================================
__global__ void find_min_distances_kernel(const double *d_distances,
                                          int *d_assign, double *d_err, int N, int K)
{
    int i = blockIdx.x; // each block = one point
    int tid = threadIdx.x;

    if (i >= N)
        return;

    extern __shared__ double s_dist[];
    int *s_idx = (int *)&s_dist[blockDim.x];

    // Each thread finds local minimum over its assigned centroids
    double local_min = 1e300;
    int local_idx = -1;

    for (int c = tid; c < K; c += blockDim.x)
    {
        double dist = d_distances[i * K + c];
        if (dist < local_min)
        {
            local_min = dist;
            local_idx = c;
        }
    }

    s_dist[tid] = local_min;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Parallel reduction in shared memory (tree reduction)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride && s_dist[tid + stride] < s_dist[tid])
        {
            s_dist[tid] = s_dist[tid + stride];
            s_idx[tid] = s_idx[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes final result
    if (tid == 0)
    {
        d_assign[i] = s_idx[0];
        d_err[i] = s_dist[0];
    }
}

// ============================================================================
// KERNEL 3: Parallel SSE reduction using tree reduction
// ============================================================================
__global__ void reduce_sse_kernel(const double *d_err, double *d_partial_sse, int N)
{
    extern __shared__ double s_data[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    s_data[tid] = (i < N) ? d_err[i] : 0.0;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0)
    {
        d_partial_sse[blockIdx.x] = s_data[0];
    }
}

// ============================================================================
// KERNEL 4: Update centroids using atomics (parallel accumulation)
// ============================================================================
__global__ void update_atomic_kernel(const double *d_X, const int *d_assign,
                                     double *d_sum, int *d_cnt, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    int a = d_assign[i];
    atomicAdd_double(&d_sum[a], d_X[i]);
    atomicAdd(&d_cnt[a], 1);
}

// ============================================================================
// KERNEL 5: Finalize centroids on GPU (parallel division)
// ============================================================================
__global__ void finalize_centroids_kernel(double *d_C, const double *d_sum,
                                          const int *d_cnt, const double *d_X, int K)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K)
        return;

    if (d_cnt[c] > 0)
        d_C[c] = d_sum[c] / (double)d_cnt[c];
    else
        d_C[c] = d_X[0]; // fallback
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Uso: %s dados.csv centroides.csv [max_iter=50] [eps=1e-4] [block=256] [assign_out] [centroids_out]\n", argv[0]);
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;
    int blocksize = (argc > 5) ? atoi(argv[5]) : 256;
    const char *outAssign = (argc > 6) ? argv[6] : NULL;
    const char *outCentroid = (argc > 7) ? argv[7] : NULL;

    // Read data
    int N = 0, K = 0;
    double *h_X = read_csv_1col(pathX, &N);
    double *h_C = read_csv_1col(pathC, &K);
    int *h_assign = (int *)malloc((size_t)N * sizeof(int));
    double *h_err = (double *)malloc((size_t)N * sizeof(double));

    // Allocate device memory
    double *d_X, *d_C, *d_distances;
    int *d_assign;
    double *d_err, *d_sum;
    int *d_cnt;

    CHECK(cudaMalloc(&d_X, (size_t)N * sizeof(double)));
    CHECK(cudaMalloc(&d_C, (size_t)K * sizeof(double)));
    CHECK(cudaMalloc(&d_distances, (size_t)N * K * sizeof(double)));
    CHECK(cudaMalloc(&d_assign, (size_t)N * sizeof(int)));
    CHECK(cudaMalloc(&d_err, (size_t)N * sizeof(double)));
    CHECK(cudaMalloc(&d_sum, (size_t)K * sizeof(double)));
    CHECK(cudaMalloc(&d_cnt, (size_t)K * sizeof(int)));

    // For SSE reduction
    int grid_sse = (N + blocksize - 1) / blocksize;
    double *d_partial_sse, *h_partial_sse;
    CHECK(cudaMalloc(&d_partial_sse, (size_t)grid_sse * sizeof(double)));
    h_partial_sse = (double *)malloc((size_t)grid_sse * sizeof(double));

    // Copy initial data H2D
    clock_t t0_total = clock();
    CHECK(cudaMemcpy(d_X, h_X, (size_t)N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, h_C, (size_t)K * sizeof(double), cudaMemcpyHostToDevice));

    // Setup 2D grid for distance computation (N x K threads)
    dim3 block2d(16, 16); // 16x16 = 256 threads per block
    dim3 grid2d((K + block2d.x - 1) / block2d.x, (N + block2d.y - 1) / block2d.y);

    // Events for timing
    cudaEvent_t ev_start, ev_stop;
    CHECK(cudaEventCreate(&ev_start));
    CHECK(cudaEventCreate(&ev_stop));

    float total_dist_ms = 0.0f, total_assign_ms = 0.0f;
    float total_sse_ms = 0.0f, total_update_ms = 0.0f, total_finalize_ms = 0.0f;

    double prev_sse = 1e300, sse = 0.0;
    int iters = 0;

    printf("\n========================================\n");
    printf("K-means 1D CUDA (FULLY PARALLELIZED)\n");
    printf("========================================\n");
    printf("Dataset: N=%d pontos | K=%d clusters\n", N, K);
    printf("Total threads para distancias: %d x %d = %d\n", N, K, N * K);
    printf("========================================\n\n");

    for (int it = 0; it < max_iter; it++)
    {
        // STEP 1: Compute ALL distances in parallel (N*K threads)
        CHECK(cudaEventRecord(ev_start));
        compute_distances_kernel<<<grid2d, block2d>>>(d_X, d_C, d_distances, N, K);
        CHECK(cudaEventRecord(ev_stop));
        CHECK(cudaEventSynchronize(ev_stop));
        float ms1;
        CHECK(cudaEventElapsedTime(&ms1, ev_start, ev_stop));
        total_dist_ms += ms1;

        // STEP 2: Find minimum distance for each point (parallel reduction)
        size_t shared_size = blocksize * (sizeof(double) + sizeof(int));
        CHECK(cudaEventRecord(ev_start));
        find_min_distances_kernel<<<N, blocksize, shared_size>>>(d_distances, d_assign, d_err, N, K);
        CHECK(cudaEventRecord(ev_stop));
        CHECK(cudaEventSynchronize(ev_stop));
        float ms2;
        CHECK(cudaEventElapsedTime(&ms2, ev_start, ev_stop));
        total_assign_ms += ms2;

        // STEP 3: Compute SSE with parallel reduction
        CHECK(cudaEventRecord(ev_start));
        reduce_sse_kernel<<<grid_sse, blocksize, blocksize * sizeof(double)>>>(d_err, d_partial_sse, N);
        CHECK(cudaEventRecord(ev_stop));
        CHECK(cudaEventSynchronize(ev_stop));
        float ms3;
        CHECK(cudaEventElapsedTime(&ms3, ev_start, ev_stop));
        total_sse_ms += ms3;

        // Final SSE reduction on CPU (small array)
        CHECK(cudaMemcpy(h_partial_sse, d_partial_sse, (size_t)grid_sse * sizeof(double), cudaMemcpyDeviceToHost));
        sse = 0.0;
        for (int b = 0; b < grid_sse; b++)
            sse += h_partial_sse[b];

        // Check convergence
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        iters = it + 1;

        printf("Iter %2d: SSE = %.6f (rel_change = %.6e)\n", it + 1, sse, rel);

        if (rel < eps)
        {
            printf("Convergiu!\n");
            break;
        }

        // STEP 4: Update centroids
        CHECK(cudaMemset(d_sum, 0, (size_t)K * sizeof(double)));
        CHECK(cudaMemset(d_cnt, 0, (size_t)K * sizeof(int)));

        int grid_update = (N + blocksize - 1) / blocksize;
        CHECK(cudaEventRecord(ev_start));
        update_atomic_kernel<<<grid_update, blocksize>>>(d_X, d_assign, d_sum, d_cnt, N);
        CHECK(cudaEventRecord(ev_stop));
        CHECK(cudaEventSynchronize(ev_stop));
        float ms4;
        CHECK(cudaEventElapsedTime(&ms4, ev_start, ev_stop));
        total_update_ms += ms4;

        // STEP 5: Finalize centroids on GPU
        int grid_finalize = (K + 256 - 1) / 256;
        CHECK(cudaEventRecord(ev_start));
        finalize_centroids_kernel<<<grid_finalize, 256>>>(d_C, d_sum, d_cnt, d_X, K);
        CHECK(cudaEventRecord(ev_stop));
        CHECK(cudaEventSynchronize(ev_stop));
        float ms5;
        CHECK(cudaEventElapsedTime(&ms5, ev_start, ev_stop));
        total_finalize_ms += ms5;

        prev_sse = sse;
    }

    clock_t t_end = clock();
    double total_time_ms = 1000.0 * (double)(t_end - t0_total) / (double)CLOCKS_PER_SEC;

    // Copy final results
    CHECK(cudaMemcpy(h_assign, d_assign, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_C, d_C, (size_t)K * sizeof(double), cudaMemcpyDeviceToHost));

    // Print results
    printf("\n========================================\n");
    printf("RESULTADOS FINAIS\n");
    printf("========================================\n");
    printf("Iteracoes: %d\n", iters);
    printf("SSE final: %.6f\n", sse);
    printf("----------------------------------------\n");
    printf("TEMPOS DE KERNEL (ms):\n");
    printf("  Distancias (N*K threads): %.3f ms\n", total_dist_ms);
    printf("  Assignment (reducao):     %.3f ms\n", total_assign_ms);
    printf("  SSE reduction:            %.3f ms\n", total_sse_ms);
    printf("  Update (atomics):         %.3f ms\n", total_update_ms);
    printf("  Finalize centroids:       %.3f ms\n", total_finalize_ms);
    printf("  TOTAL GPU:                %.3f ms\n",
           total_dist_ms + total_assign_ms + total_sse_ms + total_update_ms + total_finalize_ms);
    printf("  TOTAL (com CPU/IO):       %.3f ms\n", total_time_ms);
    printf("----------------------------------------\n");

    double kernel_time = total_dist_ms + total_assign_ms + total_sse_ms + total_update_ms + total_finalize_ms;
    double ops_per_sec = ((double)N * (double)K * (double)iters) / (kernel_time / 1000.0);
    printf("THROUGHPUT:\n");
    printf("  Operacoes/segundo: %.2e ops/s\n", ops_per_sec);
    printf("  Tempo por iteracao: %.3f ms\n", kernel_time / (double)iters);
    printf("========================================\n");

    // Write outputs
    write_assign_csv(outAssign, h_assign, N);
    write_centroids_csv(outCentroid, h_C, K);

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_C);
    cudaFree(d_distances);
    cudaFree(d_assign);
    cudaFree(d_err);
    cudaFree(d_sum);
    cudaFree(d_cnt);
    cudaFree(d_partial_sse);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    free(h_X);
    free(h_C);
    free(h_assign);
    free(h_err);
    free(h_partial_sse);

    return 0;
}
