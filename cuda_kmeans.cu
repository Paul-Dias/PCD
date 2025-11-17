#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// Simple CUDA K-means 1D
// Usage: cuda_kmeans data.csv centroids.csv [max_iter=50] [eps=1e-4] [update=host|gpu] [block=256] [assign_out] [centroids_out]

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

// assignment kernel: one thread per point
__global__ void assignment_kernel(const double *d_X, const double *d_C, int *d_assign, double *d_err, int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    double xi = d_X[i];
    double bestd = 1e300;
    int best = -1;
    for (int c = 0; c < K; c++)
    {
        double diff = xi - d_C[c];
        double d = diff * diff;
        if (d < bestd)
        {
            bestd = d;
            best = c;
        }
    }
    d_assign[i] = best;
    d_err[i] = bestd;
}

// update kernel using atomics: accumulate sum and count
__global__ void update_atomic_kernel(const double *d_X, const int *d_assign, double *d_sum, int *d_cnt, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    int a = d_assign[i];
    // atomic add for double
    atomicAdd_double(&d_sum[a], d_X[i]);
    atomicAdd(&d_cnt[a], 1);
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [update=host|gpu] [block=256] [assign_out] [centroids_out]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;
    const char *update_mode_str = (argc > 5) ? argv[5] : "host";
    int blocksize = (argc > 6) ? atoi(argv[6]) : 256;
    const char *outAssign = (argc > 7) ? argv[7] : NULL;
    const char *outCentroid = (argc > 8) ? argv[8] : NULL;

    int update_gpu = (strcmp(update_mode_str, "gpu") == 0) ? 1 : 0;

    if (max_iter <= 0 || eps <= 0.0)
    {
        fprintf(stderr, "Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    int N = 0, K = 0;
    double *h_X = read_csv_1col(pathX, &N);
    double *h_C = read_csv_1col(pathC, &K);
    int *h_assign = (int *)malloc((size_t)N * sizeof(int));
    double *h_err = (double *)malloc((size_t)N * sizeof(double));
    if (!h_assign || !h_err)
    {
        fprintf(stderr, "Sem memoria host\n");
        return 1;
    }

    // Device buffers
    double *d_X = NULL;
    double *d_C = NULL;
    int *d_assign = NULL;
    double *d_err = NULL;

    CHECK(cudaMalloc((void **)&d_X, (size_t)N * sizeof(double)));
    CHECK(cudaMalloc((void **)&d_C, (size_t)K * sizeof(double)));
    CHECK(cudaMalloc((void **)&d_assign, (size_t)N * sizeof(int)));
    CHECK(cudaMalloc((void **)&d_err, (size_t)N * sizeof(double)));

    // For atomic update option
    double *d_sum = NULL;
    int *d_cnt = NULL;
    if (update_gpu)
    {
        CHECK(cudaMalloc((void **)&d_sum, (size_t)K * sizeof(double)));
        CHECK(cudaMalloc((void **)&d_cnt, (size_t)K * sizeof(int)));
    }

    // copy initial data H2D
    clock_t t0_total = clock();
    double h2d_time_ms = 0.0, d2h_time_ms = 0.0;
    clock_t t0 = clock();
    CHECK(cudaMemcpy(d_X, h_X, (size_t)N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, h_C, (size_t)K * sizeof(double), cudaMemcpyHostToDevice));
    clock_t t1 = clock();
    h2d_time_ms += 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    int grid = (N + blocksize - 1) / blocksize;

    // Events for kernel timing
    cudaEvent_t ev_start, ev_stop;
    CHECK(cudaEventCreate(&ev_start));
    CHECK(cudaEventCreate(&ev_stop));

    float total_kernel_ms = 0.0f;
    float total_update_kernel_ms = 0.0f;

    double prev_sse = 1e300;
    double sse = 0.0;
    int iters = 0;

    for (int it = 0; it < max_iter; it++)
    {
        // assignment kernel
        CHECK(cudaEventRecord(ev_start, 0));
        assignment_kernel<<<grid, blocksize>>>(d_X, d_C, d_assign, d_err, N, K);
        CHECK(cudaEventRecord(ev_stop, 0));
        CHECK(cudaEventSynchronize(ev_stop));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        total_kernel_ms += ms;

        // copy errors back to host and compute sse (could do device reduction)
        clock_t t2 = clock();
        CHECK(cudaMemcpy(h_err, d_err, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));
        clock_t t3 = clock();
        d2h_time_ms += 1000.0 * (double)(t3 - t2) / (double)CLOCKS_PER_SEC;

        sse = 0.0;
        for (int i = 0; i < N; i++)
            sse += h_err[i];

        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        iters = it + 1;
        if (rel < eps)
        {
            break;
        }

        if (update_gpu)
        {
            // zero sum and cnt
            CHECK(cudaMemset(d_sum, 0, (size_t)K * sizeof(double)));
            CHECK(cudaMemset(d_cnt, 0, (size_t)K * sizeof(int)));
            // update kernel using atomics
            CHECK(cudaEventRecord(ev_start, 0));
            update_atomic_kernel<<<grid, blocksize>>>(d_X, d_assign, d_sum, d_cnt, N);
            CHECK(cudaEventRecord(ev_stop, 0));
            CHECK(cudaEventSynchronize(ev_stop));
            float ms2;
            CHECK(cudaEventElapsedTime(&ms2, ev_start, ev_stop));
            total_update_kernel_ms += ms2;

            // copy sums back and finalize centroids on host
            clock_t t4 = clock();
            double *h_sum = (double *)malloc((size_t)K * sizeof(double));
            int *h_cnt = (int *)malloc((size_t)K * sizeof(int));
            if (!h_sum || !h_cnt)
            {
                fprintf(stderr, "Sem memoria host para sums\n");
                exit(1);
            }
            CHECK(cudaMemcpy(h_sum, d_sum, (size_t)K * sizeof(double), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(h_cnt, d_cnt, (size_t)K * sizeof(int), cudaMemcpyDeviceToHost));
            clock_t t5 = clock();
            d2h_time_ms += 1000.0 * (double)(t5 - t4) / (double)CLOCKS_PER_SEC;

            for (int c = 0; c < K; c++)
            {
                if (h_cnt[c] > 0)
                    h_C[c] = h_sum[c] / (double)h_cnt[c];
                else
                    h_C[c] = h_X[0];
            }
            free(h_sum);
            free(h_cnt);

            // copy updated centroids to device
            clock_t t6 = clock();
            CHECK(cudaMemcpy(d_C, h_C, (size_t)K * sizeof(double), cudaMemcpyHostToDevice));
            clock_t t7 = clock();
            h2d_time_ms += 1000.0 * (double)(t7 - t6) / (double)CLOCKS_PER_SEC;
        }
        else
        {
            // copy assign back to host and compute centroids on host
            clock_t t8 = clock();
            CHECK(cudaMemcpy(h_assign, d_assign, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
            clock_t t9 = clock();
            d2h_time_ms += 1000.0 * (double)(t9 - t8) / (double)CLOCKS_PER_SEC;

            // compute sums and counts
            double *sum = (double *)calloc((size_t)K, sizeof(double));
            int *cnt = (int *)calloc((size_t)K, sizeof(int));
            if (!sum || !cnt)
            {
                fprintf(stderr, "Sem memoria host em update\n");
                exit(1);
            }
            for (int i = 0; i < N; i++)
            {
                int a = h_assign[i];
                cnt[a] += 1;
                sum[a] += h_X[i];
            }
            for (int c = 0; c < K; c++)
            {
                if (cnt[c] > 0)
                    h_C[c] = sum[c] / (double)cnt[c];
                else
                    h_C[c] = h_X[0];
            }
            free(sum);
            free(cnt);

            // copy centroids back
            clock_t t10 = clock();
            CHECK(cudaMemcpy(d_C, h_C, (size_t)K * sizeof(double), cudaMemcpyHostToDevice));
            clock_t t11 = clock();
            h2d_time_ms += 1000.0 * (double)(t11 - t10) / (double)CLOCKS_PER_SEC;
        }

        prev_sse = sse;
    }

    clock_t t_total_end = clock();
    double total_ms = 1000.0 * (double)(t_total_end - t0_total) / (double)CLOCKS_PER_SEC;

    // copy final assignments and centroids to host
    clock_t t12 = clock();
    CHECK(cudaMemcpy(h_assign, d_assign, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_C, d_C, (size_t)K * sizeof(double), cudaMemcpyDeviceToHost));
    clock_t t13 = clock();
    d2h_time_ms += 1000.0 * (double)(t13 - t12) / (double)CLOCKS_PER_SEC;

    printf("\n========================================\n");
    printf("K-means 1D (CUDA)\n");
    printf("========================================\n");
    printf("Dataset: N=%d pontos | K=%d clusters\n", N, K);
    printf("Parametros: max_iter=%d | eps=%g | update=%s | block=%d\n", max_iter, eps, update_gpu ? "gpu(atomics)" : "host", blocksize);
    printf("----------------------------------------\n");
    printf("Iteracoes realizadas: %d\n", iters);
    printf("SSE final: %.6f\n", sse);
    printf("----------------------------------------\n");
    printf("TEMPOS (ms):\n");
    printf("  H2D total: %.3f ms\n", h2d_time_ms);
    printf("  Kernel assign total: %.3f ms\n", total_kernel_ms);
    if (update_gpu)
        printf("  Kernel update (atomics) total: %.3f ms\n", total_update_kernel_ms);
    printf("  D2H total: %.3f ms\n", d2h_time_ms);
    printf("  Tempo total: %.3f ms\n", total_ms);
    printf("----------------------------------------\n");

    double ms_kmeans = total_kernel_ms + total_update_kernel_ms + h2d_time_ms + d2h_time_ms; // approx
    double ms_per_iter = (iters > 0) ? (ms_kmeans / (double)iters) : 0.0;
    double total_ops = (double)N * (double)K * (double)iters;
    double throughput_ops = (ms_kmeans > 0) ? (total_ops / (ms_kmeans / 1000.0)) : 0.0;
    double throughput_points = (ms_kmeans > 0) ? ((double)N * (double)iters / (ms_kmeans / 1000.0)) : 0.0;

    printf("THROUGHPUT:\n");
    printf("  Operacoes totais: %.2e\n", total_ops);
    printf("  Operacoes/segundo: %.2e ops/s\n", throughput_ops);
    printf("  Pontos/segundo: %.2e pts/s\n", throughput_points);
    printf("========================================\n");

    // write outputs
    write_assign_csv(outAssign, h_assign, N);
    write_centroids_csv(outCentroid, h_C, K);

    // cleanup
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    if (d_sum)
        cudaFree(d_sum);
    if (d_cnt)
        cudaFree(d_cnt);
    cudaFree(d_X);
    cudaFree(d_C);
    cudaFree(d_assign);
    cudaFree(d_err);

    free(h_X);
    free(h_C);
    free(h_assign);
    free(h_err);

    return 0;
}
