#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


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

        /* aceita vírgula/ponto-e-vírgula/espaco/tab, pega o primeiro token numérico */
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












int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            printf("Uso: %s dados.csv centroides.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
            printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        }

        MPI_Finalize();
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;
    const char *outAssign = (argc > 5) ? argv[5] : NULL;
    const char *outCentroid = (argc > 6) ? argv[6] : NULL;
    int N = 0, K = 0;
    double *X = NULL;
    double *C = NULL;
    int *assign = NULL;

    if (rank == 0) {
        X = read_csv_1col(pathX, &N);
        C = read_csv_1col(pathC, &K);

        if (N % size != 0){
            fprintf(stderr, "Erro: N (%d) deve ser divisivel pelo numero de processos (%d)\n", N, size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        assign = malloc(N * sizeof(int));
    }

    /* Broadcast N e K */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_N = N / size;

    double *X_local = malloc(local_N * sizeof(double));
    int *assign_local = malloc(local_N * sizeof(int));

    /* Distribui os dados */
    MPI_Scatter(X, local_N, MPI_DOUBLE,
                X_local, local_N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Broadcast centroides iniciais */
    if (rank != 0)
        C = malloc(K * sizeof(double));

    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double prev_sse = 1e300;
    double sse_global = 0.0;

    clock_t t_kmeans_start = clock();

    int it;
    for (it = 0; it < max_iter; it++)
    {
        /* -------- Assignment + SSE local -------- */
        double sse_local = 0.0;

        for (int i = 0; i < local_N; i++)
        {
            int best = -1;
            double bestd = 1e300;

            for (int c = 0; c < K; c++)
            {
                double diff = X_local[i] - C[c];
                double d = diff * diff;
                if (d < bestd)
                {
                    bestd = d;
                    best = c;
                }
            }

            assign_local[i] = best;
            sse_local += bestd;
        }

        /* -------- SSE global -------- */
        MPI_Allreduce(&sse_local, &sse_global,
                      1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);

        /* -------- Critério de parada -------- */
        double rel = fabs(sse_global - prev_sse) /
                     (prev_sse > 0.0 ? prev_sse : 1.0);

        int stop = (rel < eps);
        MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (stop)
        {
            it++;
            break;
        }

        prev_sse = sse_global;

        /* -------- Update dos centroides -------- */
        double *sum_local = calloc(K, sizeof(double));
        double *sum_global = calloc(K, sizeof(double));
        int *cnt_local = calloc(K, sizeof(int));
        int *cnt_global = calloc(K, sizeof(int));

        for (int i = 0; i < local_N; i++)
        {
            int a = assign_local[i];
            sum_local[a] += X_local[i];
            cnt_local[a]++;
        }

        MPI_Allreduce(sum_local, sum_global, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(cnt_local, cnt_global, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        for (int c = 0; c < K; c++)
        {
            if (cnt_global[c] > 0)
                C[c] = sum_global[c] / cnt_global[c];
        }

        free(sum_local);
        free(sum_global);
        free(cnt_local);
        free(cnt_global);
    }

    clock_t t_kmeans_end = clock();

    /* -------- Reúne assign final -------- */
    MPI_Gather(assign_local, local_N, MPI_INT,
               assign, local_N, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        write_assign_csv(outAssign, assign, N);
        write_centroids_csv(outCentroid, C, K);

        double ms_kmeans = 1000.0 *
            (double)(t_kmeans_end - t_kmeans_start) / CLOCKS_PER_SEC;

        printf("\n========================================\n");
        printf("K-means 1D (MPI)\n");
        printf("========================================\n");
        printf("Processos MPI: %d\n", size);
        printf("Iteracoes: %d\n", it);
        printf("SSE final: %.6f\n", sse_global);
        printf("Tempo k-means: %.3f ms\n", ms_kmeans);
        printf("========================================\n");
    }

    free(X_local);
    free(assign_local);
    if (rank == 0){
        free(X);
        free(assign);
    }
    free(C);

    MPI_Finalize();
    return 0;
}
