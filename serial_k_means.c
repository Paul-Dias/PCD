#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---------- util CSV 1D: cada linha tem 1 número ---------- */
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

/* ---------- k-means 1D ---------- */
/* assignment: para cada X[i], encontra c com menor (X[i]-C[c])^2 */
static double assignment_step_1d(const double *X, const double *C, int *assign, int N, int K)
{
    double sse = 0.0;
    for (int i = 0; i < N; i++)
    {
        int best = -1;
        double bestd = 1e300;
        for (int c = 0; c < K; c++)
        {
            double diff = X[i] - C[c];
            double d = diff * diff;
            if (d < bestd)
            {
                bestd = d;
                best = c;
            }
        }
        assign[i] = best;
        sse += bestd;
    }
    return sse;
}

/* update: média dos pontos de cada cluster (1D)
   se cluster vazio, copia X[0] (estratégia naive) */
static void update_step_1d(const double *X, double *C, const int *assign, int N, int K)
{
    double *sum = (double *)calloc((size_t)K, sizeof(double));
    int *cnt = (int *)calloc((size_t)K, sizeof(int));
    if (!sum || !cnt)
    {
        fprintf(stderr, "Sem memoria no update\n");
        exit(1);
    }

    for (int i = 0; i < N; i++)
    {
        int a = assign[i];
        cnt[a] += 1;
        sum[a] += X[i];
    }
    for (int c = 0; c < K; c++)
    {
        if (cnt[c] > 0)
            C[c] = sum[c] / (double)cnt[c];
        else
            C[c] = X[0]; /* simples: cluster vazio recebe o primeiro ponto */
    }
    free(sum);
    free(cnt);
}

static void kmeans_1d(const double *X, double *C, int *assign,
                      int N, int K, int max_iter, double eps,
                      int *iters_out, double *sse_out, double **sse_history)
{
    double *history = (double *)malloc((size_t)(max_iter + 1) * sizeof(double));
    if (!history)
    {
        fprintf(stderr, "Sem memoria para histórico SSE\n");
        exit(1);
    }

    double prev_sse = 1e300;
    double sse = 0.0;
    int it;

    for (it = 0; it < max_iter; it++)
    {
        sse = assignment_step_1d(X, C, assign, N, K);
        history[it] = sse; // Salva SSE de cada iteração

        /* parada por variação relativa do SSE */
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);

        if (rel < eps)
        {
            it++;
            break;
        }
        update_step_1d(X, C, assign, N, K);
        prev_sse = sse;
    }

    *iters_out = it;
    *sse_out = sse;
    *sse_history = history;
}

/* ---------- main ---------- */
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps = (argc > 4) ? atof(argv[4]) : 1e-4;
    const char *outAssign = (argc > 5) ? argv[5] : NULL;
    const char *outCentroid = (argc > 6) ? argv[6] : NULL;

    if (max_iter <= 0 || eps <= 0.0)
    {
        fprintf(stderr, "Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    // ===== MEDIÇÃO: Início total =====
    clock_t t_total_start = clock();

    int N = 0, K = 0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int *)malloc((size_t)N * sizeof(int));
    if (!assign)
    {
        fprintf(stderr, "Sem memoria para assign\n");
        free(X);
        free(C);
        return 1;
    }

    // ===== MEDIÇÃO: Início k-means (sem I/O) =====
    clock_t t_kmeans_start = clock();

    int iters = 0;
    double sse = 0.0;
    double *sse_history = NULL;
    kmeans_1d(X, C, assign, N, K, max_iter, eps, &iters, &sse, &sse_history);

    // ===== MEDIÇÃO: Fim k-means =====
    clock_t t_kmeans_end = clock();

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    // ===== MEDIÇÃO: Fim total =====
    clock_t t_total_end = clock();

    // ===== CÁLCULOS DE MÉTRICAS =====
    double ms_total = 1000.0 * (double)(t_total_end - t_total_start) / (double)CLOCKS_PER_SEC;
    double ms_kmeans = 1000.0 * (double)(t_kmeans_end - t_kmeans_start) / (double)CLOCKS_PER_SEC;
    double ms_per_iter = (iters > 0) ? (ms_kmeans / (double)iters) : 0.0;

    // Throughput
    double total_ops = (double)N * (double)K * (double)iters;
    double throughput_ops = (ms_kmeans > 0) ? (total_ops / (ms_kmeans / 1000.0)) : 0.0;
    double throughput_points = (ms_kmeans > 0) ? ((double)N * (double)iters / (ms_kmeans / 1000.0)) : 0.0;

    // SSE inicial e final
    double sse_initial = (iters > 0) ? sse_history[0] : 0.0;
    double sse_final = sse;
    double sse_reduction = ((sse_initial > 0) ? ((sse_initial - sse_final) / sse_initial * 100.0) : 0.0);

    // ===== IMPRESSÃO DE RESULTADOS =====
    printf("\n========================================\n");
    printf("K-means 1D (SERIAL - Baseline)\n");
    printf("========================================\n");
    printf("Dataset: N=%d pontos | K=%d clusters\n", N, K);
    printf("Parametros: max_iter=%d | eps=%g\n", max_iter, eps);
    printf("----------------------------------------\n");
    printf("ALGORITMO:\n");
    printf("  Iteracoes realizadas: %d\n", iters);
    printf("  SSE inicial:  %.6f\n", sse_initial);
    printf("  SSE final:    %.6f\n", sse_final);
    printf("  Reducao SSE:  %.2f%%\n", sse_reduction);
    printf("----------------------------------------\n");
    printf("TEMPO:\n");
    printf("  Tempo total (com I/O):  %.3f ms\n", ms_total);
    printf("  Tempo k-means (puro):   %.3f ms\n", ms_kmeans);
    printf("  Tempo por iteracao:     %.3f ms\n", ms_per_iter);
    printf("----------------------------------------\n");
    printf("THROUGHPUT:\n");
    printf("  Operacoes totais:       %.2e\n", total_ops);
    printf("  Operacoes/segundo:      %.2e ops/s\n", throughput_ops);
    printf("  Pontos/segundo:         %.2e pts/s\n", throughput_points);
    printf("========================================\n");

    // ===== VALIDAÇÃO: SSE não deve crescer =====
    printf("\nVALIDACAO (SSE monotonico):\n");
    int monotonic = 1;
    for (int i = 1; i < iters; i++)
    {
        if (sse_history[i] > sse_history[i - 1] + 1e-9) // tolerância numérica
        {
            printf("  AVISO: SSE aumentou na iteracao %d (%.6f -> %.6f)\n",
                   i, sse_history[i - 1], sse_history[i]);
            monotonic = 0;
        }
    }
    if (monotonic)
        printf("SSE monotonicamente nao crescente\n");

    // ===== SALVAR MÉTRICAS EM CSV (para análise posterior) =====
    FILE *metrics = fopen("metrics_serial_log/metrics_serial.csv", "w");
    if (metrics)
    {
        fprintf(metrics, "N,K,max_iter,eps,iterations,sse_initial,sse_final,");
        fprintf(metrics, "time_total_ms,time_kmeans_ms,time_per_iter_ms,");
        fprintf(metrics, "throughput_ops_per_sec,throughput_points_per_sec\n");

        fprintf(metrics, "%d,%d,%d,%g,%d,%.6f,%.6f,%.3f,%.3f,%.3f,%.2e,%.2e\n",
                N, K, max_iter, eps, iters, sse_initial, sse_final,
                ms_total, ms_kmeans, ms_per_iter, throughput_ops, throughput_points);
        fclose(metrics);
        printf("\n Metricas salvas em: metrics_serial.csv\n");
    }

    // ===== SALVAR HISTÓRICO SSE =====
    FILE *hist = fopen("metrics_serial_log/sse_history_serial.csv", "w");
    if (hist)
    {
        fprintf(hist, "iteration,sse\n");
        for (int i = 0; i < iters; i++)
            fprintf(hist, "%d,%.6f\n", i, sse_history[i]);
        fclose(hist);
        printf(" Historico SSE salvo em: sse_history_serial.csv\n");
    }

    free(sse_history);
    free(assign);
    free(X);
    free(C);
    return 0;
}
