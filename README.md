# Projeto da disciplina de Programação Concorrente & Distribuída
Grupo: 3 integrantes

- Paulo Dias
- Lucas Molinari
- Thomas Pires Correia

<strong>Objetivo do projeto:</strong>
O projeto tem como foco a implementação do algoritmo k-means unidimensional (com pontos 𝑋 [ 𝑖 ] e centróides 𝐶 [ 𝑐 ]), com medição do SSE (Sum of Squared Errors) e análise de desempenho. O núcleo do algoritmo será paralelizado utilizando três abordagens distintas, explorando diferentes modelos de computação concorrente:

1. OpenMP – paralelização em CPU com memória compartilhada;
2. CUDA – execução paralela na GPU;
3. MPI – paralelização em memória distribuída.

<strong>Medições e avaliação de desempenho:</strong>

1. Escalonamento em threads: testar T ∈ {1, 2, 4, 8, 16, ...};
2. Speedup: calcular como 
Speedup = tempo serial/ tempo OpenMP;
3. Afinamento da execução: ajustar parâmetros do OpenMP, como schedule (static vs dynamic) e chunk size;
4. Validação do algoritmo: garantir que o SSE não aumente ao longo das iterações (pode permanecer igual se o algoritmo convergir).

O objetivo final é comparar a eficiência e escalabilidade de cada abordagem em diferentes arquiteturas computacionais, observando ganhos de desempenho e comportamento do algoritmo em ambientes concorrentes.

---































Para Rodar:

Compilar:
gcc -O2 -std=c99 serial_k_means.c -o serial_k_means.exe -lm

Executar:
.\serial_k_means.exe dados.csv centroides_iniciais.csv 50 1e-6 assign.csv centroids.csv

Abrir arquivo gerado:
cat .\centroids.csv
