# PCD
Projeto dedicado para o projeto de programação concorrente distribuída 


# Para rodar:
Compilar:
gcc -O2 -std=c99 serial_k_means.c -o serial_k_means.exe -lm

Executar:
.\serial_k_means.exe dados.csv centroides_iniciais.csv 50 1e-6 assign.csv centroids.csv

Abrir arquivo gerado:
cat .\centroids.csv
