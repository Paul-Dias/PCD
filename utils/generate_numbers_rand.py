import random

def gerar_dados(N, K, filename_dados='dados.csv', filename_centroides='centroides_iniciais.csv'):
    # Gerar dados.csv com N pontos aleatórios
    with open(filename_dados, 'w') as f:
        for _ in range(N):
            valor = random.uniform(0, 100)
            f.write(f"{valor:.1f}\n")

    # Gerar centroides_iniciais.csv com K centroides
    with open(filename_centroides, 'w') as f:
        for _ in range(K):
            centroid = random.uniform(0, 100)
            f.write(f"{centroid:.1f}\n")

    print(f"Arquivos gerados: {filename_dados} ({N} pontos) e {filename_centroides} ({K} clusters)")

# Testes sugeridos:
# Pequeno
gerar_dados(50000, 5, 'dados_50k_k5.csv', 'centroides_50k_k5.csv')

# Médio - RECOMENDADO para ver diferença
gerar_dados(100000, 20, 'dados_100k_k20.csv', 'centroides_100k_k20.csv')

# Grande - melhor para benchmark
gerar_dados(500000, 50, 'dados_500k_k50.csv', 'centroides_500k_k50.csv')