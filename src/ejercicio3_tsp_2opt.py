import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def tour_length(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

def nearest_neighbor_tour(dist_matrix, start_node=0):
    n = dist_matrix.shape[0]
    unvisited = set(range(n))
    tour = [start_node]
    unvisited.remove(start_node)

    current = start_node
    while unvisited:
        nxt = min(unvisited, key=lambda city: dist_matrix[current, city])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tour

def two_opt(initial_tour, dist_matrix):
    tour = initial_tour.copy()
    n = len(tour)
    num_improvements = 0
    improved = True

    while improved:
        improved = False
        best_delta = 0.0
        best_i, best_j = -1, -1

        for i in range(n - 1):
            for j in range(i + 2, n if i > 0 else n - 1):
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                current_cost = dist_matrix[a, b] + dist_matrix[c, d]
                new_cost = dist_matrix[a, c] + dist_matrix[b, d]
                delta = new_cost - current_cost

                if delta < -1e-6 and delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
                    improved = True

        if improved:
            tour[best_i + 1 : best_j + 1] = reversed(tour[best_i + 1 : best_j + 1])
            num_improvements += 1

    final_cost = tour_length(tour, dist_matrix)
    return tour, num_improvements, final_cost

def main():
    os.makedirs("plots", exist_ok=True)
    np.random.seed(42)
    n_cities = 30
    num_trials = 35

    coords = np.random.uniform(0, 100, size=(n_cities, 2))
    dist_matrix = cdist(coords, coords)

    improvements_random = []
    costs_random = []
    improvements_nn = []
    costs_nn = []

    print(f"{'='*60}")
    print(f" EJERCICIO 3: Búsqueda Local 2-opt (TSP {n_cities} ciudades, {num_trials} corridas)")
    print(f"{'='*60}")

    for _ in range(num_trials):
        rand_tour = list(np.random.permutation(n_cities))
        _, imp_r, cost_r = two_opt(rand_tour, dist_matrix)
        improvements_random.append(imp_r)
        costs_random.append(cost_r)

        start_city = np.random.randint(0, n_cities)
        nn_init = nearest_neighbor_tour(dist_matrix, start_node=start_city)
        _, imp_nn, cost_nn = two_opt(nn_init, dist_matrix)
        improvements_nn.append(imp_nn)
        costs_nn.append(cost_nn)

    print(f"Inicio Aleatorio          -> Mejoras medias: {np.mean(improvements_random):.1f} | Costo medio: {np.mean(costs_random):.1f}")
    print(f"Inicio Vecino Más Cercano -> Mejoras medias: {np.mean(improvements_nn):.1f} | Costo medio: {np.mean(costs_nn):.1f}")
    print(f"{'='*60}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))

    labels = ["Aleatorio", "Vecino Más Cercano"]
    ax1.boxplot([improvements_random, improvements_nn], tick_labels=labels, patch_artist=True,
                boxprops=dict(facecolor="#9ecae1", color="#3182bd"),
                medianprops=dict(color="#d95f0e", linewidth=2))
    ax1.set_title("Número de Mejoras 2-opt", fontsize=11)
    ax1.set_ylabel("Iteraciones de Mejora", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.boxplot([costs_random, costs_nn], tick_labels=labels, patch_artist=True,
                boxprops=dict(facecolor="#a1d99b", color="#31a354"),
                medianprops=dict(color="#d95f0e", linewidth=2))
    ax2.set_title("Costo del Tour Obtenido", fontsize=11)
    ax2.set_ylabel("Distancia Euclidiana Total", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig("plots/two_opt_comparison.png", dpi=300)
    plt.close()
    print("-> Gráfica guardada en: plots/two_opt_comparison.png")

if __name__ == "__main__":
    main()
