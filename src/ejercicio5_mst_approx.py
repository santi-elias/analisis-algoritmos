import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree

def tour_length(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

def solve_tsp_exact_bruteforce(dist_matrix):
    n = dist_matrix.shape[0]
    best_c = float("inf")
    for perm in itertools.permutations(range(1, n)):
        tour = [0] + list(perm)
        cost = tour_length(tour, dist_matrix)
        if cost < best_c:
            best_c = cost
    return best_c

def tsp_mst_approximation(dist_matrix):
    n = dist_matrix.shape[0]
    mst = minimum_spanning_tree(dist_matrix).toarray()
    adj = mst + mst.T

    visited = []
    seen = set()

    def dfs(u):
        seen.add(u)
        visited.append(u)
        neighbors = np.where(adj[u] > 0)[0]
        for v in neighbors:
            if v not in seen:
                dfs(v)

    dfs(0)
    cost = tour_length(visited, dist_matrix)
    return visited, cost

def two_opt_refine(tour, dist_matrix):
    tour = tour.copy()
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n if i > 0 else n - 1):
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])
                if delta < -1e-6:
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    improved = True
                    break
            if improved:
                break
    return tour, tour_length(tour, dist_matrix)

def main():
    os.makedirs("plots", exist_ok=True)
    np.random.seed(999)
    n_instances = 100
    n_cities = 9

    ratios = []
    costs_opt = []
    costs_mst = []
    costs_mst_2opt = []

    print(f"{'='*60}")
    print(f" EJERCICIO 5: Aproximación por MST ({n_instances} instancias de {n_cities} ciudades)")
    print(f"{'='*60}")
    print("Ejecutando instancias...", flush=True)

    for inst in range(n_instances):
        coords = np.random.uniform(0, 100, size=(n_cities, 2))
        dist_matrix = cdist(coords, coords)

        c_opt = solve_tsp_exact_bruteforce(dist_matrix)
        mst_tour, c_mst = tsp_mst_approximation(dist_matrix)
        _, c_refined = two_opt_refine(mst_tour, dist_matrix)

        ratio = c_mst / c_opt
        ratios.append(ratio)
        costs_opt.append(c_opt)
        costs_mst.append(c_mst)
        costs_mst_2opt.append(c_refined)

    ratios = np.array(ratios)
    print(f"Razón media (Cobtenido / Cóptimo): {np.mean(ratios):.3f}")
    print(f"Razón máxima observada           : {np.max(ratios):.3f} (Límite teórico = 2.00)")
    print(f"Costo medio Óptimo              : {np.mean(costs_opt):.1f}")
    print(f"Costo medio MST Puro            : {np.mean(costs_mst):.1f}")
    print(f"Costo medio MST + 2-opt         : {np.mean(costs_mst_2opt):.1f}")
    print(f"{'='*60}\n")

    plt.figure(figsize=(7, 4.2))
    plt.hist(ratios, bins=16, color="#4292c6", edgecolor="black", alpha=0.85)
    plt.axvline(2.0, color="#cb181d", linestyle="--", linewidth=2.5, label="Límite Teórico (2.0)")
    plt.axvline(np.mean(ratios), color="#08519c", linestyle="-", linewidth=2, label=f"Media Empírica ({np.mean(ratios):.2f})")
    plt.title("Distribución de la Razón de Aproximación MST en 100 Instancias", fontsize=11)
    plt.xlabel("Razón $C_{\\text{obtenido}} / C_{\\text{óptimo}}$", fontsize=11)
    plt.ylabel("Frecuencia", fontsize=11)
    plt.xlim(1.0, 2.1)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/mst_approximation_ratio.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4.2))
    labels = ["Óptimo Exacto", "MST Puro", "MST + 2-opt"]
    plt.boxplot([costs_opt, costs_mst, costs_mst_2opt], tick_labels=labels, patch_artist=True,
                boxprops=dict(facecolor="#c7e9c0", color="#238b45"),
                medianprops=dict(color="#d95f0e", linewidth=2))
    plt.title("Comparativa de Costo: Óptimo vs. MST vs. MST + 2-opt", fontsize=11)
    plt.ylabel("Longitud del Tour", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/mst_cost_comparison.png", dpi=300)
    plt.close()
    print("-> Gráficas guardadas: plots/mst_approximation_ratio.png y plots/mst_cost_comparison.png")

if __name__ == "__main__":
    main()
