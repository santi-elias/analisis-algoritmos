import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def tour_length(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

def simulated_annealing(dist_matrix, schedule="exponential", max_iter=8000, t_start=150.0, t_end=1e-2):
    n = dist_matrix.shape[0]
    current_tour = list(np.random.permutation(n))
    current_cost = tour_length(current_tour, dist_matrix)
    best_tour = current_tour.copy()
    best_cost = current_cost

    cost_history = []
    temp_history = []

    start_time = time.perf_counter()

    for k in range(1, max_iter + 1):
        if schedule == "linear":
            t = t_start - (t_start - t_end) * (k / max_iter)
        elif schedule == "exponential":
            alpha = (t_end / t_start) ** (1.0 / max_iter)
            t = t_start * (alpha ** k)
        elif schedule == "logarithmic":
            c = t_start
            t = c / np.log(1.0 + k)
        else:
            raise ValueError("Esquema desconocido")

        t = max(t, 1e-4)

        i, j = np.random.choice(n, size=2, replace=False)
        if i > j:
            i, j = j, i

        a, b = current_tour[i], current_tour[(i + 1) % n]
        c, d = current_tour[j], current_tour[(j + 1) % n]
        delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])

        if delta < 0 or np.random.rand() < np.exp(-delta / t):
            current_tour[i + 1 : j + 1] = reversed(current_tour[i + 1 : j + 1])
            current_cost += delta

            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = current_tour.copy()

        cost_history.append(current_cost)
        temp_history.append(t)

    elapsed_time = time.perf_counter() - start_time
    return best_tour, best_cost, elapsed_time, cost_history, temp_history

def main():
    os.makedirs("plots", exist_ok=True)
    np.random.seed(123)
    n_cities = 25
    coords = np.random.uniform(0, 100, size=(n_cities, 2))
    dist_matrix = cdist(coords, coords)

    schedules = ["linear", "exponential", "logarithmic"]
    results = {}

    print(f"{'='*60}")
    print(f" EJERCICIO 4: Recocido Simulado para TSP ({n_cities} ciudades)")
    print(f"{'='*60}")

    for sch in schedules:
        _, best_c, elapsed, c_hist, _ = simulated_annealing(dist_matrix, schedule=sch, max_iter=7000)
        results[sch] = {"cost": best_c, "time": elapsed, "history": c_hist}
        print(f"Esquema: {sch:<12} | Costo Final: {best_c:6.2f} | Tiempo: {elapsed:5.3f} s")
    print(f"{'='*60}\n")

    plt.figure(figsize=(7.5, 4.2))
    plt.plot(results["linear"]["history"], label="Lineal", color="#e41a1c", alpha=0.7)
    plt.plot(results["exponential"]["history"], label="Exponencial", color="#377eb8", alpha=0.9)
    plt.plot(results["logarithmic"]["history"], label="Logarítmico", color="#4daf4a", alpha=0.7)
    plt.title("Curvas de Convergencia de Recocido Simulado", fontsize=12)
    plt.xlabel("Iteración", fontsize=11)
    plt.ylabel("Costo del Tour Actual", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/sa_convergence.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6.5, 4.2))
    for sch, color in zip(schedules, ["#e41a1c", "#377eb8", "#4daf4a"]):
        plt.scatter(results[sch]["time"], results[sch]["cost"], color=color, s=120, label=sch.capitalize())
        plt.annotate(sch.capitalize(), (results[sch]["time"], results[sch]["cost"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=10)
    plt.title("Tiempo de Ejecución vs. Calidad de la Solución Final", fontsize=12)
    plt.xlabel("Tiempo de Cómputo (s)", fontsize=11)
    plt.ylabel("Costo del Tour (menor es mejor)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/sa_time_vs_quality.png", dpi=300)
    plt.close()
    print("-> Gráficas guardadas: plots/sa_convergence.png y plots/sa_time_vs_quality.png")

if __name__ == "__main__":
    main()
