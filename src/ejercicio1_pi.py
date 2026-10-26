import os
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

def count_inside_circle(n_samples):
    rng = np.random.default_rng()
    x = rng.random(n_samples)
    y = rng.random(n_samples)
    inside = np.sum(x**2 + y**2 <= 1.0)
    return inside

def estimate_pi(total_samples, num_processes):
    samples_per_worker = total_samples // num_processes
    tasks = [samples_per_worker] * num_processes

    start_time = time.perf_counter()
    with Pool(processes=num_processes) as pool:
        results = pool.map(count_inside_circle, tasks)
    total_inside = sum(results)
    elapsed_time = time.perf_counter() - start_time

    pi_hat = (4.0 * total_inside) / (samples_per_worker * num_processes)
    return pi_hat, elapsed_time

def main():
    os.makedirs("plots", exist_ok=True)
    total_samples = 40_000_000
    process_counts = [1, 2, 4, 8]
    times = []
    speedups = []

    print(f"{'='*55}")
    print(f" EJERCICIO 1: Aceleración Paralela (Monte Carlo Pi)")
    print(f" Muestras totales: {total_samples:,}")
    print(f"{'='*55}")

    for p in process_counts:
        pi_val, t = estimate_pi(total_samples, p)
        times.append(t)
        print(f"Procesos: {p:2d} | Tiempo: {t:6.3f} s | Estimado Pi: {pi_val:.6f}")

    t1 = times[0]
    for p, t in zip(process_counts, times):
        s = t1 / t
        speedups.append(s)

    print(f"{'-'*55}")
    print(f"{'Procesos (p)':<14} {'Tiempo (s)':<14} {'Speedup S(p)':<16} {'Eficiencia'}")
    print(f"{'-'*55}")
    for p, t, s in zip(process_counts, times, speedups):
        efficiency = (s / p) * 100
        print(f"{p:<14} {t:<14.3f} {s:<16.3f} {efficiency:.1f}%")
    print(f"{'='*55}\n")

    # Gráfica
    plt.figure(figsize=(7, 4.5))
    plt.plot(process_counts, process_counts, 'k--', label="Speedup Ideal (Lineal)", alpha=0.8)
    plt.plot(process_counts, speedups, 'o-', color="#1f77b4", linewidth=2, label="Speedup Empírico $S(p)$")
    plt.title("Aceleración Paralela vs. Número de Procesos", fontsize=12)
    plt.xlabel("Número de Procesos ($p$)", fontsize=11)
    plt.ylabel("Aceleración $S(p) = T_1 / T_p$", fontsize=11)
    plt.xticks(process_counts)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/speedup.png", dpi=300)
    plt.close()
    print("-> Gráfica guardada en: plots/speedup.png")

if __name__ == "__main__":
    main()
