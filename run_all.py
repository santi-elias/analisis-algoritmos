import subprocess
import sys

scripts = [
    ("Ejercicio 1: Aceleración Paralela", "src/ejercicio1_pi.py"),
    ("Ejercicio 2: Mochila Voraz", "src/ejercicio2_knapsack.py"),
    ("Ejercicio 3: Búsqueda Local 2-opt", "src/ejercicio3_tsp_2opt.py"),
    ("Ejercicio 4: Recocido Simulado", "src/ejercicio4_simulated_annealing.py"),
    ("Ejercicio 5: Aproximación por MST", "src/ejercicio5_mst_approx.py"),
]

def main():
    for title, script in scripts:
        print(f"\n{'='*60}")
        print(f" INICIANDO: {title}")
        print(f"{'='*60}\n")
        ret = subprocess.run([sys.executable, script])
        if ret.returncode != 0:
            print(f"Error al ejecutar {script}")
            break

if __name__ == "__main__":
    main()
