import os
import matplotlib.pyplot as plt

def knapsack_min_weight(items, capacity):
    sorted_items = sorted(items, key=lambda x: x[1])
    selected = []
    used_cap = 0
    total_val = 0
    for name, w, v in sorted_items:
        if used_cap + w <= capacity:
            selected.append((name, w, v))
            used_cap += w
            total_val += v
    return selected, used_cap, total_val

def knapsack_value_density(items, capacity):
    sorted_items = sorted(items, key=lambda x: x[2] / x[1], reverse=True)
    selected = []
    used_cap = 0
    total_val = 0
    for name, w, v in sorted_items:
        if used_cap + w <= capacity:
            selected.append((name, w, v))
            used_cap += w
            total_val += v
    return selected, used_cap, total_val

def main():
    os.makedirs("plots", exist_ok=True)
    capacity = 10
    items = [
        ("Obj 1", 2, 4),
        ("Obj 2", 3, 6),
        ("Obj 3", 4, 8),
        ("Obj 4", 7, 70)
    ]

    sel_min_w, cap_min_w, val_min_w = knapsack_min_weight(items, capacity)
    sel_dens, cap_dens, val_dens = knapsack_value_density(items, capacity)

    print(f"{'='*60}")
    print(f" EJERCICIO 2: Algoritmo Voraz de la Mochila (Capacidad W = {capacity})")
    print(f"{'='*60}")
    print("Instancia disponible:")
    for name, w, v in items:
        print(f"  * {name}: Peso = {w}, Valor = {v}, Densidad = {v/w:.2f}")

    print("\n--- Criterio 1: Priorizar Menor Peso ---")
    print(f"Objetos seleccionados: {[x[0] for x in sel_min_w]}")
    print(f"Capacidad usada: {cap_min_w} / {capacity} | Valor total: {val_min_w}")

    print("\n--- Criterio 2: Priorizar Valor / Peso (Densidad) ---")
    print(f"Objetos seleccionados: {[x[0] for x in sel_dens]}")
    print(f"Capacidad usada: {cap_dens} / {capacity} | Valor total: {val_dens}")
    print(f"{'='*60}\n")

    labels = ["Menor Peso", "Valor / Peso"]
    weights_used = [cap_min_w, cap_dens]
    total_values = [val_min_w, val_dens]

    x = [0, 1]
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax2 = ax1.twinx()

    b1 = ax1.bar([p - width/2 for p in x], weights_used, width=width, color="#4C72B0", label="Capacidad Usada")
    b2 = ax2.bar([p + width/2 for p in x], total_values, width=width, color="#55A868", label="Valor Total")

    ax1.set_ylabel("Capacidad Usada (unidades)", color="#4C72B0", fontsize=11)
    ax2.set_ylabel("Valor Total Obtenido", color="#55A868", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylim(0, 12)
    ax2.set_ylim(0, 100)

    for rect in b1:
        h = rect.get_height()
        ax1.annotate(f"{h}", (rect.get_x() + rect.get_width() / 2, h),
                     ha="center", va="bottom", fontsize=10)
    for rect in b2:
        h = rect.get_height()
        ax2.annotate(f"{h}", (rect.get_x() + rect.get_width() / 2, h),
                     ha="center", va="bottom", fontsize=10)

    plt.title("Comparativa: Menor Peso vs. Densidad de Valor", fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/knapsack_comparison.png", dpi=300)
    plt.close()
    print("-> Gráfica guardada en: plots/knapsack_comparison.png")

if __name__ == "__main__":
    main()
