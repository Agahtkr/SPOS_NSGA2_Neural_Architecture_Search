import matplotlib.pyplot as plt
import os


def plot_pareto_front(generation, pop_fitness, save_dir):
    f1_scores = [f[0] for f in pop_fitness]
    latencies = [f[1] for f in pop_fitness]

    plt.figure(figsize=(8, 6))
    plt.scatter(latencies, f1_scores, c='blue', alpha=0.6)
    plt.xlabel("Latency (ms) -> Lower is better")
    plt.ylabel("F1-Score -> Higher is better")
    plt.title(f"Pareto Front - Generation {generation}")
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"pareto_gen_{generation}.png"))
    plt.close()


def plot_evolution(history, save_dir):
    """Plots best F1 and minimum latency achieved per generation."""
    os.makedirs(save_dir, exist_ok=True)

    gens = [h["generation"] for h in history]
    best_f1 = [max(f for f, _ in h["metrics"]) for h in history]
    min_lat = [min(l for _, l in h["metrics"]) for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(gens, best_f1, 'g-o', label="Best F1")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("F1 Score", color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    ax2 = ax1.twinx()
    ax2.plot(gens, min_lat, 'r-s', label="Min Latency (ms)")
    ax2.set_ylabel("Latency (ms)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title("Evolution of Metrics Across Generations")
    fig.tight_layout()

    plt.savefig(os.path.join(save_dir, "evolution.png"), dpi=300)
    plt.close()