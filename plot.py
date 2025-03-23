import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_ea_metrics(generations, fitness_history, makespan_history, conflict_history, distance_history):
    sns.set(style="whitegrid")

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plot Fitness
    sns.lineplot(x=generations, y=fitness_history, ax=axs[0, 0], marker='o', color='blue')
    axs[0, 0].set_title('Fitness Convergence')
    axs[0, 0].set_xlabel('Generations')
    axs[0, 0].set_ylabel('Fitness (Weighted)')

    # Plot Makespan
    sns.lineplot(x=generations, y=makespan_history, ax=axs[0, 1], marker='o', color='green')
    axs[0, 1].set_title('Makespan Evolution')
    axs[0, 1].set_xlabel('Generations')
    axs[0, 1].set_ylabel('Makespan (Steps)')

    # Plot Conflicts
    sns.lineplot(x=generations, y=conflict_history, ax=axs[1, 0], marker='o', color='red')
    axs[1, 0].set_title('Conflict Evolution')
    axs[1, 0].set_xlabel('Generations')
    axs[1, 0].set_ylabel('Number of Conflicts')

    # Plot Travel Distance
    sns.lineplot(x=generations, y=distance_history, ax=axs[1, 1], marker='o', color='purple')
    axs[1, 1].set_title('Travel Distance Evolution')
    axs[1, 1].set_xlabel('Generations')
    axs[1, 1].set_ylabel('Total Travel Distance (Steps)')

    plt.tight_layout()
    plt.show()


def plot_combined_metrics(generations, fitness_history, makespan_history, conflict_history, distance_history):
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.lineplot(x=generations, y=fitness_history, label='Fitness (Weighted)', marker='o', ax=ax)
    sns.lineplot(x=generations, y=makespan_history, label='Makespan', marker='s', ax=ax)
    sns.lineplot(x=generations, y=conflict_history, label='Conflicts', marker='X', ax=ax)
    sns.lineplot(x=generations, y=distance_history, label='Travel Distance', marker='^', ax=ax)

    ax.set_title('Comprehensive Evolution of EA Metrics')
    ax.set_xlabel('Generations')
    ax.set_ylabel('Metric Values')
    ax.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.show()