from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform


@dataclass
class RunSummary:
    cue_agents: int
    neighbor_agents: int
    total_agents: int
    cue_levels: int
    timesteps: int
    radius: float
    seed: int
    g_paper: float
    c_final: float
    g_final: float
    g_tail: float
    output_dir: str


@dataclass
class BatchSummary:
    cue_agents: int
    neighbor_agents: int
    total_agents: int
    cue_levels: int
    timesteps: int
    radius: float
    runs: int
    seed_start: int
    median_g_paper: float
    median_c_final: float
    mean_g_paper: float
    mean_c_final: float
    std_g_paper: float
    std_c_final: float
    median_g_final: float
    median_g_tail: float
    output_dir: str


def paper_cue_field(x: np.ndarray | float, y: np.ndarray | float,
                    arena_width: float = 10.0,
                    arena_height: float = 10.0) -> np.ndarray | float:
    """
    Paper-like cue field:
    - minimum in the arena center
    - increases radially toward the borders
    - clipped to [0, 1]
    """
    cx = arena_width / 2.0
    cy = arena_height / 2.0
    r = np.sqrt((np.asarray(x) - cx) ** 2 + (np.asarray(y) - cy) ** 2)
    max_r = min(arena_width, arena_height) / 2.0
    return np.clip(r / max_r, 0.0, 1.0)


def quantize_cue(cue_value: float | np.ndarray, n_levels: int) -> np.ndarray:
    cue_arr = np.asarray(cue_value)
    edges = np.linspace(0.0, 1.0, n_levels + 1)
    levels = np.searchsorted(edges[1:], cue_arr, side="right") + 1
    return np.clip(levels, 1, n_levels)


def compute_cluster_metric(positions: np.ndarray, radius: float) -> float:
    n_agents = len(positions)
    if n_agents <= 1:
        return 0.0

    dist = squareform(pdist(positions))
    adjacency = (dist <= radius).astype(int)
    np.fill_diagonal(adjacency, 0)
    graph = nx.from_numpy_array(adjacency)
    largest_cluster = max((len(component) for component in nx.connected_components(graph)), default=0)
    return largest_cluster / n_agents


def infer_controller_name(cue_agents: int, neighbor_agents: int) -> str:
    if cue_agents > 0 and neighbor_agents > 0:
        return "hetero"
    if cue_agents > 0:
        return "cue"
    return "neighbor"


def create_output_dir(base_dir: str, cue_agents: int, neighbor_agents: int,
                      cue_levels: int, timesteps: int, seed: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = (
        f"run_{stamp}_cue{cue_agents}_nbr{neighbor_agents}"
        f"_L{cue_levels}_T{timesteps}_seed{seed}"
    )
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    return path


def create_batch_output_dir(base_dir: str, cue_agents: int, neighbor_agents: int,
                            cue_levels: int, timesteps: int, runs: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = (
        f"batch_{stamp}_cue{cue_agents}_nbr{neighbor_agents}"
        f"_L{cue_levels}_T{timesteps}_R{runs}"
    )
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    return path


def save_timeseries_csv(output_dir: str, timesteps: Iterable[int],
                        g_series: Iterable[float], c_series: Iterable[float]) -> str:
    path = os.path.join(output_dir, "timeseries_metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "G_instant", "C_instant"])
        for t, g_val, c_val in zip(timesteps, g_series, c_series):
            writer.writerow([t, f"{g_val:.8f}", f"{c_val:.8f}"])
    return path


def save_summary_csv(output_dir: str, summary: RunSummary) -> str:
    path = os.path.join(output_dir, "summary_metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cue_agents",
            "neighbor_agents",
            "total_agents",
            "cue_levels",
            "timesteps",
            "radius",
            "seed",
            "G_paper",
            "C_final",
            "G_final",
            "G_tail",
        ])
        writer.writerow([
            summary.cue_agents,
            summary.neighbor_agents,
            summary.total_agents,
            summary.cue_levels,
            summary.timesteps,
            f"{summary.radius:.6f}",
            summary.seed,
            f"{summary.g_paper:.8f}",
            f"{summary.c_final:.8f}",
            f"{summary.g_final:.8f}",
            f"{summary.g_tail:.8f}",
        ])
    return path


def save_batch_runs_csv(output_dir: str, summaries: list[RunSummary]) -> str:
    path = os.path.join(output_dir, "all_runs_metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_index",
            "seed",
            "cue_agents",
            "neighbor_agents",
            "total_agents",
            "cue_levels",
            "timesteps",
            "radius",
            "G_paper",
            "C_final",
            "G_final",
            "G_tail",
        ])
        for idx, summary in enumerate(summaries, start=1):
            writer.writerow([
                idx,
                summary.seed,
                summary.cue_agents,
                summary.neighbor_agents,
                summary.total_agents,
                summary.cue_levels,
                summary.timesteps,
                f"{summary.radius:.6f}",
                f"{summary.g_paper:.8f}",
                f"{summary.c_final:.8f}",
                f"{summary.g_final:.8f}",
                f"{summary.g_tail:.8f}",
            ])
    return path


def save_batch_summary_csv(output_dir: str, summary: BatchSummary) -> str:
    path = os.path.join(output_dir, "summary_statistics.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cue_agents",
            "neighbor_agents",
            "total_agents",
            "cue_levels",
            "timesteps",
            "radius",
            "runs",
            "seed_start",
            "median_G_paper",
            "median_C_final",
            "mean_G_paper",
            "mean_C_final",
            "std_G_paper",
            "std_C_final",
            "median_G_final",
            "median_G_tail",
        ])
        writer.writerow([
            summary.cue_agents,
            summary.neighbor_agents,
            summary.total_agents,
            summary.cue_levels,
            summary.timesteps,
            f"{summary.radius:.6f}",
            summary.runs,
            summary.seed_start,
            f"{summary.median_g_paper:.8f}",
            f"{summary.median_c_final:.8f}",
            f"{summary.mean_g_paper:.8f}",
            f"{summary.mean_c_final:.8f}",
            f"{summary.std_g_paper:.8f}",
            f"{summary.std_c_final:.8f}",
            f"{summary.median_g_final:.8f}",
            f"{summary.median_g_tail:.8f}",
        ])
    return path


def save_median_timeseries_csv(output_dir: str, g_runs: np.ndarray, c_runs: np.ndarray) -> str:
    path = os.path.join(output_dir, "median_timeseries.csv")
    g_median = np.median(g_runs, axis=0)
    c_median = np.median(c_runs, axis=0)
    g_q25 = np.quantile(g_runs, 0.25, axis=0)
    g_q75 = np.quantile(g_runs, 0.75, axis=0)
    c_q25 = np.quantile(c_runs, 0.25, axis=0)
    c_q75 = np.quantile(c_runs, 0.75, axis=0)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "G_median", "G_q25", "G_q75", "C_median", "C_q25", "C_q75"])
        for t in range(g_runs.shape[1]):
            writer.writerow([
                t + 1,
                f"{g_median[t]:.8f}",
                f"{g_q25[t]:.8f}",
                f"{g_q75[t]:.8f}",
                f"{c_median[t]:.8f}",
                f"{c_q25[t]:.8f}",
                f"{c_q75[t]:.8f}",
            ])
    return path


def _draw_quantized_background(ax: plt.Axes, arena_width: float, arena_height: float,
                               cue_levels: int) -> None:
    grid_x = np.linspace(0.0, arena_width, 300)
    grid_y = np.linspace(0.0, arena_height, 300)
    xx, yy = np.meshgrid(grid_x, grid_y)
    cue = paper_cue_field(xx, yy, arena_width=arena_width, arena_height=arena_height)
    quantized = quantize_cue(cue, cue_levels)
    ax.contourf(xx, yy, quantized, levels=np.arange(0.5, cue_levels + 1.5, 1.0), alpha=0.22)
    ax.contour(xx, yy, quantized, levels=np.arange(1.5, cue_levels + 0.5, 1.0), linewidths=0.8, alpha=0.5)


def plot_final_snapshot(output_dir: str, positions: np.ndarray, agent_types: np.ndarray,
                        cue_levels: int, arena_width: float, arena_height: float) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_quantized_background(ax, arena_width, arena_height, cue_levels)

    cue_mask = agent_types == "cue"
    nbr_mask = agent_types == "neighbor"

    if np.any(cue_mask):
        ax.scatter(positions[cue_mask, 0], positions[cue_mask, 1], s=22, label="cue agents")
    if np.any(nbr_mask):
        ax.scatter(positions[nbr_mask, 0], positions[nbr_mask, 1], s=22, marker="s", label="neighbor agents")

    ax.set_xlim(0.0, arena_width)
    ax.set_ylim(0.0, arena_height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Final swarm snapshot with quantized cue regions")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    fig.tight_layout()

    path = os.path.join(output_dir, "final_snapshot.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_metrics_over_time(output_dir: str, g_series: np.ndarray, c_series: np.ndarray) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    time = np.arange(1, len(g_series) + 1)

    axes[0].plot(time, g_series)
    axes[0].set_ylabel("G(t)")
    axes[0].set_title("Cue metric over time")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, c_series)
    axes[1].set_xlabel("time step")
    axes[1].set_ylabel("C(t)")
    axes[1].set_title("Cluster metric over time")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "metrics_over_time.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_initial_final_snapshot(output_dir: str, initial_positions: np.ndarray,
                                final_positions: np.ndarray, agent_types: np.ndarray,
                                cue_levels: int, arena_width: float,
                                arena_height: float) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = ["Initial positions", "Final positions"]
    data = [initial_positions, final_positions]

    for ax, title, positions in zip(axes, titles, data):
        _draw_quantized_background(ax, arena_width, arena_height, cue_levels)
        cue_mask = agent_types == "cue"
        nbr_mask = agent_types == "neighbor"
        if np.any(cue_mask):
            ax.scatter(positions[cue_mask, 0], positions[cue_mask, 1], s=20, label="cue")
        if np.any(nbr_mask):
            ax.scatter(positions[nbr_mask, 0], positions[nbr_mask, 1], s=20, marker="s", label="neighbor")
        ax.set_xlim(0.0, arena_width)
        ax.set_ylim(0.0, arena_height)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "initial_vs_final.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_batch_boxplots(output_dir: str, g_values: np.ndarray, c_values: np.ndarray) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    axes[0].boxplot(g_values, vert=True)
    axes[0].set_title("Boxplot of G over repeated runs")
    axes[0].set_ylabel("G (lower is better)")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot(c_values, vert=True)
    axes[1].set_title("Boxplot of C over repeated runs")
    axes[1].set_ylabel("C (higher is better)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "boxplots_G_C.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_batch_median_time_series(output_dir: str, g_runs: np.ndarray, c_runs: np.ndarray) -> str:
    time = np.arange(1, g_runs.shape[1] + 1)
    g_median = np.median(g_runs, axis=0)
    c_median = np.median(c_runs, axis=0)
    g_q25 = np.quantile(g_runs, 0.25, axis=0)
    g_q75 = np.quantile(g_runs, 0.75, axis=0)
    c_q25 = np.quantile(c_runs, 0.25, axis=0)
    c_q75 = np.quantile(c_runs, 0.75, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(time, g_median)
    axes[0].fill_between(time, g_q25, g_q75, alpha=0.25)
    axes[0].set_ylabel("G median")
    axes[0].set_title("Median cue metric over repeated runs")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, c_median)
    axes[1].fill_between(time, c_q25, c_q75, alpha=0.25)
    axes[1].set_xlabel("time step")
    axes[1].set_ylabel("C median")
    axes[1].set_title("Median cluster metric over repeated runs")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "median_metrics_over_time.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_batch_pareto_point(output_dir: str, g_values: np.ndarray, c_values: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(c_values, g_values, s=28, alpha=0.7, label="single runs")
    ax.scatter(np.median(c_values), np.median(g_values), s=90, marker="X", label="median point")
    ax.set_xlabel("C (higher is better)")
    ax.set_ylabel("G (lower is better)")
    ax.set_title("Repeated-run G vs C points")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    path = os.path.join(output_dir, "pareto_like_scatter.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_run_info(output_dir: str, summary: RunSummary) -> str:
    path = os.path.join(output_dir, "run_info.txt")
    controller_name = infer_controller_name(summary.cue_agents, summary.neighbor_agents)
    with open(path, "w", encoding="utf-8") as f:
        f.write("Paper-style swarm run\n")
        f.write(f"controller_mode: {controller_name}\n")
        f.write(f"cue_agents: {summary.cue_agents}\n")
        f.write(f"neighbor_agents: {summary.neighbor_agents}\n")
        f.write(f"total_agents: {summary.total_agents}\n")
        f.write(f"cue_levels: {summary.cue_levels}\n")
        f.write(f"timesteps: {summary.timesteps}\n")
        f.write(f"radius: {summary.radius}\n")
        f.write(f"seed: {summary.seed}\n")
        f.write("\nPaper metrics:\n")
        f.write(f"G_paper: {summary.g_paper:.8f}\n")
        f.write(f"C_final: {summary.c_final:.8f}\n")
        f.write("\nExtra diagnostics:\n")
        f.write(f"G_final: {summary.g_final:.8f}\n")
        f.write(f"G_tail: {summary.g_tail:.8f}\n")
    return path


def write_batch_info(output_dir: str, summary: BatchSummary) -> str:
    path = os.path.join(output_dir, "batch_info.txt")
    controller_name = infer_controller_name(summary.cue_agents, summary.neighbor_agents)
    with open(path, "w", encoding="utf-8") as f:
        f.write("Paper-style repeated-run experiment\n")
        f.write(f"controller_mode: {controller_name}\n")
        f.write(f"cue_agents: {summary.cue_agents}\n")
        f.write(f"neighbor_agents: {summary.neighbor_agents}\n")
        f.write(f"total_agents: {summary.total_agents}\n")
        f.write(f"cue_levels: {summary.cue_levels}\n")
        f.write(f"timesteps: {summary.timesteps}\n")
        f.write(f"radius: {summary.radius}\n")
        f.write(f"runs: {summary.runs}\n")
        f.write(f"seed_start: {summary.seed_start}\n")
        f.write("\nAggregate statistics across repeated runs:\n")
        f.write(f"median_G_paper: {summary.median_g_paper:.8f}\n")
        f.write(f"median_C_final: {summary.median_c_final:.8f}\n")
        f.write(f"mean_G_paper: {summary.mean_g_paper:.8f}\n")
        f.write(f"mean_C_final: {summary.mean_c_final:.8f}\n")
        f.write(f"std_G_paper: {summary.std_g_paper:.8f}\n")
        f.write(f"std_C_final: {summary.std_c_final:.8f}\n")
        f.write(f"median_G_final: {summary.median_g_final:.8f}\n")
        f.write(f"median_G_tail: {summary.median_g_tail:.8f}\n")
    return path
