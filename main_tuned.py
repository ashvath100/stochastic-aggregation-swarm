from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from fsm_controller_tuned import CueFSMController, HeteroFSMController, NeighborFSMController
from results_utils import (
    RunSummary,
    compute_cluster_metric,
    create_output_dir,
    infer_controller_name,
    paper_cue_field,
    plot_final_snapshot,
    plot_initial_final_snapshot,
    plot_metrics_over_time,
    quantize_cue,
    save_summary_csv,
    save_timeseries_csv,
    write_run_info,
)
from simulator_tuned import SwarmSimulator


@dataclass
class SingleRunResult:
    summary: RunSummary
    g_series: np.ndarray
    c_series: np.ndarray
    initial_positions: np.ndarray
    final_positions: np.ndarray
    agent_types: np.ndarray


@dataclass
class SimulationConfig:
    cue_agents: int = 25
    neighbor_agents: int = 0
    cue_levels: int = 2
    timesteps: int = 3600
    radius: float = 2.0
    seed: int = 0
    max_step: float = 0.1
    arena_width: float = 10.0
    arena_height: float = 10.0
    body_radius: float = 0.10
    draw_every: int = 10
    output_dir: str = "results"
    animate: bool = True
    save_outputs: bool = True


def build_controller(cue_agents: int, neighbor_agents: int, cue_levels: int,
                     max_step: float, seed: int):
    total_agents = cue_agents + neighbor_agents
    if total_agents <= 0:
        raise ValueError("You must have at least one agent in total.")

    if cue_agents > 0 and neighbor_agents > 0:
        controller = HeteroFSMController(
            max_step=max_step,
            n_levels=cue_levels,
            cue_agents=cue_agents,
            neighbor_agents=neighbor_agents,
            assignment_seed=seed,
        )
        agent_types = controller.agent_types
    elif cue_agents > 0:
        controller = CueFSMController(max_step=max_step, n_levels=cue_levels)
        agent_types = np.array(["cue"] * total_agents, dtype="U8")
        controller.agent_types = agent_types
    else:
        controller = NeighborFSMController(max_step=max_step)
        agent_types = np.array(["neighbor"] * total_agents, dtype="U8")
        controller.agent_types = agent_types

    return controller, agent_types


def draw_scene(ax, positions: np.ndarray, agent_types: np.ndarray,
               arena_width: float, arena_height: float, cue_levels: int,
               step_idx: int, g_now: float, c_now: float) -> None:
    ax.clear()
    grid_x = np.linspace(0.0, arena_width, 250)
    grid_y = np.linspace(0.0, arena_height, 250)
    xx, yy = np.meshgrid(grid_x, grid_y)
    cue = paper_cue_field(xx, yy, arena_width=arena_width, arena_height=arena_height)
    quant = quantize_cue(cue, cue_levels)
    ax.contourf(xx, yy, quant, levels=np.arange(0.5, cue_levels + 1.5, 1.0), alpha=0.22)
    ax.contour(xx, yy, quant, levels=np.arange(1.5, cue_levels + 0.5, 1.0), linewidths=0.8, alpha=0.5)

    cue_mask = agent_types == "cue"
    nbr_mask = agent_types == "neighbor"
    if np.any(cue_mask):
        ax.scatter(positions[cue_mask, 0], positions[cue_mask, 1], s=26, label="cue agents")
    if np.any(nbr_mask):
        ax.scatter(positions[nbr_mask, 0], positions[nbr_mask, 1], s=26, marker="s", label="neighbor agents")

    ax.set_xlim(0.0, arena_width)
    ax.set_ylim(0.0, arena_height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"step={step_idx} | G(t)={g_now:.3f} | C(t)={c_now:.3f} | cue levels={cue_levels}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")


def run_single_simulation(config: SimulationConfig, output_dir_override: Optional[str] = None) -> SingleRunResult:
    total_agents = config.cue_agents + config.neighbor_agents
    controller, agent_types = build_controller(config.cue_agents, config.neighbor_agents, config.cue_levels, config.max_step, config.seed)

    output_dir = output_dir_override
    if output_dir is None and config.save_outputs:
        output_dir = create_output_dir(
            base_dir=config.output_dir,
            cue_agents=config.cue_agents,
            neighbor_agents=config.neighbor_agents,
            cue_levels=config.cue_levels,
            timesteps=config.timesteps,
            seed=config.seed,
        )

    sim = SwarmSimulator(
        n_agents=total_agents,
        arena_size=(config.arena_width, config.arena_height),
        dt=0.1,
        max_step=config.max_step,
        neighbor_radius=config.radius,
        cue_field=lambda x, y: paper_cue_field(x, y, arena_width=config.arena_width, arena_height=config.arena_height),
        controller=controller,
        rng_seed=config.seed,
        body_radius=config.body_radius,
    )

    initial_positions = sim.get_positions().copy()
    g_series = np.zeros(config.timesteps, dtype=float)
    c_series = np.zeros(config.timesteps, dtype=float)

    fig = ax = None
    if config.animate:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.show(block=False)

    for t in range(config.timesteps):
        sim.step()
        positions = sim.get_positions()
        g_now = float(np.mean(paper_cue_field(positions[:, 0], positions[:, 1], arena_width=config.arena_width, arena_height=config.arena_height)))
        c_now = float(compute_cluster_metric(positions, config.radius))
        g_series[t] = g_now
        c_series[t] = c_now

        if config.animate and (t % config.draw_every == 0 or t == config.timesteps - 1):
            draw_scene(ax, positions, agent_types, config.arena_width, config.arena_height, config.cue_levels, t + 1, g_now, c_now)
            fig.canvas.draw_idle()
            plt.pause(0.001)

    final_positions = sim.get_positions().copy()
    g_paper = float(np.mean(g_series))
    c_final = float(c_series[-1])
    g_final = float(g_series[-1])
    tail_start = int(0.8 * config.timesteps)
    g_tail = float(np.mean(g_series[tail_start:]))
    output_dir = output_dir or ""

    summary = RunSummary(
        cue_agents=config.cue_agents,
        neighbor_agents=config.neighbor_agents,
        total_agents=total_agents,
        cue_levels=config.cue_levels,
        timesteps=config.timesteps,
        radius=config.radius,
        seed=config.seed,
        g_paper=g_paper,
        c_final=c_final,
        g_final=g_final,
        g_tail=g_tail,
        output_dir=output_dir,
    )

    if config.save_outputs and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_summary_csv(output_dir, summary)
        save_timeseries_csv(output_dir, range(1, config.timesteps + 1), g_series, c_series)
        plot_metrics_over_time(output_dir, g_series, c_series)
        plot_final_snapshot(output_dir, final_positions, agent_types, config.cue_levels, config.arena_width, config.arena_height)
        plot_initial_final_snapshot(output_dir, initial_positions, final_positions, agent_types, config.cue_levels, config.arena_width, config.arena_height)
        write_run_info(output_dir, summary)

    if config.animate:
        draw_scene(ax, final_positions, agent_types, config.arena_width, config.arena_height, config.cue_levels, config.timesteps, g_final, c_final)
        plt.ioff()
        plt.show()

    return SingleRunResult(summary, g_series, c_series, initial_positions, final_positions, agent_types)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one tuned swarm simulation with live animation and saved results.")
    parser.add_argument("--cue-agents", type=int, default=25)
    parser.add_argument("--neighbor-agents", type=int, default=0)
    parser.add_argument("--cue-levels", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("-t", "--timesteps", type=int, default=3600)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-step", type=float, default=0.1)
    parser.add_argument("--arena-width", type=float, default=10.0)
    parser.add_argument("--arena-height", type=float, default=10.0)
    parser.add_argument("--body-radius", type=float, default=0.10)
    parser.add_argument("--draw-every", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--no-animate", action="store_true")
    args = parser.parse_args()

    config = SimulationConfig(
        cue_agents=args.cue_agents,
        neighbor_agents=args.neighbor_agents,
        cue_levels=args.cue_levels,
        timesteps=args.timesteps,
        radius=args.radius,
        seed=args.seed,
        max_step=args.max_step,
        arena_width=args.arena_width,
        arena_height=args.arena_height,
        body_radius=args.body_radius,
        draw_every=args.draw_every,
        output_dir=args.output_dir,
        animate=not args.no_animate,
        save_outputs=True,
    )
    result = run_single_simulation(config)
    mode = infer_controller_name(args.cue_agents, args.neighbor_agents)
    print("\n===== RESULTS (single run, tuned version) =====")
    print(f"mode: {mode}")
    print(f"cue_agents: {args.cue_agents}")
    print(f"neighbor_agents: {args.neighbor_agents}")
    print(f"cue_levels (nd): {args.cue_levels}")
    print(f"G (lower is better): {result.summary.g_paper:.4f}")
    print(f"C (higher is better): {result.summary.c_final:.4f}")
    print(f"saved to: {os.path.abspath(result.summary.output_dir)}")
