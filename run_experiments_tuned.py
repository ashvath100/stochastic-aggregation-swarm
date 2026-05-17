from __future__ import annotations

import argparse
import os

import numpy as np

from main_tuned import SimulationConfig, run_single_simulation
from results_utils import (
    BatchSummary,
    create_batch_output_dir,
    infer_controller_name,
    plot_batch_boxplots,
    plot_batch_median_time_series,
    plot_batch_pareto_point,
    save_batch_runs_csv,
    save_batch_summary_csv,
    save_median_timeseries_csv,
    write_batch_info,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the tuned configuration many times.")
    parser.add_argument("--cue-agents", type=int, default=10)
    parser.add_argument("--neighbor-agents", type=int, default=15)
    parser.add_argument("--cue-levels", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("-t", "--timesteps", type=int, default=3600)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--max-step", type=float, default=0.1)
    parser.add_argument("--arena-width", type=float, default=10.0)
    parser.add_argument("--arena-height", type=float, default=10.0)
    parser.add_argument("--body-radius", type=float, default=0.10)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--save-per-run", action="store_true")
    args = parser.parse_args()

    total_agents = args.cue_agents + args.neighbor_agents
    batch_dir = create_batch_output_dir(args.output_dir, args.cue_agents, args.neighbor_agents, args.cue_levels, args.timesteps, args.runs)
    summaries = []
    g_runs = np.zeros((args.runs, args.timesteps), dtype=float)
    c_runs = np.zeros((args.runs, args.timesteps), dtype=float)

    per_run_root = os.path.join(batch_dir, "per_run")
    if args.save_per_run:
        os.makedirs(per_run_root, exist_ok=True)

    for run_idx in range(args.runs):
        seed = args.seed_start + run_idx
        run_dir = os.path.join(per_run_root, f"run_{run_idx + 1:03d}_seed{seed}") if args.save_per_run else None
        config = SimulationConfig(
            cue_agents=args.cue_agents,
            neighbor_agents=args.neighbor_agents,
            cue_levels=args.cue_levels,
            timesteps=args.timesteps,
            radius=args.radius,
            seed=seed,
            max_step=args.max_step,
            arena_width=args.arena_width,
            arena_height=args.arena_height,
            body_radius=args.body_radius,
            draw_every=10,
            output_dir=batch_dir,
            animate=False,
            save_outputs=args.save_per_run,
        )
        result = run_single_simulation(config, output_dir_override=run_dir)
        summaries.append(result.summary)
        g_runs[run_idx, :] = result.g_series
        c_runs[run_idx, :] = result.c_series
        print(f"run {run_idx + 1}/{args.runs} | seed={seed} | G={result.summary.g_paper:.4f} | C={result.summary.c_final:.4f}")

    g_values = np.array([s.g_paper for s in summaries], dtype=float)
    c_values = np.array([s.c_final for s in summaries], dtype=float)
    g_final_values = np.array([s.g_final for s in summaries], dtype=float)
    g_tail_values = np.array([s.g_tail for s in summaries], dtype=float)

    batch_summary = BatchSummary(
        cue_agents=args.cue_agents,
        neighbor_agents=args.neighbor_agents,
        total_agents=total_agents,
        cue_levels=args.cue_levels,
        timesteps=args.timesteps,
        radius=args.radius,
        runs=args.runs,
        seed_start=args.seed_start,
        median_g_paper=float(np.median(g_values)),
        median_c_final=float(np.median(c_values)),
        mean_g_paper=float(np.mean(g_values)),
        mean_c_final=float(np.mean(c_values)),
        std_g_paper=float(np.std(g_values, ddof=0)),
        std_c_final=float(np.std(c_values, ddof=0)),
        median_g_final=float(np.median(g_final_values)),
        median_g_tail=float(np.median(g_tail_values)),
        output_dir=batch_dir,
    )

    save_batch_runs_csv(batch_dir, summaries)
    save_batch_summary_csv(batch_dir, batch_summary)
    save_median_timeseries_csv(batch_dir, g_runs, c_runs)
    plot_batch_boxplots(batch_dir, g_values, c_values)
    plot_batch_median_time_series(batch_dir, g_runs, c_runs)
    plot_batch_pareto_point(batch_dir, g_values, c_values)
    write_batch_info(batch_dir, batch_summary)

    mode = infer_controller_name(args.cue_agents, args.neighbor_agents)
    print("\n===== BATCH RESULTS (tuned version) =====")
    print(f"mode: {mode}")
    print(f"runs: {args.runs}")
    print(f"median G (lower is better): {batch_summary.median_g_paper:.4f}")
    print(f"median C (higher is better): {batch_summary.median_c_final:.4f}")
    print(f"saved to: {os.path.abspath(batch_dir)}")
