import argparse
from simulator import SwarmSimulator
from fsm_controller_template import CueFSMController, NeighborFSMController, HeteroFSMController
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import networkx as nx


def cue_field(x, y):
    # raw field: center = high (~1), edges = low (~0)
    return np.exp(-((x - 5)**2 + (y - 5)**2) / 4.0)


def compute_cluster_metric(pos, radius=0.5):
    S = len(pos)
    D = squareform(pdist(pos))
    adj = (D < radius).astype(int)
    np.fill_diagonal(adj, 0)
    G = nx.from_numpy_array(adj)
    largest_cluster = max((len(c) for c in nx.connected_components(G)), default=0)
    return largest_cluster / S


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-controller", type=str, default="hetero",
                        choices=["cue", "neighbor", "hetero"])
    parser.add_argument("-n_agents", type=int, default=25)
    parser.add_argument("-cue_ratio", type=float, default=0.4)

    parser.add_argument("-t", "--timesteps", type=int, default=3600)

    parser.add_argument("-radius", type=float, default=2)

    args = parser.parse_args()

    # Controller selection
    if args.controller == "cue":
        controller = CueFSMController()
    elif args.controller == "neighbor":
        controller = NeighborFSMController()
    else:
        controller = HeteroFSMController(cue_ratio=args.cue_ratio,
                                         n_agents=args.n_agents)

    sim = SwarmSimulator(
        n_agents=args.n_agents,
        arena_size=(10, 10),
        dt=0.1,
        max_step=0.1,
        neighbor_radius=args.radius,
        cue_field=cue_field,
        controller=controller,
        rng_seed=0,
    )

    #plt.ion()
    fig, ax = plt.subplots(figsize=(5, 5))

    T = args.timesteps
    S = args.n_agents

    G_tail_accum = 0.0
    tail_start = int(0.8 * T)

    for t in range(T):
        sim.step()
        pos = sim.get_positions()

        if t >= tail_start:
            G_tail_accum += np.mean(cue_field(pos[:, 0], pos[:, 1]))

        ax.clear()
        ax.set_xlim(0, sim.W)
        ax.set_ylim(0, sim.H)
        ax.scatter(pos[:, 0], pos[:, 1], s=10)
        ax.set_title(f"t={t}")
        #plt.pause(0.001)

    final_pos = sim.get_positions()
    G_final = np.mean(cue_field(final_pos[:, 0], final_pos[:, 1]))
    G_tail = G_tail_accum / (T - tail_start)
    C_metric = compute_cluster_metric(final_pos, args.radius)

    print("\n===== RESULTS =====")
    print(f"G (final cue score): {G_final:.4f}")
    print(f"C (cluster metric): {C_metric:.4f}")