import numpy as np

class CueFSMController:
    def __init__(self, max_step=0.1):
        self.max_step = max_step

    def step(self, agent_id, obs, rng):
        cue = obs["cue_value"]

        if cue < 0.4:
            # STRONG exploration (outer)
            step_size = self.max_step
            turn_delta = rng.normal(0, 0.02)   # very straight

        else:
            # STRONG trapping (center)
            step_size = self.max_step * 0.3    # slow down
            turn_delta = rng.normal(0, 1.5)    # very random

        return {"step_size": step_size, "turn_delta": turn_delta}

class NeighborFSMController:
    """
    True 2-state FSM from paper:
    - n >= t0 → cluster (Brownian)
    - n < t1 → explore (straight)
    """
    def __init__(self, max_step=0.1):
        self.max_step = max_step

        # Critical thresholds (from paper-like behavior)
        self.t0 = 4   # enter cluster
        self.t1 = 1   # leave cluster

        # Internal state per agent
        self.state = {}

    def step(self, agent_id, obs, rng):
        n = obs["neighbor_count"]

        if agent_id not in self.state:
            self.state[agent_id] = "explore"

        # --- FSM transitions ---
        if self.state[agent_id] == "explore" and n >= self.t0:
            self.state[agent_id] = "cluster"

        elif self.state[agent_id] == "cluster" and n <= self.t1:
            self.state[agent_id] = "explore"

        # --- Behavior ---
        if self.state[agent_id] == "explore":
            # Lévy-like (straight)
            step_size = self.max_step
            turn_delta = rng.normal(0, 0.05)

        else:
            # Brownian (stay)
            step_size = self.max_step
            turn_delta = rng.normal(0, 1.0)

        return {"step_size": step_size, "turn_delta": turn_delta}
    
class HeteroFSMController:
    def __init__(self, max_step=0.1, cue_ratio=0.4, n_agents=200):
        self.max_step = max_step
        self.cue_ratio = cue_ratio

        self.cue_controller = CueFSMController(max_step=max_step)
        self.neigh_controller = NeighborFSMController(max_step=max_step)

        rng = np.random.default_rng(0)
        self.is_cue_type = rng.random(n_agents) < cue_ratio

    def step(self, agent_id, obs, rng):
        if self.is_cue_type[agent_id]:
            return self.cue_controller.step(agent_id, obs, rng)
        else:
            return self.neigh_controller.step(agent_id, obs, rng)

