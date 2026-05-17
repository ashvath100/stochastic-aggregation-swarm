import numpy as np


class CueFSMController:
    """
    Tuned cue controller.

    Main idea of this revision:
    - make nd=4 less sticky in the intermediate region
    - make level 3 more exploratory so robots cross toward the center faster
    - keep very strong retention only in levels 1 and 2
    """

    def __init__(self, max_step=0.1, n_levels=2):
        if n_levels not in (2, 3, 4):
            raise ValueError("n_levels must be one of: 2, 3, 4")
        self.max_step = max_step
        self.n_levels = n_levels
        self.agent_types = None

        # Base motion per cue level.
        # Level 1 = innermost, level nd = outermost.
        self.level_presets = {
            2: {
                1: (0.16, 2.00),
                2: (1.00, 0.05),
            },
            3: {
                1: (0.11, 2.10),
                2: (0.22, 1.75),
                3: (1.00, 0.05),
            },
            4: {
                1: (0.05, 2.35),
                2: (0.09, 2.00),
                # IMPORTANT: level 3 is now clearly exploratory,
                # so robots do not loiter too much in the mid ring.
                3: (0.88, 0.08),
                4: (1.00, 0.04),
            },
        }

        # Sticky trap motion.
        # Very strong only in the good inner regions.
        self.trap_presets = {
            2: {
                1: (0.07, 2.30),
                2: (0.55, 0.20),
            },
            3: {
                1: (0.05, 2.35),
                2: (0.10, 2.00),
                3: (0.55, 0.20),
            },
            4: {
                1: (0.025, 2.55),
                2: (0.05, 2.20),
                # defined for completeness, but level 3 is no longer held strongly
                3: (0.22, 1.10),
                4: (0.50, 0.18),
            },
        }

        # Sticky retention per level.
        # IMPORTANT CHANGE:
        # for nd=4 we removed sticky hold from level 3.
        self.sticky_holds = {
            2: {1: 22},
            3: {1: 30, 2: 10},
            4: {1: 52, 2: 26},
        }

        self.sticky_steps = {}
        self.last_level = {}

    def _cue_to_level(self, cue_value):
        edges = np.linspace(0.0, 1.0, self.n_levels + 1)
        level = np.searchsorted(edges[1:], cue_value, side="right") + 1
        return int(np.clip(level, 1, self.n_levels))

    def step(self, agent_id, obs, rng):
        cue = obs["cue_value"]
        level = self._cue_to_level(cue)
        prev_level = self.last_level.get(agent_id, level)
        sticky = self.sticky_steps.get(agent_id, 0)

        # If robot is in an inner good level, keep it there.
        if level in self.sticky_holds[self.n_levels]:
            sticky = max(sticky, self.sticky_holds[self.n_levels][level])

        # If robot has just moved inward, reinforce retention.
        if level < prev_level and level in self.sticky_holds[self.n_levels]:
            sticky = max(sticky, self.sticky_holds[self.n_levels][level] + 6)

        # If robot drifts outward while still sticky, keep a small amount of memory.
        elif level > prev_level and sticky > 0:
            sticky = max(sticky, 4)

        if sticky > 0:
            step_scale, turn_std = self.trap_presets[self.n_levels][level]
            sticky -= 1
        else:
            step_scale, turn_std = self.level_presets[self.n_levels][level]

        self.sticky_steps[agent_id] = sticky
        self.last_level[agent_id] = level

        return {
            "step_size": self.max_step * step_scale,
            "turn_delta": rng.normal(0.0, turn_std),
            "cue_level": level,
            "sticky_steps": sticky,
        }


class NeighborFSMController:
    """
    Tuned neighbour controller.

    Main changes:
    - earlier switch to cluster mode
    - strong hysteresis so clusters do not immediately dissolve
    - density-dependent slowdown, so large groups become much more stable
    - cluster mode still keeps a small amount of drift, so agents do not freeze
      too much after finding a group
    """

    def __init__(self, max_step=0.1, t0=2, t1=0, cluster_hold_steps=30):
        self.max_step = max_step
        self.t0 = t0
        self.t1 = t1
        self.cluster_hold_steps = cluster_hold_steps
        self.state = {}
        self.hold = {}
        self.agent_types = None

    def step(self, agent_id, obs, rng):
        n = obs["neighbor_count"]

        if agent_id not in self.state:
            self.state[agent_id] = "explore"
            self.hold[agent_id] = 0

        if self.state[agent_id] == "explore":
            if n >= self.t0:
                self.state[agent_id] = "cluster"
                self.hold[agent_id] = self.cluster_hold_steps
        else:
            if n >= self.t0:
                self.hold[agent_id] = self.cluster_hold_steps
            else:
                self.hold[agent_id] = max(0, self.hold[agent_id] - 1)

            if n <= self.t1 and self.hold[agent_id] == 0:
                self.state[agent_id] = "explore"

        if self.state[agent_id] == "explore":
            step_size = self.max_step
            turn_delta = rng.normal(0.0, 0.05)
        else:
            n_eff = min(max(n, self.t0), 8)
            density_ratio = (n_eff - self.t0) / max(1, 8 - self.t0)

            # Slightly more motion than before, but still strongly stable.
            step_scale = 0.22 - 0.13 * density_ratio
            turn_std = 1.15 + 0.70 * density_ratio

            if self.hold[agent_id] > 0:
                step_scale *= 0.90

            step_size = self.max_step * max(0.05, step_scale)
            turn_delta = rng.normal(0.0, turn_std)

        return {
            "step_size": step_size,
            "turn_delta": turn_delta,
            "fsm_state": self.state[agent_id],
            "hold_steps": self.hold[agent_id],
        }


class HeteroFSMController:
    def __init__(self, max_step=0.1, n_levels=2, cue_agents=10, neighbor_agents=15,
                 assignment_seed=0):
        self.max_step = max_step
        self.n_levels = n_levels
        self.cue_agents = cue_agents
        self.neighbor_agents = neighbor_agents
        self.n_agents = cue_agents + neighbor_agents

        self.cue_controller = CueFSMController(max_step=max_step, n_levels=n_levels)
        self.neigh_controller = NeighborFSMController(max_step=max_step)

        rng = np.random.default_rng(assignment_seed)
        agent_types = np.array(["cue"] * cue_agents + ["neighbor"] * neighbor_agents, dtype="U8")
        rng.shuffle(agent_types)
        self.agent_types = agent_types

    def step(self, agent_id, obs, rng):
        if self.agent_types[agent_id] == "cue":
            return self.cue_controller.step(agent_id, obs, rng)
        return self.neigh_controller.step(agent_id, obs, rng)
