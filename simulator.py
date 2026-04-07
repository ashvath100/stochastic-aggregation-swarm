import numpy as np
import matplotlib.pyplot as plt

class Agent:
    """
    Simulator-side agent.
    Holds position / heading and delegates decisions to an external controller (FSM).
    """
    def __init__(self, x, y, theta, controller_id):
        self.x = x
        self.y = y
        self.theta = theta  # heading in radians
        self.controller_id = controller_id  # which FSM instance to use (if you have many types)


class SwarmSimulator:
    """
    2D swarm simulator with:
    - bounded rectangular arena
    - simple collision / wall handling (bounce)
    - local neighbor sensing
    - scalar cue field with quantization
    The controller/FSM is kept external and injected.
    """
    def __init__(
        self,
        n_agents=100,
        arena_size=(10.0, 10.0),
        dt=0.1,
        max_step=0.1,
        neighbor_radius=0.5,
        cue_field=None,
        controller=None,
        rng_seed=None,
    ):
        """
        Parameters
        ----------
        n_agents : int
            Number of agents.
        arena_size : (float, float)
            Width, height of arena (0..W, 0..H).
        dt : float
            Time step.
        max_step : float
            Maximum step length per time step (used as scale).
        neighbor_radius : float
            Radius for neighbor sensing.
        cue_field : callable or None
            Function f(x, y) -> scalar cue value (float). If None, cue=0 everywhere.
        controller : object
            Must implement method:
                step(agent_id, obs_dict, rng) -> dict with keys:
                    "step_size": float
                    "turn_delta": float (radians)
        rng_seed : int or None
            Random seed.
        """
        self.n_agents = n_agents
        self.W, self.H = arena_size
        self.dt = dt
        self.max_step = max_step
        self.neighbor_radius = neighbor_radius
        self.cue_field = cue_field if cue_field is not None else (lambda x, y: 0.0)
        self.controller = controller
        self.rng = np.random.default_rng(rng_seed)

        self.agents = []
        self._init_agents()

    # -------------------- public API -------------------- #
    def reset(self):
        self._init_agents()

    def get_positions(self):
        """Return Nx2 array of agent positions."""
        return np.array([[a.x, a.y] for a in self.agents])

    def step(self):
        """Advance the simulation by one time step."""
        positions = self.get_positions()

        # Precompute neighbor info for speed
        neighbor_counts = self._compute_neighbor_counts(positions)
        cue_values = self._compute_cues(positions)

        # Per-agent update
        for i, agent in enumerate(self.agents):
            obs = {
                "position": positions[i].copy(),
                "theta": agent.theta,
                "neighbor_count": neighbor_counts[i],
                "cue_value": cue_values[i],
            }

            # Query external controller/FSM
            if self.controller is None:
                # fallback: simple correlated random walk
                ctrl = self._default_controller(obs)
            else:
                ctrl = self.controller.step(i, obs, self.rng)

            step_size = np.clip(ctrl.get("step_size", self.max_step),
                                0.0, self.max_step)
            dtheta = ctrl.get("turn_delta", 0.0)

            # Update heading
            agent.theta = (agent.theta + dtheta) % (2 * np.pi)

            # Update position (forward motion)
            dx = step_size * np.cos(agent.theta)
            dy = step_size * np.sin(agent.theta)
            agent.x += dx
            agent.y += dy

            # Handle walls (simple elastic bounce)
            self._handle_walls(agent)

    # -------------------- internals -------------------- #
    def _init_agents(self):
        self.agents = []
        for _ in range(self.n_agents):
            x = self.rng.uniform(0, self.W)
            y = self.rng.uniform(0, self.H)
            theta = self.rng.uniform(0, 2 * np.pi)
            self.agents.append(Agent(x, y, theta, controller_id=0))

    def _compute_neighbor_counts(self, positions):
        counts = np.zeros(self.n_agents, dtype=int)
        # naive O(N^2); fine for moderate N
        for i in range(self.n_agents):
            d = positions - positions[i]
            dist2 = np.sum(d * d, axis=1)
            # exclude self (dist=0)
            mask = (dist2 > 0) & (dist2 <= self.neighbor_radius ** 2)
            counts[i] = np.sum(mask)
        return counts

    def _compute_cues(self, positions):
        cues = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            x, y = positions[i]
            cues[i] = self.cue_field(x, y)
        return cues

    def _handle_walls(self, agent):
        # Reflect at boundaries and flip heading accordingly
        bounced = False
        if agent.x < 0:
            agent.x = -agent.x
            agent.theta = np.pi - agent.theta
            bounced = True
        elif agent.x > self.W:
            agent.x = 2 * self.W - agent.x
            agent.theta = np.pi - agent.theta
            bounced = True

        if agent.y < 0:
            agent.y = -agent.y
            agent.theta = -agent.theta
            bounced = True
        elif agent.y > self.H:
            agent.y = 2 * self.H - agent.y
            agent.theta = -agent.theta
            bounced = True

        if bounced:
            agent.theta %= 2 * np.pi

    def _default_controller(self, obs):
        # Simple correlated random walk with small random turn
        turn_std = 0.3
        dtheta = self.rng.normal(0.0, turn_std)
        return {
            "step_size": self.max_step,
            "turn_delta": dtheta,
        }


# -------------------- simple visual test -------------------- #
if __name__ == "__main__":
    # Example cue field: hill in the center
    def cue(x, y):
        cx, cy = 5.0, 5.0
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        return np.exp(-r2 / 4.0)  # peak at center

    # Dummy controller that biases motion toward higher cue
    class DemoCueController:
        def __init__(self, arena_size, max_step):
            self.W, self.H = arena_size
            self.max_step = max_step

        def step(self, agent_id, obs, rng):
            x, y = obs["position"]
            theta = obs["theta"]
            # Estimate gradient by finite differences
            eps = 0.05
            c0 = cue(x, y)
            cx = cue(x + eps, y) - c0
            cy = cue(x, y + eps) - c0
            # Desired heading pointing uphill
            desired_theta = np.arctan2(cy, cx) if (cx != 0 or cy != 0) else theta
            # Turn a bit toward desired heading
            angle_diff = np.arctan2(
                np.sin(desired_theta - theta),
                np.cos(desired_theta - theta),
            )
            dtheta = 0.3 * angle_diff + rng.normal(0.0, 0.1)
            step_size = self.max_step
            return {"step_size": step_size, "turn_delta": dtheta}

    controller = DemoCueController(arena_size=(10.0, 10.0), max_step=0.1)

    sim = SwarmSimulator(
        n_agents=200,
        arena_size=(10.0, 10.0),
        dt=0.1,
        max_step=0.1,
        neighbor_radius=0.5,
        cue_field=cue,
        controller=controller,
        rng_seed=42,
    )

    #plt.ion()
    fig, ax = plt.subplots(figsize=(5, 5))

    for t in range(500):
        sim.step()
        pos = sim.get_positions()
        ax.clear()
        ax.set_xlim(0, sim.W)
        ax.set_ylim(0, sim.H)
        ax.scatter(pos[:, 0], pos[:, 1], s=10, alpha=0.7)
        ax.set_title(f"t = {t}")
        #plt.pause(0.01)

    #plt.ioff()
    #plt.show()
