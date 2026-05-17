import numpy as np


class Agent:
    def __init__(self, x, y, theta, controller_id):
        self.x = x
        self.y = y
        self.theta = theta
        self.controller_id = controller_id


class SwarmSimulator:
    """
    Tuned 2D swarm simulator.

    Adds simple embodiment through soft robot-robot collision handling.
    That helps prevent unrealistic overlap and makes local groups more stable.
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
        body_radius=0.10,
        collision_iterations=2,
    ):
        self.n_agents = n_agents
        self.W, self.H = arena_size
        self.dt = dt
        self.max_step = max_step
        self.neighbor_radius = neighbor_radius
        self.cue_field = cue_field if cue_field is not None else (lambda x, y: 0.0)
        self.controller = controller
        self.rng = np.random.default_rng(rng_seed)
        self.body_radius = body_radius
        self.collision_iterations = collision_iterations

        self.agents = []
        self._init_agents()

    def reset(self):
        self._init_agents()

    def get_positions(self):
        return np.array([[a.x, a.y] for a in self.agents])

    def step(self):
        positions = self.get_positions()
        neighbor_counts = self._compute_neighbor_counts(positions)
        cue_values = self._compute_cues(positions)

        controls = []
        for i, agent in enumerate(self.agents):
            obs = {
                "position": positions[i].copy(),
                "theta": agent.theta,
                "neighbor_count": neighbor_counts[i],
                "cue_value": cue_values[i],
            }
            ctrl = self.controller.step(i, obs, self.rng) if self.controller is not None else self._default_controller(obs)
            controls.append(ctrl)

        for agent, ctrl in zip(self.agents, controls):
            step_size = np.clip(ctrl.get("step_size", self.max_step), 0.0, self.max_step)
            dtheta = ctrl.get("turn_delta", 0.0)
            agent.theta = (agent.theta + dtheta) % (2 * np.pi)
            agent.x += step_size * np.cos(agent.theta)
            agent.y += step_size * np.sin(agent.theta)
            self._handle_walls(agent)

        if self.body_radius > 0.0 and self.collision_iterations > 0:
            self._resolve_robot_collisions()

    def _init_agents(self):
        self.agents = []
        margin = min(self.body_radius, 0.45 * min(self.W, self.H))
        for _ in range(self.n_agents):
            x = self.rng.uniform(margin, self.W - margin)
            y = self.rng.uniform(margin, self.H - margin)
            theta = self.rng.uniform(0, 2 * np.pi)
            self.agents.append(Agent(x, y, theta, controller_id=0))

    def _compute_neighbor_counts(self, positions):
        counts = np.zeros(self.n_agents, dtype=int)
        for i in range(self.n_agents):
            d = positions - positions[i]
            dist2 = np.sum(d * d, axis=1)
            mask = (dist2 > 0) & (dist2 <= self.neighbor_radius ** 2)
            counts[i] = np.sum(mask)
        return counts

    def _compute_cues(self, positions):
        cues = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            x, y = positions[i]
            cues[i] = self.cue_field(x, y)
        return cues

    def _resolve_robot_collisions(self):
        min_dist = 2.0 * self.body_radius
        for _ in range(self.collision_iterations):
            any_overlap = False
            for i in range(self.n_agents):
                ai = self.agents[i]
                for j in range(i + 1, self.n_agents):
                    aj = self.agents[j]
                    dx = aj.x - ai.x
                    dy = aj.y - ai.y
                    dist = float(np.hypot(dx, dy))
                    if dist >= min_dist:
                        continue

                    any_overlap = True
                    if dist < 1e-12:
                        angle = self.rng.uniform(0.0, 2.0 * np.pi)
                        ux = np.cos(angle)
                        uy = np.sin(angle)
                        overlap = min_dist
                    else:
                        ux = dx / dist
                        uy = dy / dist
                        overlap = min_dist - dist

                    shift = 0.5 * overlap
                    ai.x -= shift * ux
                    ai.y -= shift * uy
                    aj.x += shift * ux
                    aj.y += shift * uy
                    self._clip_inside(ai)
                    self._clip_inside(aj)

            if not any_overlap:
                break

    def _clip_inside(self, agent):
        agent.x = float(np.clip(agent.x, self.body_radius, self.W - self.body_radius))
        agent.y = float(np.clip(agent.y, self.body_radius, self.H - self.body_radius))

    def _handle_walls(self, agent):
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
        return {
            "step_size": self.max_step,
            "turn_delta": self.rng.normal(0.0, 0.3),
        }
