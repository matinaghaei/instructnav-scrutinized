import numpy as np

from habitat_sim.nav import GreedyGeodesicFollower  # GGF
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower  # SPF
from habitat_sim.errors import GreedyFollowerError


def _sorted_target_viewpoints(env):
    """
    Collect all target viewpoints across all goals and return them
    sorted by geodesic distance from the agent's current position.
    """
    sim = env.sim
    episode = env.current_episode
    start_pos = sim.get_agent_state().position

    candidates = []

    for goal in episode.goals:
        for vp in goal.view_points:
            vp_pos = vp.agent_state.position
            d = sim.geodesic_distance(start_pos, vp_pos)
            if np.isfinite(d):
                candidates.append((d, vp_pos))

    # Sort by distance, closest first
    candidates.sort(key=lambda x: x[0])

    # Return only positions
    return [pos for _, pos in candidates]


def optimal_num_actions_ggf(env):
    """
    Optimal #actions (including STOP) to the closest *target viewpoint*
    using GreedyGeodesicFollower (sim-side oracle).

    If GreedyGeodesicFollower fails (GreedyFollowerError) for a viewpoint,
    we recover by trying the next-closest viewpoint, etc.

    Returns:
        int: number of actions including STOP, or
        None: if no reachable viewpoint.
    """
    sim = env.sim
    agent = sim.get_agent(0)

    candidate_viewpoints = _sorted_target_viewpoints(env)
    if not candidate_viewpoints:
        return None

    goal_radius = env._config.task.measurements.success.success_distance

    follower = GreedyGeodesicFollower(
        pathfinder=sim.pathfinder,
        agent=agent,
        goal_radius=goal_radius,
    )

    for goal_pos in candidate_viewpoints:
        try:
            # Plans purely in navmesh space; does *not* step the simulator.
            actions = follower.find_path(goal_pos)
        except GreedyFollowerError:
            # Could not generate a path to this viewpoint, try the next
            continue

        # actions list ends with None (STOP), so STOP is already counted.
        return len(actions)

    # All candidate viewpoints failed
    return None


def optimal_num_actions_spf(env):
    """
    Optimal #actions (including STOP) to the closest *target viewpoint*
    using ShortestPathFollower (lab-side oracle).

    We try viewpoints in increasing geodesic distance and recover from
    GreedyFollowerError by switching to the next viewpoint.

    Returns:
        int: number of actions including STOP, or
        None: if no reachable viewpoint.
    """
    sim = env.sim
    candidate_viewpoints = _sorted_target_viewpoints(env)
    if not candidate_viewpoints:
        return None

    goal_radius = env._config.task.measurements.success.success_distance
    start_state = sim.get_agent_state()

    for goal_pos in candidate_viewpoints:

        follower = ShortestPathFollower(
            sim=sim,
            goal_radius=goal_radius,
            return_one_hot=False,
            stop_on_error=False,
        )

        num_primitive = 0

        try:
            while True:
                action = follower.get_next_action(goal_pos)

                num_primitive += 1  # count this primitive action

                # If STOP is an explicit action id (commonly 0), don't step.
                if action == 0:
                    break

                sim.step(action)

        except GreedyFollowerError:
            # Failed to follow path to this viewpoint
            # Reset agent to original state for the next attempt
            sim.set_agent_state(start_state.position, start_state.rotation)
            continue

        return num_primitive

    # No viewpoint produced a valid plan
    return None
