#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Implements evaluation of ``habitat.Agent`` inside ``habitat.Env``.
``habitat.Benchmark`` creates a ``habitat.Env`` which is specified through
the ``config_env`` parameter in constructor. The evaluation is task agnostic
and is implemented through metrics defined for ``habitat.EmbodiedTask``.
"""

import os
from collections import defaultdict
from typing import Dict, Optional

from tqdm import tqdm

from habitat.config.default import get_config
from habitat.core.agent import Agent
from habitat.core.env import Env
from habitat_sim.errors import GreedyFollowerError
from habitat.sims.habitat_simulator.actions import HabitatSimActions

import numpy as np


class Benchmark:
    r"""Benchmark for evaluating agents in environments."""

    def __init__(
        self, env: Optional[Env] = None, config_paths: Optional[str] = None, log_path: Optional[str] = None
    ) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        """
        if env is not None:
            self._env = env
        else:
            config_env = get_config(config_paths)
            self._env = Env(config=config_env)
        
        self.log_path = log_path


    def evaluate(
        self, agent: "Agent", name: str, num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)

        # Initialize log file
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        with open(os.path.join(self.log_path, f"{name}.txt"), "w") as log_file:
            log_file.write("")

        episode_index = 0

        pbar = tqdm(total=num_episodes)
        while episode_index < num_episodes:
            observations = self._env.reset()
            # Skip the first ? episodes
            # if episode_index < ?:
            #     episode_index += 1
            #     pbar.update(1)
            #     continue
            scene_id = self._env.current_episode.scene_id.split("/")[-1]
            episode_id = self._env.current_episode.episode_id
            agent.reset()

            steps = 0
            while not self._env.episode_over:
                action = agent.act(observations)
                observations = self._env.step(action)
                steps += 1

            metrics = self._env.get_metrics()
            # If SPL is NaN, set it zero
            if np.isnan(metrics["spl"]):
                metrics["spl"] = 0.0
            with open(os.path.join(self.log_path, f"{name}.txt"), "a") as log_file:
                log_file.write(
                    f"index={episode_index} "
                    f"scene_id={scene_id} "
                    f"episode_id={episode_id} "
                    f"success={metrics['success']} " 
                    f"spl={metrics['spl']} "
                    f"steps={steps}\n"
                )
            for m, v in metrics.items():
                if not isinstance(v, dict):
                    agg_metrics[m] += v
            agg_metrics['steps'] += steps
            episode_index += 1
            pbar.update(1)

        avg_metrics = {k: v / num_episodes for k, v in agg_metrics.items()}

        return avg_metrics
