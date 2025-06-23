from habitat.core.dataset import Dataset
from collections import defaultdict
from habitat.core.dataset import EpisodeIterator, Episode


class PerSceneDataset(Dataset):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.episodes_per_scene = config.episodes_per_scene
    def get_episode_iterator(self, *args, **kwargs):
        return LimitingEpisodeIterator(
            episodes=self.episodes,
            episodes_per_scene=self.episodes_per_scene,
            *args,
            **kwargs,
        )


class LimitingEpisodeIterator(EpisodeIterator):
    """EpisodeIterator that caps the number of episodes per scene."""
    def __init__(
        self,
        episodes: list[Episode],
        episodes_per_scene: int,
        *args,
        **kwargs
    ):
        # 1. Group episodes by scene_id
        scenes = defaultdict(list)
        for ep in episodes:
            scenes[ep.scene_id].append(ep)
        # 2. Sample up to episodes_per_scene per scene
        filtered_eps = []
        for eps in scenes.values():
            filtered_eps.extend(eps[:episodes_per_scene])  # or use random.sample(eps, episodes_per_scene)
        # 3. Initialize the base iterator with filtered episodes
        super().__init__(
            filtered_eps,
            *args,
            **kwargs
        )
