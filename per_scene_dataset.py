import habitat
from collections import defaultdict
from omegaconf import DictConfig
from habitat.core.registry import registry
ObjectNavDataset = registry.get_dataset(name='ObjectNav-v1')


class PerSceneDataset(ObjectNavDataset):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        scenes = defaultdict(list)
        for ep in self.episodes:
            scenes[ep.scene_id].append(ep)
        filtered = []
        for eps in scenes.values():
            filtered.extend(eps[:config.episodes_per_scene])
        self.episodes = filtered
