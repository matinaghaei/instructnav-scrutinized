import habitat
from collections import defaultdict
from omegaconf import DictConfig
from habitat.core.registry import registry
from config_utils import HM3D_CONFIG_PATH

config = habitat.get_config(HM3D_CONFIG_PATH)
_dataset = registry.get_dataset(
    name=config.habitat.dataset.type
)


class PerSceneDataset(_dataset):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        scenes = defaultdict(list)
        for ep in self.episodes:
            scenes[ep.scene_id].append(ep)
        filtered = []
        for eps in scenes.values():
            filtered.extend(eps[:config.episodes_per_scene])
        self.episodes = filtered
