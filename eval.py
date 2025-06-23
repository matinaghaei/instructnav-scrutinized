import dotenv
dotenv.load_dotenv()

import habitat
import os
import argparse
from config_utils import hm3d_config,mp3d_config,hssd_config
from mapping_utils.transform import habitat_camera_intrinsic
from mapper import Instruct_Mapper
from objnav_agent import HM3D_Objnav_Agent
from habitat.datasets import make_dataset
from benchmark import Benchmark
from constants import HABITAT_DIR
from constants import HSSD_TARGET_OBJECTS
from functools import partial
from habitat.core.registry import registry
from per_scene_dataset import PerSceneDataset
registry.register_dataset(PerSceneDataset, name="PerSceneDataset")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes",type=int,default=-1)
    parser.add_argument("--dataset",type=str,default='hm3dv1', choices=['hm3dv1', 'hm3dv2', 'hssd', 'mp3d'])
    parser.add_argument("--mapper_resolution",type=float,default=0.05)
    parser.add_argument("--path_resolution",type=float,default=0.2)
    parser.add_argument("--path_scale",type=int,default=5)
    parser.add_argument("--split",type=str,default='val')
    parser.add_argument("--agent",type=str,default='default')
    parser.add_argument("--track_target_only",action='store_true')
    parser.add_argument("--max_episode_steps",type=int,default=500)
    parser.add_argument("--snap_point",action='store_true')
    parser.add_argument("--episodes_per_scene",type=int,default=-1)
    return parser.parse_known_args()[0]

if __name__ == "__main__":
    args = get_args()
    dataset_config_func = {
        'hm3dv1': partial(hm3d_config, version='v1'),
        'hm3dv2': partial(hm3d_config, version='v2'),
        'hssd': hssd_config,
        'mp3d': mp3d_config
    }
    habitat_config = dataset_config_func[args.dataset](
        stage=args.split,
        episodes=args.eval_episodes,
        max_episode_steps=args.max_episode_steps,
        episodes_per_scene=args.episodes_per_scene
    )
    dataset = make_dataset(id_dataset=habitat_config.habitat.dataset.type, config=habitat_config.habitat.dataset)
    for episode in dataset.episodes:
        episode.scene_dataset_config = os.path.join(HABITAT_DIR, episode.scene_dataset_config)
    habitat_env = habitat.Env(config=habitat_config, dataset=dataset)
    habitat_mapper = Instruct_Mapper(habitat_camera_intrinsic(habitat_config),
                                    pcd_resolution=args.mapper_resolution,
                                    grid_resolution=args.path_resolution,
                                    grid_size=args.path_scale,
                                    gt_seg=True)
    habitat_agent = HM3D_Objnav_Agent(habitat_env,habitat_mapper,chainon_mode=args.agent,args=args)
    benchmark = Benchmark(habitat_env, log_path=f"logs/{args.dataset}_{args.split}")
    habitat.logger.info(benchmark.evaluate(habitat_agent, name=f"{args.agent}_{args.max_episode_steps}" + ("_snap_point" if args.snap_point else "")))
