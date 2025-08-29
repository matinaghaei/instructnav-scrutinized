import dotenv
dotenv.load_dotenv(override=True)

import habitat
import os
import shutil
import argparse
import csv
from tqdm import tqdm
from config_utils import hm3d_config,mp3d_config,hssd_config
from mapping_utils.transform import habitat_camera_intrinsic
from mapper import Instruct_Mapper
from objnav_agent import HM3D_Objnav_Agent
from habitat.datasets import make_dataset
from habitat.utils.visualizations.utils import images_to_video
from utils import generate_image, draw_on_grid, dilate
import cv2
from copy import deepcopy
from constants import HABITAT_DIR
from constants import HSSD_TARGET_OBJECTS
from functools import partial
from habitat.core.registry import registry
from per_scene_dataset import PerSceneDataset
registry.register_dataset(PerSceneDataset, name="PerSceneDataset")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

# def write_metrics(metrics,path="objnav_hm3d.csv"):
#     with open(path, mode="w", newline="") as csv_file:
#         fieldnames = metrics[0].keys()
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(metrics)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes",type=int,default=-1)
    parser.add_argument("--dataset",type=str,default='hm3dv1',choices=['hm3dv1', 'hm3dv2', 'hssd', 'mp3d'])
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
                                    visualize_seg=True,
                                    visualize_aff_maps=True)
    habitat_agent = HM3D_Objnav_Agent(habitat_env,habitat_mapper,chainon_mode=args.agent,args=args,debug=True)
    evaluation_metrics = []
    for i in tqdm(range(habitat_env.number_of_episodes)):
        observations = habitat_env.reset()
        habitat_agent.reset()
        dataset_name = f"{args.dataset}_{args.split}_{len(habitat_env.episodes)}"
        algorithm_name = f"{args.agent}_{args.max_episode_steps}" + ("_snap_point" if args.snap_point else "")
        dirname = os.path.join("images", dataset_name, algorithm_name, "%02d" % i)
        os.makedirs(dirname, exist_ok=True)
        images = []
        obsdir = os.path.join(dirname, "observation.png")
        while not habitat_env.episode_over:
            action = habitat_agent.act(observations)
            info = deepcopy(habitat_env.get_metrics())
            frontier_map_coords, frontier_map_centers = habitat_mapper.get_frontier_map()
            frontier_map = draw_on_grid(info["top_down_map"]["map"], frontier_map_coords)
            centers_map = draw_on_grid(info["top_down_map"]["map"], frontier_map_centers)
            info["top_down_map"]["map"] += 10 * dilate(frontier_map, radius=2) + 100 * dilate(centers_map, radius=5)
            # pcd = habitat_mapper.navigable_pcd
            # world_points = habitat_mapper.transform_pcd_to_world(pcd)
            # map_coords = habitat_mapper.project_world_to_map(world_points)
            # pcd_map = draw_on_grid(info["top_down_map"]["map"], map_coords)
            # info["top_down_map"]["map"] += 10 * dilate(pcd_map, radius=2)
            output_im = generate_image(habitat_mapper.segmentation, info)
            images.append(output_im)
            observations = habitat_env.step(action)
            cv2.imwrite(obsdir, output_im)
            for k, v in habitat_mapper.affordance_colormaps.items():
                cv2.imwrite(os.path.join(dirname, k + ".png"), v)
        images_to_video(images, dirname, "trajectory")
        habitat_agent.save_trajectory(os.path.join(dirname, ""))
        # evaluation_metrics.append({'success':habitat_agent.metrics['success'],
        #                         'spl':habitat_agent.metrics['spl'],
        #                         'distance_to_goal':habitat_agent.metrics['distance_to_goal'],
        #                         'object_goal':habitat_agent.instruct_goal})
        # write_metrics(evaluation_metrics)
