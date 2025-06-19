import habitat
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
    HabitatSimSemanticSensorConfig
)
from habitat.config.default import get_agent_config
import os
from constants import HABITAT_DIR, DATA_DIR

HM3D_CONFIG_PATH = os.path.join(HABITAT_DIR, "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml")
HSSD_CONFIG_PATH = os.path.join(HABITAT_DIR, "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hssd-hab.yaml")
MP3D_CONFIG_PATH = os.path.join(HABITAT_DIR, "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml")
R2R_CONFIG_PATH = os.path.join(HABITAT_DIR, "habitat-lab/habitat/config/benchmark/nav/vln_r2r.yaml")

def hm3d_config(path:str=HM3D_CONFIG_PATH,stage:str='val',episodes=-1, max_episode_steps=500, version:str='v1'):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = os.path.join(HABITAT_DIR, habitat_config.habitat.dataset.scenes_dir)
        if version == 'v1':
            habitat_config.habitat.dataset.data_path = os.path.join(HABITAT_DIR, "data/datasets/objectnav/hm3d/v1/{split}/{split}.json.gz")
        if version == 'v2':
            habitat_config.habitat.dataset.data_path = os.path.join(HABITAT_DIR, "data/datasets/objectnav/hm3d/v2/{split}/{split}.json.gz")
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.environment.max_episode_steps = max_episode_steps
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=False,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 0.25
        agent_config = get_agent_config(sim_config=habitat_config.habitat.simulator)
        sensor_config = agent_config.sim_sensors.rgb_sensor
        agent_config.sim_sensors.update({
            "semantic_sensor": HabitatSimSemanticSensorConfig(
                height=sensor_config.height,
                width=sensor_config.width,
                hfov=sensor_config.hfov,
                position=sensor_config.position
            )
        })
        habitat_config.habitat.environment.iterator_options.update({
            "cycle": False,
            "shuffle": False,
            "group_by_scene": False,
            "max_scene_repeat_steps": -1,
            # "max_scene_repeat_episodes": 1
        })
    return habitat_config

def hssd_config(path:str=HSSD_CONFIG_PATH,stage:str='val',episodes=-1, max_episode_steps=500):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = os.path.join(HABITAT_DIR, habitat_config.habitat.dataset.scenes_dir)
        habitat_config.habitat.dataset.data_path = os.path.join(HABITAT_DIR, habitat_config.habitat.dataset.data_path)
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.environment.max_episode_steps = max_episode_steps
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=False,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth=0.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=10.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 0.2
        agent_config = get_agent_config(sim_config=habitat_config.habitat.simulator)
        sensor_config = agent_config.sim_sensors.rgb_sensor
        agent_config.sim_sensors.update({
            "semantic_sensor": HabitatSimSemanticSensorConfig(
                height=sensor_config.height,
                width=sensor_config.width,
                hfov=sensor_config.hfov,
                position=sensor_config.position
            )
        })
        habitat_config.habitat.environment.iterator_options.update({
            "cycle": False,
            "shuffle": False,
            "group_by_scene": False,
            "max_scene_repeat_steps": -1,
            # "max_scene_repeat_episodes": 1
        })
    return habitat_config
    
def mp3d_config(path:str=MP3D_CONFIG_PATH,stage:str='val',episodes=200, max_episode_steps=500):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = os.path.join(DATA_DIR, "scene_datasets")
        habitat_config.habitat.dataset.data_path = os.path.join(DATA_DIR, "datasets/objectnav/mp3d/v1/{split}/{split}.json.gz")
        habitat_config.habitat.simulator.scene_dataset = os.path.join(DATA_DIR, "scene_datasets/mp3d/mp3d.scene_dataset_config.json")
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.environment.max_episode_steps = max_episode_steps
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 0.25
    return habitat_config

def r2r_config(path:str=R2R_CONFIG_PATH,stage:str='val_seen',episodes=200, max_episode_steps=500):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = os.path.join(DATA_DIR, "scene_datasets")
        habitat_config.habitat.dataset.data_path = os.path.join(DATA_DIR, "datasets/vln/r2r/{split}/{split}.json.gz")
        habitat_config.habitat.simulator.scene_dataset = os.path.join(DATA_DIR, "scene_datasets/mp3d/mp3d.scene_dataset_config.json")
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.environment.max_episode_steps = max_episode_steps
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })  
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 0.25
    return habitat_config
