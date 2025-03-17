from cv_utils.object_list import categories
import json
import os

GLEE_CONFIG_PATH = "./thirdparty/GLEE/configs/SwinL.yaml"
GLEE_CHECKPOINT_PATH = "./thirdparty/GLEE/GLEE_SwinL_Scaleup10m.pth"
DETECT_OBJECTS = [[cat['name'].lower()] for cat in categories]
INTEREST_OBJECTS = ['bed','chair','toilet','potted_plant','television_set', 'sofa', 'monitor', 'rocking chair', 'dog bed', 'bean bag chair', 'office chair', 'headboard', 'display', 'highchair', 'circular sofa', 'sofa', 'toilet seat', 'flowerpot', 'folding chair', 'flower stand', 'bed', 'recliner', 'couch', 'bed sheet', 'bulletin board', 'dried flowers', 'bar chair', 'folded chair', 'armchair', 'chair', 'computer chair', 'flower', 'l-shaped sofa', 'sofa set', 'flower vase', 'tv', 'dining chair', 'plant', 'desk chair', 'decorative plant', 'sofa chair', 'toilet', 'patio chair', 'lounge chair', 'bedframe']

PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
HABITAT_DIR = os.path.join(PARENT_DIR, "habitat-lab")
DATA_DIR = os.path.join(HABITAT_DIR, "data")

HSSD_SCENE_DATASET_PATH = os.path.join(DATA_DIR, "scene_datasets/hssd-hab")
semantic_config_path = os.path.join(HSSD_SCENE_DATASET_PATH, "semantics/hssd-hab_semantic_lexicon.json")
with open(semantic_config_path, "r") as f:
    semantic_config = json.load(f)
HSSD_TARGET_OBJECTS = {int(x['id']): x["name"] for x in semantic_config["classes"]}
HSSD_TARGET_OBJECTS[0] = 'unknown'
