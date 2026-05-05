# InstructNav Scrutinized

This repository contains the scrutinized version of the original
[InstructNav](https://github.com/LYX0501/InstructNav) project used for the paper:

**When Engineering Outruns Intelligence: Rethinking Instruction-Guided Navigation**  
https://arxiv.org/abs/2507.20021

## Abstract

Recent ObjectNav systems credit large language models (LLMs) for sizable
zero-shot gains, yet it remains unclear how much comes from language versus
geometry. We revisit this question by re-evaluating an instruction-guided
pipeline, InstructNav, under a detector-controlled setting and introducing two
training-free variants that only alter the action value map: a geometry-only
Frontier Proximity Explorer (FPE) and a lightweight Semantic-Heuristic Frontier
(SHF) that polls the LLM with simple frontier votes. Across HM3D and MP3D, FPE
matches or exceeds the detector-controlled instruction follower while using no
API calls and running faster; SHF attains comparable accuracy with a smaller,
localized language prior. These results suggest that carefully engineered
frontier geometry accounts for much of the reported progress, and that language
is most reliable as a light heuristic rather than an end-to-end planner.

The code focuses on ObjectNav experiments in Habitat and includes debug and
benchmark entry points for comparing the original InstructNav-style agent with
frontier exploration and LLM-guided variants.

## Repository Layout

The code assumes this repository and `habitat-lab` are sibling directories:

```text
workspace/
  instructnav-scrutinized/
  habitat-lab/
```

`constants.py` resolves Habitat data through `../habitat-lab/data`, so keep this
layout unless you also update the paths there.

## Installation

Clone this repository and download the GLEE/CLIP weights used by the perception
stack:

```bash
git clone https://github.com/matinaghaei/instructnav-scrutinized.git
cd instructnav-scrutinized

huggingface-cli download openai/clip-vit-base-patch32 \
  --local-dir ./thirdparty/GLEE/clip-vit-base-patch32

huggingface-cli download Junfeng5/GLEE_demo GLEE_SwinL_Scaleup10m.pth \
  --repo-type space \
  --local-dir ./thirdparty/GLEE/
```

Create and activate the Conda environment:

```bash
conda env create -f environment_droplet.yml
conda activate instructnav
```

Install Habitat from the sibling checkout:

```bash
cd ..
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab

pip install -e habitat-lab
pip install -e habitat-baselines
pip install git+https://github.com/openai/CLIP.git
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Return to this repository before running experiments:

```bash
cd ../instructnav-scrutinized
```

## LLM API Setup

The scripts load `.env` automatically via `python-dotenv`. Create a `.env` file
in the repository root for agents that call an LLM, including `default`, `llm`,
and `lfg`.

For the current OpenAI client path used by this repo:

```bash
OPENAI_API_KEY=your_api_key_here
GPT_API_DEPLOY=gpt-4o
```

Use the model name you want in `GPT_API_DEPLOY`. If you adapt the commented
Azure OpenAI code in `llm_utils/gpt_request.py`, also provide the corresponding
Azure endpoint, deployment, key, and API version variables there.

## Dataset Setup

Run these commands from the `habitat-lab` root:

```bash
mkdir -p data/datasets/objectnav/mp3d data/datasets/objectnav/hm3d
```

Download MP3D ObjectNav episodes:

```bash
wget -c https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip \
  -P data/datasets/objectnav/mp3d/

unzip -q data/datasets/objectnav/mp3d/objectnav_mp3d_v1.zip \
  -d data/datasets/objectnav/mp3d/
```

Download HM3Dv1 / HM3DSem-v0.1 ObjectNav episodes:

```bash
wget -c https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip \
  -P data/datasets/objectnav/hm3d/

unzip -q data/datasets/objectnav/hm3d/objectnav_hm3d_v1.zip \
  -d data/datasets/objectnav/hm3d/
```

Prepare MP3D scenes. After receiving MP3D access and obtaining
`download_mp.py` from Matterport3D:

```bash
mkdir -p data/scene_datasets/mp3d
python2 download_mp.py --task habitat -o data/scene_datasets/mp3d/

wget -c https://dl.fbaipublicfiles.com/habitat/mp3d/config_v1/mp3d.scene_dataset_config.json \
  -O data/scene_datasets/mp3d/mp3d.scene_dataset_config.json
```

Prepare HM3D scenes. Replace the token values with your Matterport credentials:

```bash
export MATTERPORT_TOKEN_ID="..."
export MATTERPORT_TOKEN_SECRET="..."

python -m habitat_sim.utils.datasets_download \
  --username "$MATTERPORT_TOKEN_ID" \
  --password "$MATTERPORT_TOKEN_SECRET" \
  --uids hm3d_v0.1 \
  --data-path data/
```

The final data layout should look like:

```text
habitat-lab/data/
  scene_datasets/
    mp3d/{scene}/{scene}.glb
    hm3d/{split}/00xxx-{scene}/{scene}.basis.glb
  datasets/
    objectnav/mp3d/v1/
    objectnav/hm3d/v1/
```

### Other Datasets

The code also exposes `--dataset hm3dv2` and `--dataset hssd`. Install the
matching Habitat ObjectNav episode files and scene datasets under
`habitat-lab/data` before using those flags. For HSSD with ground-truth
semantics, the semantic lexicon is expected at:

```text
habitat-lab/data/scene_datasets/hssd-hab/semantics/hssd-hab_semantic_lexicon.json
```

## Running

Use `objnav_benchmark.py` for debug runs. It writes per-episode visualizations,
trajectory videos, observations, and affordance maps under `images/`.

```bash
python objnav_benchmark.py \
  --dataset hm3dv1 \
  --split val \
  --agent default \
  --eval_episodes 1
```

Use `eval.py` for benchmarking. It writes aggregate logs under `logs/`.

```bash
python eval.py \
  --dataset hm3dv1 \
  --split val \
  --agent default \
  --eval_episodes -1
```

Important arguments:

| Argument | Values / behavior |
| --- | --- |
| `--dataset` | `hm3dv1`, `hm3dv2`, `hssd`, or `mp3d` |
| `--agent default` | Original InstructNav-style agent |
| `--agent frontier` | Frontier-based exploration, used for FPE |
| `--agent llm` | LLM-based semantic heuristic frontier selection, used for SHF |
| `--agent lfg` | LLM frontier guidance, used for LFG |
| `--gt_semantics` | Use Habitat ground-truth object semantics instead of GLEE detections |
| `--eval_episodes` | Number of episodes to run; `-1` uses the configured full split |
| `--episodes_per_scene` | Limit episodes sampled per scene when positive |
| `--max_episode_steps` | Maximum simulator steps per episode, default `500` |
| `--track_target_only` | Restrict perception tracking to target object categories |
| `--snap_point` | Snap selected navigation points to the current navmesh island |

Example comparisons:

```bash
# Original InstructNav-style policy
python eval.py --dataset hm3dv1 --agent default --eval_episodes -1

# Frontier exploration baseline
python eval.py --dataset hm3dv1 --agent frontier --eval_episodes -1

# LLM semantic heuristic frontier selection
python eval.py --dataset hm3dv1 --agent llm --eval_episodes -1

# LLM frontier guidance
python eval.py --dataset hm3dv1 --agent lfg --eval_episodes -1

# Ground-truth semantics ablation
python eval.py --dataset hm3dv1 --agent frontier --gt_semantics --eval_episodes -1
```

For MP3D:

```bash
python eval.py --dataset mp3d --split val --agent frontier --eval_episodes -1
```

## Quick Smoke Test

After installation and dataset setup, a small debug run is the fastest way to
check that Habitat, GLEE, API keys, and paths are wired correctly:

```bash
python objnav_benchmark.py \
  --dataset hm3dv1 \
  --agent frontier \
  --eval_episodes 1 \
  --max_episode_steps 50
```

Use `--agent frontier` first if you want to test the simulator/perception path
without making LLM calls.

## Citation

If you use this scrutinized code or analysis, please cite:

```bibtex
@misc{aghaei2025engineering,
  title={When Engineering Outruns Intelligence: Rethinking Instruction-Guided Navigation},
  author={Aghaei, Matin and Zhang, Lingfeng and Alomrani, Mohammad Ali and Biparva, Mahdi and Zhang, Yingxue},
  year={2025},
  eprint={2507.20021},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

The original InstructNav paper and codebase are:

```bibtex
@misc{long2024instructnav,
  title={InstructNav: Zero-shot System for Generic Instruction Navigation in Unexplored Environment},
  author={Long, Yuxing and Cai, Wenzhe and Wang, Hongcheng and Zhan, Guanqi and Dong, Hao},
  year={2024},
  eprint={2406.04882},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```
