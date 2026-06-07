# CVAA: Counterfactual Visual Attribution for Autonomous Driving

CVAA is a pipeline for measuring the causal influence of individual scene
objects on the trajectory predictions of autonomous driving models, using
photorealistic counterfactual inpainting.

For each scene object, the object is removed via inpainting and the resulting
shift in the model's predicted trajectory distribution is measured using:

- **AD (Average Deviation)** — mean displacement between trajectory distributions over the full prediction horizon
- **FD (Final Deviation)** — displacement at the terminal waypoint

This repository contains the inference pipeline, metric computation, and
interactive scene explorer. The counterfactual dataset
(**Counter-nuScenes**) will be released upon acceptance.

The scene explorer with a demo set of scenes is available at https://cvaa.onrender.com
---

## Repository Structure

```
CVAA/
├── dataset_runner.py        # Runs AV model inference over original + counterfactual images
├── ad_fd_gen.py             # Computes AD and FD from saved trajectory npz files
├── visualisation_final.py   # Interactive scene explorer (Flask backend)
├── index_v2.html            # Scene explorer frontend
│
├── src/
│   └── alpamayo/            # AV model source — place your model here
│       └── ...
│
├── alpamayo_wrapper.py      # Reference wrapper for Alpamayo R1
│
└── demo_scenes/
    └── nuscenes_with_inpainted/
        ├── scene-0029/
        ├── scene-0252/
        └── scene-0731/
```

---

## Dataset

The Counter-nuScenes dataset will be released upon acceptance.

In the meantime, use the included `demo_scenes/` for a quick start (3 scenes).

---

## Setup

```bash
pip install -r requirements.txt
```

Place your AV model source under `src/alpamayo/` (or whichever model you
are using). For Alpamayo R1, clone the model repository there:

```bash
mkdir -p src
git clone https://github.com/your-av-model src/alpamayo
```

---

## Adapting to Your Own Model

`alpamayo_wrapper.py` is the reference wrapper for Alpamayo R1. To use a
different AV model, implement the same interface:

```python
class YourModelInference:
    def __init__(self, device="cuda"):
        # Load your model here
        ...

    def predict(self, image_path, ego_xyz, ego_rot,
                num_traj=6, seed=42) -> np.ndarray:
        """
        Args:
            image_path : Path to front-camera image (original or inpainted)
            ego_xyz    : (T, 3) ego vehicle history positions
            ego_rot    : (T, 4) ego vehicle history rotations
            num_traj   : number of trajectory samples K
            seed       : random seed for reproducibility

        Returns:
            pred_xyz   : (K, T, 3) predicted future trajectories
        """
        ...
```

Then pass your wrapper to `dataset_runner.py` by replacing the import:

```python
# dataset_runner.py line 12 — swap in your wrapper
from your_model_wrapper import YourModelInference as AlpamayoInference
```

---

## Usage

### 1. Run inference

```bash
python dataset_runner.py \
    --csv     scene_list.csv \
    --root    nuscenes_with_inpainted/ \
    --results_dir results/ \
    --num_traj 6 \
    --seed 42
```

`scene_list.csv` requires columns: `scene_name`, `saved_image`, `data_npz`.

### 2. Compute AD / FD

```bash
python ad_fd_gen.py \
    results/global_results.csv \
    results/comp_all.csv
```

### 3. Launch scene explorer

```bash
python visualisation_final.py \
    --root  demo_scenes \
    --csvs  demo_scenes/nuscenes_with_inpainted/comp_all_v2_clean.csv \
            demo_scenes/nuscenes_with_inpainted/comp_all_v3_clean.csv \
            demo_scenes/nuscenes_with_inpainted/comp_all_v4_clean.csv \
            demo_scenes/nuscenes_with_inpainted/demo_merged.csv \
    --port  5050
```

Open `http://localhost:5050` in your browser.

---

## Scene Explorer/ Visualisation Tool

The interactive scene explorer lets you:

- Browse scenes and per-object AD/FD rankings
- Overlay segmentation masks coloured by attribution rank
- Compare rankings across multiple inference runs (seeds)
- Inspect object metadata (label, depth, confidence, bounding box)

--


Then pass your wrapper to `dataset_runner.py` by replacing the import:

```python
# dataset_runner.py line 12 — swap in your wrapper
from your_model_wrapper import YourModelInference as AlpamayoInference
```

---

## Usage

### 1. Run inference

```bash
python dataset_runner.py \
    --csv     scene_list.csv \
    --root    nuscenes_with_inpainted/ \
    --results_dir results/ \
    --num_traj 6 \
    --seed 42
```

`scene_list.csv` requires columns: `scene_name`, `saved_image`, `data_npz`.

### 2. Compute AD / FD

```bash
python ad_fd_gen.py \
    results/global_results.csv \
    results/comp_all.csv
```

### 3. Launch scene explorer

```bash
python visualisation_final.py \
    --root  nuscenes_with_inpainted/ \
    --csvs  results/comp_all.csv \
    --port  5050
```

Open `http://localhost:5050` in your browser.

---

## Scene Explorer

The interactive scene explorer lets you:

- Browse scenes and per-object AD/FD rankings
- Overlay segmentation masks coloured by attribution rank
- Compare rankings across multiple inference runs (seeds)
- Inspect object metadata (label, depth, confidence, bounding box)

---


## License

TODO
