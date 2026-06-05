#!/usr/bin/env python3
#!/usr/bin/env python3

import os, csv, time, argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torch

from alpamayo_wrapper import AlpamayoInference


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

IMG_H, IMG_W = 1080, 1920
COL_GT      = "#00FF9F"
COL_ORIG    = "#4FC3F7"
COL_INPAINT = "#FF6B6B"
COL_BG      = "#0D1117"


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(pred_xyz, gt_xyz):
    gt_xy = gt_xyz[:, :2]
    pred_xy = pred_xyz[:, :, :2]

    disp = np.linalg.norm(pred_xy - gt_xy[None], axis=-1)

    ade = disp.mean(axis=-1)
    fde = disp[:, -1]

    return {
        "minADE": float(ade.min()),
        "avgADE": float(ade.mean()),
        "minFDE": float(fde.min()),
        "avgFDE": float(fde.mean()),
        "best_idx": int(ade.argmin()),
    }


def compute_ad_fd(orig_pred, var_pred):
    """
    Mean-over-candidates approach (matches analyse_deviation_v4 / ad_fd_gen.py).

    Summarise each distribution by its mean trajectory over all K candidates,
    then measure displacement between the two means.

    AD = mean over timesteps of ||mean_variant - mean_original||
    FD = ||mean_variant[-1] - mean_original[-1]||
    """
    orig_mean = orig_pred[:, :, :2].mean(axis=0)   # [T, 2]
    var_mean  = var_pred[:, :, :2].mean(axis=0)    # [T, 2]
    disp = np.linalg.norm(var_mean - orig_mean, axis=-1)  # [T]
    return float(disp.mean()), float(disp[-1])


# ─────────────────────────────────────────────
# PLOTTING (UNCHANGED)
# ─────────────────────────────────────────────

def plot_bev(scene_name, title, gt_xyz,
             pred, best_idx, color, out_path):

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=COL_BG)

    for i, traj in enumerate(pred):
        lw = 2.5 if i == best_idx else 0.8
        alpha = 1.0 if i == best_idx else 0.3
        ax.plot(traj[:, 1], traj[:, 0], color=color, lw=lw, alpha=alpha)

    ax.plot(gt_xyz[:, 1], gt_xyz[:, 0], color=COL_GT, lw=2.5, linestyle="--")

    ax.set_title(title, color="white")
    ax.set_aspect("equal")
    ax.grid(True, color="#1e2730")

    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=COL_BG)
    plt.close(fig)


def plot_overlay(image_path, gt_xyz, pred, best_idx, color, out_path):
    img = np.array(Image.open(image_path).convert("RGB"))

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=COL_BG)
    ax.imshow(img)
    ax.axis("off")

    def project(xyz):
        fx = 800
        cx, cy = IMG_W / 2, IMG_H * 0.72
        out = []
        for x, y, z in xyz:
            if x < 0.5:
                out.append(None)
                continue
            u = cx - (y / x) * fx
            v = cy - (1.5 / x) * fx * 0.5
            out.append((u, v))
        return out

    def draw(traj, lw=2.5, alpha=1.0):
        pts = project(traj)
        pts = [p for p in pts if p]
        for i in range(len(pts) - 1):
            ax.plot([pts[i][0], pts[i+1][0]],
                    [pts[i][1], pts[i+1][1]],
                    color=color, lw=lw, alpha=alpha)

    draw(pred[best_idx])
    draw(gt_xyz, lw=2.5)

    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor=COL_BG)
    plt.close(fig)


# ─────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--results_dir", default="comparison_results")
    p.add_argument("--num_traj", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--global_csv",type=str,default=None,help="Path to global results CSV file")
    return p.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    model = AlpamayoInference()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    if args.global_csv is not None:
        global_csv = Path(args.global_csv)
    else:
        global_csv = results_dir / "global_results.csv"

    with open(args.csv) as f:
        scenes = list(csv.DictReader(f))

    global_rows = []

    for i, row in enumerate(scenes):
        scene_name = row["scene_name"]
        print(f"\n[{i+1}/{len(scenes)}] {scene_name}")

        root = Path(args.root)
        front_img = root / row["saved_image"]
        scene_dir = front_img.parent

        npz = np.load(root / row["data_npz"])
        ego_xyz = npz["ego_history_xyz"]
        ego_rot = npz["ego_history_rot"]
        gt_xyz = npz["future_xyz"]

        scene_out = results_dir / scene_name
        scene_out.mkdir(exist_ok=True)

        traj_dir = scene_out / "trajectories"
        traj_dir.mkdir(exist_ok=True)

        plots_dir = scene_out / "plots"
        plots_dir.mkdir(exist_ok=True)

        scene_rows = []

        # ── ORIGINAL ─────────────────────────
        print("  original...", end=" ", flush=True)

        t0 = time.time()
        torch.cuda.manual_seed_all(args.seed)
        orig_pred = model.predict(front_img, ego_xyz, ego_rot,
                                  args.num_traj, args.seed)
        t = time.time() - t0

        orig_metrics = compute_metrics(orig_pred, gt_xyz)

        print(f"minADE={orig_metrics['minADE']:.3f} ({t:.1f}s)")

        orig_npz = traj_dir / "original.npz"
        np.savez_compressed(orig_npz, pred_xyz=orig_pred, gt_xyz=gt_xyz)

        scene_rows.append({
            "scene_name": scene_name,
            "variant": "original",
            "minADE": orig_metrics["minADE"],
            "minFDE": orig_metrics["minFDE"],
            "delta_minADE": 0.0,
            "delta_minFDE": 0.0,
            "AD": 0.0,
            "FD": 0.0,
            "traj_npz": str(orig_npz),
        })

        # ── VARIANTS ─────────────────────────
        inpaint_dir = scene_dir / "inpainted_clean_fin"

        if not inpaint_dir.exists():
            continue

        for img_path in sorted(inpaint_dir.glob("*_imagen.png")):
            variant = img_path.stem

            print(f"  {variant}...", end=" ", flush=True)

            t0 = time.time()
            torch.cuda.manual_seed_all(args.seed)
            pred = model.predict(img_path, ego_xyz, ego_rot,
                                 args.num_traj, args.seed)
            t = time.time() - t0

            metrics = compute_metrics(pred, gt_xyz)

            d_ade = metrics["minADE"] - orig_metrics["minADE"]
            d_fde = metrics["minFDE"] - orig_metrics["minFDE"]

            AD, FD = compute_ad_fd(orig_pred, pred)

            print(f"ADE={metrics['minADE']:.3f} Δ={d_ade:+.3f}")

            var_npz = traj_dir / f"{variant}.npz"
            np.savez_compressed(var_npz, pred_xyz=pred, gt_xyz=gt_xyz)

            plot_bev(scene_name, variant, gt_xyz,
                     pred, metrics["best_idx"], COL_INPAINT,
                     plots_dir / f"{variant}_bev.png")

            plot_overlay(img_path, gt_xyz,
                         pred, metrics["best_idx"], COL_INPAINT,
                         plots_dir / f"{variant}_overlay.png")

            row_out = {
                "scene_name": scene_name,
                "variant": variant,
                "minADE": metrics["minADE"],
                "minFDE": metrics["minFDE"],
                "delta_minADE": d_ade,
                "delta_minFDE": d_fde,
                "AD": AD,
                "FD": FD,
                "traj_npz": str(var_npz),
            }

            scene_rows.append(row_out)
            global_rows.append(row_out)

        # ── SAVE SCENE CSV ───────────────────
        scene_csv = scene_out / f"{scene_name}_comparison.csv"
        with open(scene_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=scene_rows[0].keys())
            w.writeheader()
            for r in scene_rows:
                w.writerow(r)

        print(f"  ✓ saved → {scene_out}")

    # ── GLOBAL CSV ───────────────────────────
    with open(global_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=global_rows[0].keys())
        w.writeheader()
        for r in global_rows:
            w.writerow(r)

    print(f"\nSaved global CSV → {global_csv}")


if __name__ == "__main__":
    main()