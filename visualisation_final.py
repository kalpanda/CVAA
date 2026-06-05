#!/usr/bin/env python3
"""
visualisation_compare.py

Clean multi-run scene explorer (visualisation_v2 style).
No mask tools — purely for comparing AD/FD across runs.

Usage:
    python visualisation_compare.py \
        --root /home/kalpanapanda/data/gemini_imgen \
        --csvs comp_all_v3_clean.csv comp_all_v4_clean.csv \
        --run-names v3 v4 \
        --port 5050
"""

import json, csv, io, base64, argparse, re
import numpy as np
from pathlib import Path
from collections import defaultdict
from flask import Flask, jsonify, send_file, abort
from PIL import Image

app = Flask(__name__, static_url_path="")

SCENES_ROOT = Path(".")
RUN_DATA    = {}
RUN_RANKED  = {}
RUN_NAMES   = []
SCENE_LIST  = []
HTML_PATH   = Path(__file__).parent / "index_v2.html"


def _parse_variant(variant):
    m = re.match(r"^(\d+)_(.+?)_conf([\d.]+)", variant or "")
    if not m:
        # handle combined/drawn style
        stem = re.sub(r"_imagen$", "", variant or "")
        if stem.startswith("drawn_mask_"):
            return -1, "drawn_mask", 1.0
        if stem.startswith("combined_"):
            rest = re.sub(r"^combined_scene-\d+_", "", stem)
            return -1, f"combined_{rest}", 1.0
        return None, None, None
    return int(m.group(1)) - 1, m.group(2), float(m.group(3))


def load_csv(csv_path, run_name):
    global SCENE_LIST

    if not Path(csv_path).exists():
        print(f"[WARN] {csv_path} not found — skipping '{run_name}'")
        return

    def sf(v):
        try:    return float(v) if v not in ("", None) else None
        except: return None

    scene_objects = defaultdict(list)

    with open(csv_path, newline="") as f:
        reader  = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        has_idx = "object_idx" in columns

        for row in reader:
            variant = row.get("variant", "")
            if variant == "original":
                continue
            scene = row["scene_name"]

            if has_idx:
                try:    obj_idx = int(float(row["object_idx"]))
                except: obj_idx = -1
                obj_label = row.get("object_label", "")
                obj_conf  = sf(row.get("object_conf"))
            else:
                obj_idx, obj_label, obj_conf = _parse_variant(variant)
                if obj_idx is None:
                    obj_idx = -1

            scene_objects[scene].append({
                "object_label": obj_label,
                "object_idx":   obj_idx if obj_idx is not None else -1,
                "object_conf":  obj_conf,
                "AD":           sf(row.get("AD")),
                "FD":           sf(row.get("FD")),
                "depth_m":      sf(row.get("depth_m")),
                "coverage_pct": sf(row.get("coverage_pct")),
                "variant":      variant,
            })

    scene_ranked = {}
    for scene, objs in scene_objects.items():
        # Deduplicate: keep highest-AD per object_idx
        # For combined/drawn (idx=-1) use variant as key so they don't collapse
        best = {}
        for obj in objs:
            idx = obj["object_idx"]
            key = (idx, obj["variant"]) if idx < 0 else idx
            if key not in best or (obj["AD"] or 0) > (best[key]["AD"] or 0):
                best[key] = obj
        objs = list(best.values())

        ranked = sorted(objs, key=lambda r: r["AD"] if r["AD"] is not None else -1, reverse=True)
        for i, obj in enumerate(ranked):
            obj["rank_ad"] = i + 1
        ranked_fd = sorted(objs, key=lambda r: r["FD"] if r["FD"] is not None else -1, reverse=True)
        for i, obj in enumerate(ranked_fd):
            obj["rank_fd"] = i + 1
        scene_ranked[scene] = ranked

    RUN_DATA[run_name]   = dict(scene_objects)
    RUN_RANKED[run_name] = scene_ranked
    RUN_NAMES.append(run_name)
    SCENE_LIST = sorted(set(SCENE_LIST) | set(scene_objects.keys()))

    total = sum(len(v) for v in scene_objects.values())
    print(f"  [{run_name}] {total} objects across {len(scene_objects)} scenes")


def load_seg_json(scene_name):
    path = SCENES_ROOT / "nuscenes_with_inpainted" / scene_name / f"{scene_name}_seg.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {
        obj.get("id", -1) - 1: {
            "label":    obj.get("label", ""),
            "box_xyxy": obj.get("box_xyxy", []),
        }
        for obj in data.get("objects", [])
        if obj.get("id", -1) >= 1
    }


def load_masks(scene_name):
    npz_list = sorted((SCENES_ROOT / "nuscenes_per_scene" / scene_name).glob("*_masks.npz"))
    if not npz_list:
        return None, 0, 0
    try:
        masks = (np.load(npz_list[0])["masks"] > 0).astype(bool)
        return masks, masks.shape[1], masks.shape[2]
    except Exception as e:
        print(f"[WARN] mask load failed for {scene_name}: {e}")
        return None, 0, 0


@app.route("/")
def index():
    if not HTML_PATH.exists():
        return (f"<h2>visualisation_compare.html not found</h2>"
                f"<p>Expected: <code>{HTML_PATH.resolve()}</code></p>", 404)
    return send_file(str(HTML_PATH))


@app.route("/api/runs")
def api_runs():
    return jsonify(RUN_NAMES)


@app.route("/api/scenes")
def api_scenes():
    return jsonify(SCENE_LIST)


@app.route("/api/scene/<run_name>/<scene_name>")
def api_scene(run_name, scene_name):
    run = RUN_RANKED.get(run_name)
    if run is None:
        abort(404)
    seg = load_seg_json(scene_name)
    # Build label→idx lookup for combined/drawn resolution
    label_to_idx = {info["label"]: idx for idx, info in seg.items()}

    # Priority order for combined mask label selection
    LABEL_PRIORITY = ["bus", "truck", "car", "person",
                      "bicycle", "motorcycle", "traffic light"]

    def combined_label(source_labels):
        """
        Pick the most meaningful label from a list of source labels.
        Priority: bus > truck > car > person > bicycle > motorcycle > traffic light.
        If none match, return the most common label.
        """
        if not source_labels:
            return "combined"
        for preferred in LABEL_PRIORITY:
            if preferred in source_labels:
                return preferred
        # Fall back to most common
        from collections import Counter
        return Counter(source_labels).most_common(1)[0][0]

    objects = []
    for rec in run.get(scene_name, []):
        obj = dict(rec)
        idx = obj["object_idx"]

        # Resolve combined/drawn idx from seg.json by matching label
        if idx < 0:
            label = obj["object_label"] or ""
            idx   = label_to_idx.get(label, -1)
            obj["object_idx"] = idx

        info = seg.get(idx, {})

        # For combined masks, derive display label from source objects
        source_labels = info.get("source_labels", [])
        if source_labels:
            obj["object_label"] = combined_label(source_labels)
            obj["label_seg"]    = obj["object_label"]
        else:
            obj["label_seg"] = info.get("label", obj["object_label"])

        obj["box_xyxy"] = info.get("box_xyxy", [])
        objects.append(obj)
    return jsonify({"scene_name": scene_name, "run_name": run_name, "objects": objects})


@app.route("/api/image/<scene_name>")
def api_image(scene_name):
    p = SCENES_ROOT / "nuscenes_per_scene" / scene_name / f"{scene_name}_front.jpg"
    if not p.exists():
        abort(404)
    return send_file(str(p), mimetype="image/jpeg")


@app.route("/api/masks_all/<scene_name>")
def api_masks_all(scene_name):
    masks, H, W = load_masks(scene_name)
    if masks is None:
        return jsonify({})
    result = {}
    for i, mask in enumerate(masks):
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[mask, 3] = 255
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
        result[str(i)] = base64.b64encode(buf.getvalue()).decode()
    return jsonify(result)


def main():
    global SCENES_ROOT, HTML_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--root",      default=".",
                        help="Project root containing nuscenes_per_scene/")
    parser.add_argument("--csvs",      nargs="+", required=True)
    parser.add_argument("--run-names", nargs="+", default=None)
    parser.add_argument("--port",      type=int, default=5050)
    args = parser.parse_args()

    SCENES_ROOT = Path(args.root)
    HTML_PATH   = Path(__file__).parent / "index_v2.html"

    run_names = args.run_names or [Path(c).stem for c in args.csvs]
    if len(run_names) != len(args.csvs):
        print("[ERROR] --run-names must match --csvs count")
        raise SystemExit(1)

    for csv_path, name in zip(args.csvs, run_names):
        full = Path(csv_path) if Path(csv_path).is_absolute() else SCENES_ROOT / csv_path
        load_csv(str(full), name)

    print(f"\nRuns: {run_names}  |  Scenes: {len(SCENE_LIST)}")
    print(f"Open http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
