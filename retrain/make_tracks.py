"""Generate the retrain-v1 track variants (step 1 of retrain/run_all.sh).

For every track in config.yaml retrain.tracks and every width in retrain.target_widths_m:
  1. Make a uniformly shrunk copy of the map, f1tenth_racetracks/<Track>_w<cm>/.
     Only the map resolution and origin change (the image is reused), so every length,
     including the track width and the corner radius, is multiplied by
     k = target width / original width.
  2. For each direction (cw, ccw) run generate_raceline.generate_lanes with the matching
     `clockwise` flag, then generate_raceline.generate_raceline for each lane.
     Files: lane<i>_<dir>.csv and raceline<i>_<dir>.csv.
  3. Check the driving direction (signed area) and the tightest raceline radius.
     A lane is dropped when its radius < margin * wheelbase / tan(max steer).

Output: f1tenth_racetracks/retrain_tracks_manifest.json (read by retrain/collect.py)
and preview images in retrain/previews/.
"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")  # generate_raceline calls plt.show() on errors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "f1tenth_sim" / "f1tenth_racetracks"))

import numpy as np
import yaml

from f1tenth_sim.utils import load_simulator_config, racetrack_path

TRACKS_DIR = ROOT / "f1tenth_sim" / "f1tenth_racetracks"
MANIFEST_PATH = TRACKS_DIR / "retrain_tracks_manifest.json"
PREVIEW_DIR = ROOT / "retrain" / "previews"


def load_retrain_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)["retrain"]


def original_width(track):
    """Mean track width (w_tr_right + w_tr_left) from the bundled centerline file."""
    values = np.loadtxt(racetrack_path(track, f"{track}_centerline.csv"), delimiter=",", comments="#")
    return float(np.mean(values[:, 2] + values[:, 3]))


def variant_name(track, width):
    return f"{track}_w{int(round(width * 100))}"


def make_scaled_map(track, variant, scale):
    """Copy the map image and write a yaml whose resolution and origin are multiplied by scale."""
    source_dir = TRACKS_DIR / track
    target_dir = TRACKS_DIR / variant
    target_dir.mkdir(exist_ok=True)
    with (source_dir / f"{track}_map.yaml").open(encoding="utf-8") as stream:
        map_yaml = yaml.safe_load(stream)
    map_yaml["image"] = f"{variant}_map.png"
    map_yaml["resolution"] = float(map_yaml["resolution"]) * scale
    origin = [float(value) for value in map_yaml["origin"]]
    map_yaml["origin"] = [origin[0] * scale, origin[1] * scale, origin[2]]
    shutil.copyfile(source_dir / f"{track}_map.png", target_dir / f"{variant}_map.png")
    with (target_dir / f"{variant}_map.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(map_yaml, stream, default_flow_style=None, sort_keys=False)
    return target_dir


def wall_clearance(variant, xy, half_width):
    """Distance from the car edge to the nearest wall pixel, for each point (map distance transform)."""
    import cv2

    map_dir = TRACKS_DIR / variant
    with (map_dir / f"{variant}_map.yaml").open(encoding="utf-8") as stream:
        map_yaml = yaml.safe_load(stream)
    image = cv2.imread(str(map_dir / f"{variant}_map.png"), cv2.IMREAD_GRAYSCALE)
    resolution, origin = map_yaml["resolution"], map_yaml["origin"]
    distance = cv2.distanceTransform((image > 127).astype(np.uint8), cv2.DIST_L2, 5) * resolution
    height, width = image.shape
    columns = np.clip(((xy[:, 0] - origin[0]) / resolution).astype(int), 0, width - 1)
    rows = np.clip((height - (xy[:, 1] - origin[1]) / resolution).astype(int), 0, height - 1)
    return distance[rows, columns] - half_width


def check_lane(lane, variant, settings, min_radius, half_width):
    """Fill min_wall_clearance_m and usable for one manifest lane entry."""
    values = np.loadtxt(TRACKS_DIR / variant / f"{lane['raceline']}.csv", delimiter=";", skiprows=1, ndmin=2)
    clearance = float(np.min(wall_clearance(variant, values[:, 1:3], half_width)))
    lane["min_wall_clearance_m"] = round(clearance, 3)
    lane["usable"] = bool(lane["min_radius_m"] >= min_radius and clearance >= settings["min_wall_clearance_m"])
    return lane


def recheck_manifest(settings):
    """Recompute the usable flags of an existing manifest (no raceline regeneration)."""
    simulator = load_simulator_config()
    wheelbase = simulator.dynamics.lf + simulator.dynamics.lr
    min_radius = settings["min_turn_radius_margin"] * wheelbase / np.tan(simulator.vehicle.steering_limit)
    manifest = json.loads(MANIFEST_PATH.read_text())
    for variant, info in manifest["variants"].items():
        info["min_allowed_radius_m"] = round(float(min_radius), 3)
        for direction, entry in info["directions"].items():
            for lane in entry["lanes"]:
                check_lane(lane, variant, settings, min_radius, simulator.vehicle.width / 2.0)
            print(f"  {variant:<18} {direction}: " + ", ".join(
                f"{lane['raceline']} R={lane['min_radius_m']} wall={lane['min_wall_clearance_m']}"
                f"{'' if lane['usable'] else ' DROP'}" for lane in entry["lanes"]))
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest rechecked: {MANIFEST_PATH}")


def signed_area(xy):
    """Shoelace area. Positive = counter clockwise, negative = clockwise."""
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def build_variant(track, width, settings):
    """Generate one width variant (both directions). Runs in a worker process."""
    import generate_raceline as gr  # needs cv2 and trajectory_planning_helpers

    simulator = load_simulator_config()
    wheelbase = simulator.dynamics.lf + simulator.dynamics.lr
    min_radius = settings["min_turn_radius_margin"] * wheelbase / np.tan(simulator.vehicle.steering_limit)

    width_original = original_width(track)
    scale = width / width_original
    variant = variant_name(track, width)
    map_dir = make_scaled_map(track, variant, scale)

    result = {
        "variant": variant, "track": track, "target_width_m": width,
        "original_width_m": round(width_original, 3), "scale": round(scale, 5),
        "min_allowed_radius_m": round(float(min_radius), 3), "directions": {},
    }
    for direction in settings["directions"]:
        config = SimpleNamespace(**vars(simulator.track), **vars(simulator.vehicle))
        config.map_name = variant
        config.clockwise = direction == "cw"
        entry = {"lanes": [], "error": None}
        try:
            lanes = gr.generate_lanes(config, map_dir)
            for index, lane in enumerate(lanes):
                (map_dir / f"lane{index}.csv").replace(map_dir / f"lane{index}_{direction}.csv")
                trajectory, laptime = gr.generate_raceline(lane, config, TRACKS_DIR)
                raceline_file = f"raceline{index}_{direction}"
                np.savetxt(
                    map_dir / f"{raceline_file}.csv", trajectory, delimiter=";", fmt="%.6f",
                    header="s_m;x_m;y_m;psi_rad;kappa_radpm;vx_mps;ax_mps2",
                )
                radius = 1.0 / max(float(np.max(np.abs(trajectory[:, 4]))), 1e-9)
                area = signed_area(trajectory[:-1, 1:3])
                entry["lanes"].append({
                    "raceline": raceline_file,
                    "length_m": round(float(trajectory[-1, 0]), 2),
                    "laptime_s": round(float(laptime), 2),
                    "min_radius_m": round(radius, 3),
                    "max_speed_mps": round(float(np.max(trajectory[:, 5])), 2),
                    "measured_direction": "ccw" if area > 0 else "cw",
                })
                check_lane(entry["lanes"][-1], variant, settings, min_radius, simulator.vehicle.width / 2.0)
        except Exception as exc:  # keep going with the other variants
            entry["error"] = f"{type(exc).__name__}: {exc}"
        result["directions"][direction] = entry
    save_preview(result, map_dir)
    return result


def save_preview(result, map_dir):
    """Plot the racelines on top of each other so the variant can be checked by eye."""
    import matplotlib.pyplot as plt

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for axis, (direction, entry) in zip(axes, result["directions"].items()):
        for lane in entry["lanes"]:
            values = np.loadtxt(map_dir / f"{lane['raceline']}.csv", delimiter=";", ndmin=2)
            style = "-" if lane["usable"] else ":"
            axis.plot(values[:, 1], values[:, 2], style, linewidth=0.8,
                      label=f"{lane['raceline']} R={lane['min_radius_m']} ({lane['measured_direction']})")
            axis.annotate("start", values[0, 1:3])
            axis.arrow(values[0, 1], values[0, 2], values[5, 1] - values[0, 1], values[5, 2] - values[0, 2],
                       width=0.3, color="red")
        axis.set_title(f"{result['variant']} {direction}" + (f"  ERROR {entry['error']}" if entry["error"] else ""))
        axis.set_aspect("equal")
        axis.legend(fontsize=7)
    fig.savefig(PREVIEW_DIR / f"{result['variant']}.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--tracks", nargs="*", default=None, help="subset of retrain.tracks (for testing)")
    parser.add_argument("--recheck", action="store_true", help="only recompute the usable flags of the manifest")
    args = parser.parse_args()

    settings = load_retrain_config()
    if args.recheck:
        recheck_manifest(settings)
        return
    tracks = args.tracks or settings["tracks"]
    workers = args.workers or settings.get("workers", 4)
    jobs = [(track, float(width)) for track in tracks for width in settings["target_widths_m"]]
    print(f"Building {len(jobs)} track variants with {workers} workers")

    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_variant, track, width, settings): (track, width) for track, width in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            for direction, entry in result["directions"].items():
                lanes = ", ".join(
                    f"{lane['raceline']} R={lane['min_radius_m']} wall={lane['min_wall_clearance_m']}"
                    f"{'' if lane['usable'] else ' DROP'}"
                    for lane in entry["lanes"]
                )
                print(f"  {result['variant']:<18} {direction}: {entry['error'] or lanes}", flush=True)

    old = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    merged = {**old.get("variants", {}), **{r["variant"]: r for r in results}}
    MANIFEST_PATH.write_text(json.dumps({"variants": dict(sorted(merged.items()))}, indent=2) + "\n")
    print(f"Manifest: {MANIFEST_PATH}\nPreviews: {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
