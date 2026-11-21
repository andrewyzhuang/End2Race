"""Collect single car expert demonstrations for retrain-v1 (step 2 of retrain/run_all.sh).

Expert: pure pursuit on the racelines made by retrain/make_tracks.py. No opponents.
Differences from expert/collect.py (on purpose, to match the NeoRacer):
  * The expert action is computed once per control step (30 Hz) and held for the
    physics steps in between (zero-order hold), like the real /drive loop.
  * Optional action delay of 0 or 1 control step (servo / ESC response).
  * Friction mu is sampled per episode.
  * Recorded LiDAR (180 beams): clipped to lidar.max_range_m, optional Gaussian
    jitter and beam dropout, then the rear sector is overwritten with
    lidar.occluded_value_m (the real car cannot see behind itself).
Every sampled value is saved in summary.json so a run can be reproduced.

Usage:
  python retrain/collect.py                 # full collection into retrain.dataset_dir
  python retrain/collect.py --pilot 48      # 48 random episodes into <dataset_dir>_pilot
"""
import argparse
import csv
import json
import multiprocessing as mp
import sys
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

NUM_LIDAR_FEATURES = 180  # imitation.model.End2Race.NUM_LIDAR_FEATURES (not imported to keep workers light)
MANIFEST_PATH = ROOT / "f1tenth_sim" / "f1tenth_racetracks" / "retrain_tracks_manifest.json"


def load_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


# ----------------------------------------------------------------------------
# Scenario list
# ----------------------------------------------------------------------------
def usable_lines(manifest, tracks):
    """(variant, direction, [usable raceline files]) for every variant of `tracks` that kept its center lane.

    The manifest also holds the eval tracks (retrain/eval.py), so filter by track name.
    """
    lines = []
    for variant, info in manifest["variants"].items():
        if info["track"] not in tracks:
            continue
        for direction, entry in info["directions"].items():
            usable = [lane["raceline"] for lane in entry["lanes"] if lane["usable"]]
            center = f"raceline1_{direction}"
            if entry["error"] is None and center in usable:
                lines.append((variant, direction, usable))
    return lines


def build_scenarios(settings, manifest):
    from expert.utils import get_ego_idx_range

    rng = np.random.default_rng(settings["seed"])
    randomization = settings["randomization"]
    scenarios = []
    for variant, direction, usable in usable_lines(manifest, settings["tracks"]):
        for start in range(settings["num_startpoints"]):
            # Start point i drives lane (i mod 3); fall back to the center lane if that lane was dropped.
            raceline = f"raceline{start % 3}_{direction}"
            if raceline not in usable:
                raceline = f"raceline1_{direction}"
            start_index = get_ego_idx_range(variant, raceline, settings["num_startpoints"])[start]
            for speed_scale in settings["speed_scales"]:
                noisy = bool(rng.random() < randomization["lidar_noise_probability"])
                scenarios.append({
                    "variant": variant,
                    "direction": direction,
                    "raceline": raceline,
                    "start_index": int(start_index),
                    "speed_scale": float(speed_scale),
                    "initial_speed_fraction": float(rng.uniform(*settings["initial_speed_fraction_range"])),
                    "mu": float(rng.uniform(*randomization["mu_range"])),
                    "action_delay_steps": int(rng.choice(randomization["action_delay_steps"])),
                    "lidar_noise_std_m": float(rng.uniform(*randomization["lidar_noise_std_range_m"])) if noisy else 0.0,
                    "lidar_dropout": float(rng.uniform(*randomization["lidar_dropout_range"])) if noisy else 0.0,
                    "seed": int(rng.integers(0, 2**31 - 1)),
                })
    return scenarios


def episode_name(scenario):
    return (
        f"{scenario['variant']}_{scenario['direction']}_{scenario['raceline'].split('_')[0]}"
        f"_e{scenario['start_index']}_v{scenario['speed_scale']}"
    )


# ----------------------------------------------------------------------------
# LiDAR processing (what the policy sees)
# ----------------------------------------------------------------------------
def occlusion_mask(num_beams, fov, occlusion_deg):
    """True for the downsampled beams inside the blocked rear sector.

    The simulator scan starts at -fov/2 (the rear) and goes counter clockwise,
    the same order as the NeoRacer /scan, so index 0 is the rear and the middle is the front.
    """
    raw_beams = 1440
    raw_indices = np.linspace(0, raw_beams - 1, num_beams).astype(int) if raw_beams % num_beams else np.arange(num_beams) * (raw_beams // num_beams)
    angles = -fov / 2.0 + raw_indices * (fov / (raw_beams - 1))
    return np.abs(angles) > np.pi - np.deg2rad(occlusion_deg) / 2.0


def process_lidar(scan, scenario, lidar, mask, rng):
    from expert.utils import downsample_lidar

    beams = downsample_lidar(scan, NUM_LIDAR_FEATURES).astype(np.float64)
    if scenario["lidar_noise_std_m"] > 0.0:
        beams = beams + rng.normal(0.0, scenario["lidar_noise_std_m"], size=beams.shape)
    beams = np.clip(beams, 0.0, lidar.max_range_m)
    if scenario["lidar_dropout"] > 0.0:
        beams[rng.random(beams.shape) < scenario["lidar_dropout"]] = lidar.max_range_m
    beams[mask] = lidar.occluded_value_m
    return beams


# ----------------------------------------------------------------------------
# Grip based speed cap
# The raceline speed profile is smoothed by velocity_filter_window AFTER the grip
# limit is applied (trajectory_planning_helpers calc_vel_profile), so corner speeds
# can exceed the ggv lateral limit. The expert target speed is capped here instead.
# ----------------------------------------------------------------------------
def ggv_limits():
    from f1tenth_sim.utils import racetrack_path

    ggv = np.loadtxt(racetrack_path("vehicle_dynamic_info", "ggv.csv"), delimiter=",", comments="#", ndmin=2)
    return float(np.min(ggv[:, 1])), float(np.min(ggv[:, 2]))  # (ax_max, ay_max)


class GripSpeedCap:
    def __init__(self, waypoints_xy, kappa, ax_max, ay_max, horizon_points=40):
        self.xy = waypoints_xy
        self.corner_speed_sq = ay_max / np.maximum(np.abs(kappa), 1e-6)
        step = np.linalg.norm(np.diff(waypoints_xy, axis=0), axis=1)
        self.spacing = float(np.median(step))
        self.ax_max = ax_max
        self.horizon = horizon_points

    def __call__(self, x, y):
        from expert.utils import nearest_point

        _, _, _, index = nearest_point(np.asarray([x, y]), self.xy)
        ahead = (index + np.arange(self.horizon)) % len(self.xy)
        distance = np.arange(self.horizon) * self.spacing
        return float(np.sqrt(np.min(self.corner_speed_sq[ahead] + 2.0 * self.ax_max * distance)))


# ----------------------------------------------------------------------------
# One episode
# ----------------------------------------------------------------------------
def run_episode(scenario, settings, output_dir):
    import f110_gym  # noqa: F401  registers the environment
    import gym
    from f110_gym.envs.base_classes import Integrator

    from expert.controllers import PurePursuitController, RacelineFollower
    from expert.utils import raceline_pose
    from f1tenth_sim.utils import load_simulator_config, racetrack_path, simulation_config

    simulator = load_simulator_config()
    vehicle, lidar = simulator.vehicle, simulator.lidar
    timing = simulation_config()
    control_steps = round(settings["sim_duration"] * timing.control_frequency_hz)
    rng = np.random.default_rng(scenario["seed"])

    expert = RacelineFollower(scenario["variant"], scenario["raceline"])
    expert.conf = SimpleNamespace(**settings["pure_pursuit"], maximum_speed=vehicle.maximum_speed)
    expert.tracker = PurePursuitController(expert.conf)
    kappa = np.loadtxt(racetrack_path(scenario["variant"], f"{scenario['raceline']}.csv"), delimiter=";", skiprows=1, ndmin=2)[:, 4]
    speed_cap = GripSpeedCap(expert.waypoints[:, :2], kappa, *ggv_limits())

    params = dict(vars(simulator.dynamics), mu=scenario["mu"])
    map_path = racetrack_path(scenario["variant"], f"{scenario['variant']}_map")
    env = gym.make(
        "f110-v0", map=str(map_path), map_ext=".png", timestep=timing.timestep,
        num_agents=1, integrator=Integrator.RK4, params=params, seed=scenario["seed"],
    )
    mask = occlusion_mask(NUM_LIDAR_FEATURES, 6.28, lidar.rear_occlusion_deg)
    try:
        pose = raceline_pose(np.column_stack((expert.waypoints[:, :2], expert.waypoints[:, 3])), scenario["start_index"])
        start_speed = min(scenario["speed_scale"] * expert.waypoints[scenario["start_index"], 2], speed_cap(*pose[:2]))
        initial_speed = scenario["initial_speed_fraction"] * start_speed
        obs, _, done, _ = env.reset(poses=pose[None, :], velocities=np.asarray([initial_speed]))

        delayed = deque([(0.0, initial_speed)] * scenario["action_delay_steps"])
        rows, collided = [], False
        for step in range(control_steps):
            x, y, theta, speed = obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0], obs["linear_vels_x"][0]
            reference = expert.reference_trajectory(x, y)
            steer, target_speed = expert.tracker.plan(x, y, theta, speed, reference)
            steer = float(np.clip(steer, -vehicle.steering_limit, vehicle.steering_limit))
            target_speed = min(target_speed * scenario["speed_scale"], speed_cap(x, y))
            target_speed = float(np.clip(target_speed, 0.0, vehicle.maximum_speed))

            beams = process_lidar(obs["scans"][0], scenario, lidar, mask, rng)
            rows.append([round(step * timing.control_timestep, 6), float(speed), steer, target_speed] + beams.tolist())

            delayed.append((steer, target_speed))
            applied = np.asarray([delayed.popleft()])
            for _ in range(timing.steps_per_control):  # zero-order hold
                obs, _, done, _ = env.step(applied)
                if obs["collisions"][0]:
                    collided = True
                    break
            if collided or done:
                break
    finally:
        env.close()

    name = episode_name(scenario)
    result = {**scenario, "name": name, "steps": len(rows), "collision": collided,
              "mean_speed_mps": round(float(np.mean([row[1] for row in rows])), 3)}
    if not collided and len(rows) == control_steps:
        path = output_dir / "success" / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time", "current_speed", "steer", "desired_speed"]
                            + [f"lidar_{index}" for index in range(NUM_LIDAR_FEATURES)])
            writer.writerows(np.round(np.asarray(rows), 5).tolist())
    else:
        result["collision"] = True
        (output_dir / "collision" / f"{name}.json").write_text(json.dumps(result, indent=2))
    return result


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def summarize(results, output_dir, settings, failures):
    def tally(items):
        ok = sum(not item["collision"] for item in items)
        return {"episodes": len(items), "success": ok, "collision": len(items) - ok,
                "success_rate_percent": round(100.0 * ok / max(len(items), 1), 1)}

    groups = {}
    for item in results:
        width = item["variant"].rsplit("_w", 1)[1]
        for key in (f"variant={item['variant']}", f"width={width}", f"direction={item['direction']}",
                    f"speed_scale={item['speed_scale']}", f"delay={item['action_delay_steps']}"):
            groups.setdefault(key, []).append(item)
    from f1tenth_sim.utils import load_simulator_config
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "retrain_config": settings,
        "simulator_config": {name: vars(section) for name, section in vars(load_simulator_config()).items()},
        "results": {**tally(results), "process_failures": failures,
                    "breakdown": {k: tally(v) for k, v in sorted(groups.items())}},
        "episodes": sorted(results, key=lambda item: item["name"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", type=int, default=0, help="run only N random episodes into <dataset_dir>_pilot")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    settings = config["retrain"]
    manifest = json.loads(MANIFEST_PATH.read_text())
    scenarios = build_scenarios(settings, manifest)
    output_dir = ROOT / (settings["dataset_dir"] + ("_pilot" if args.pilot else ""))
    if args.pilot:
        picks = np.random.default_rng(settings["seed"] + 1).choice(len(scenarios), size=min(args.pilot, len(scenarios)), replace=False)
        scenarios = [scenarios[index] for index in sorted(picks)]
    if output_dir.exists() and any(output_dir.glob("*/*")):
        raise FileExistsError(f"{output_dir} already has episodes. Move or delete it first.")
    for sub in ("success", "collision"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    workers = args.workers or settings.get("workers", 4)
    print(f"Collecting {len(scenarios)} episodes into {output_dir} with {workers} workers", flush=True)
    results, failures = [], 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(run_episode, scenario, settings, output_dir): scenario for scenario in scenarios}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result())
            except Exception as exc:
                failures += 1
                print(f"Episode {episode_name(futures[future])} failed: {type(exc).__name__}: {exc}", flush=True)
            if count % 50 == 0 or count == len(scenarios):
                collisions = sum(item["collision"] for item in results)
                print(f"  {count}/{len(scenarios)} done, collisions={collisions}, errors={failures}", flush=True)

    summary = summarize(results, output_dir, settings, failures)
    print(json.dumps({k: v for k, v in summary["results"].items() if k != "breakdown"}, indent=2))
    for key, value in summary["results"]["breakdown"].items():
        if not key.startswith("variant="):
            print(f"  {key:<20} {value}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
