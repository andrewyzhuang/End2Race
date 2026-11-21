"""Closed-loop evaluation for retrain-v1 on unseen tracks (eval v1).

For every variant of retrain.eval_tracks (made by retrain/make_tracks.py) and both directions:
  start at 4 points along raceline1, from standstill, drive 1 lap.
Methods:
  retrain : checkpoint/bc_retrain_<version>.pt (new model)
  ppo     : checkpoint/ppo.pt (old model, the one on the car; reference only, it was
            trained for 30 m LiDAR, 40 Hz and the old car size)
  expert  : pure pursuit + grip speed cap, same as retrain/collect.py (upper bound)
Conditions follow the real car: 30 Hz control with zero-order hold, LiDAR clipped to 10 m
with the rear sector occluded (no extra noise), nominal mu, action delay 0 and 1 step.
The model speed input is the speed measured at the previous control step, as in imitation/train.py.

A run ends on: lap done (pass), wall collision, or timeout (stuck / too slow / wrong way).
Output: eval_results/retrain_<version>/results.json, summary.txt, plots/ (failed runs).

Usage: python retrain/eval.py [--methods retrain ppo expert] [--workers 8]
"""
import argparse
import json
import multiprocessing as mp
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "retrain"))

import numpy as np

import collect as C  # LiDAR processing, grip cap, manifest helpers

START_POINTS = 4
DELAYS = [0, 1]


def build_runs(config, methods):
    settings = config["retrain"]
    manifest = json.loads(C.MANIFEST_PATH.read_text())
    runs, skipped = [], []
    from expert.utils import get_ego_idx_range

    for variant, info in manifest["variants"].items():
        if info["track"] not in settings["eval_tracks"]:
            continue
        for direction, entry in info["directions"].items():
            center = f"raceline1_{direction}"
            lanes = {lane["raceline"]: lane for lane in entry["lanes"]}
            if entry["error"] or center not in lanes or not lanes[center]["usable"]:
                skipped.append(f"{variant} {direction}: " + (entry["error"] or "center lane too tight / too close to wall"))
                continue
            for start in get_ego_idx_range(variant, center, START_POINTS):
                for delay in DELAYS:
                    for method in methods:
                        runs.append({"variant": variant, "track": info["track"], "width": info["target_width_m"],
                                     "direction": direction, "raceline": center, "start_index": int(start),
                                     "delay": delay, "method": method,
                                     "timeout_s": 3.0 * lanes[center]["laptime_s"] + 20.0})
    return runs, skipped


def load_model(method, version):
    import torch
    from imitation.model import End2Race

    torch.set_num_threads(1)
    name = f"bc_retrain_{version}.pt" if method == "retrain" else "ppo.pt"
    model = End2Race()
    model.load_state_dict(torch.load(ROOT / "checkpoint" / name, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def run_one(run, config):
    import f110_gym  # noqa: F401
    import gym
    import torch
    from f110_gym.envs.base_classes import Integrator

    from expert.controllers import PurePursuitController, RacelineFollower
    from expert.utils import project_point_to_centerline, raceline_pose
    from f1tenth_sim.utils import load_simulator_config, racetrack_path, simulation_config

    settings = config["retrain"]
    simulator = load_simulator_config()
    vehicle, lidar = simulator.vehicle, simulator.lidar
    timing = simulation_config()

    reference = RacelineFollower(run["variant"], run["raceline"])
    reference.conf = SimpleNamespace(**settings["pure_pursuit"], maximum_speed=vehicle.maximum_speed)
    reference.tracker = PurePursuitController(reference.conf)
    kappa = np.loadtxt(racetrack_path(run["variant"], f"{run['raceline']}.csv"), delimiter=";", skiprows=1, ndmin=2)[:, 4]
    speed_cap = C.GripSpeedCap(reference.waypoints[:, :2], kappa, *C.ggv_limits())
    line = np.ascontiguousarray(reference.waypoints[:, :2])
    length = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())

    model = load_model(run["method"], settings["version"]) if run["method"] != "expert" else None
    hidden = None
    mask = C.occlusion_mask(C.NUM_LIDAR_FEATURES, 6.28, lidar.rear_occlusion_deg)
    clean = {"lidar_noise_std_m": 0.0, "lidar_dropout": 0.0}

    env = gym.make("f110-v0", map=str(racetrack_path(run["variant"], f"{run['variant']}_map")), map_ext=".png",
                   timestep=timing.timestep, num_agents=1, integrator=Integrator.RK4,
                   params=dict(vars(simulator.dynamics)))
    try:
        pose = raceline_pose(np.column_stack((line, reference.waypoints[:, 3])), run["start_index"])
        obs, _, _, _ = env.reset(poses=pose[None, :], velocities=np.asarray([0.0]))
        previous_progress, _ = project_point_to_centerline(pose[:2].copy(), line)
        progress, previous_speed, outcome = 0.0, 0.0, "timeout"
        delayed = deque([(0.0, 0.0)] * run["delay"])
        log = []
        max_steps = int(run["timeout_s"] * timing.control_frequency_hz)
        for step in range(max_steps):
            x, y, theta, speed = obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0], obs["linear_vels_x"][0]
            if model is None:
                steer, target = reference.tracker.plan(x, y, theta, speed, reference.reference_trajectory(x, y))
                target = min(target, speed_cap(x, y))
            else:
                beams = C.process_lidar(obs["scans"][0], clean, lidar, mask, None)
                with torch.no_grad():
                    actions, hidden = model(torch.tensor(beams, dtype=torch.float32)[None, None],
                                            torch.tensor([[[previous_speed]]], dtype=torch.float32), hidden)
                steer, target = actions[0, 0, 0].item(), actions[0, 0, 1].item()
            previous_speed = speed
            steer = float(np.clip(steer, -vehicle.steering_limit, vehicle.steering_limit))
            target = float(np.clip(target, 0.0, vehicle.maximum_speed))
            log.append((x, y, steer, speed))

            delayed.append((steer, target))
            applied = np.asarray([delayed.popleft()])
            collided = False
            for _ in range(timing.steps_per_control):
                obs, _, _, _ = env.step(applied)
                if obs["collisions"][0]:
                    collided = True
                    break
            current, _ = project_point_to_centerline(np.asarray([obs["poses_x"][0], obs["poses_y"][0]]), line)
            delta = current - previous_progress
            delta += length if delta < -length / 2 else (-length if delta > length / 2 else 0.0)
            progress, previous_progress = progress + delta, current
            if collided:
                outcome = "collision"
                break
            if progress >= length:
                outcome = "pass"
                break
    finally:
        env.close()

    log = np.asarray(log)
    result = {**run, "outcome": outcome, "lap_fraction": round(min(progress / length, 1.0), 3),
              "time_s": round(len(log) * timing.control_timestep, 2),
              "mean_speed_mps": round(float(np.mean(log[:, 3])), 3),
              "left_steer_percent": round(100.0 * float(np.mean(log[:, 2] > 0.05)), 1),
              "right_steer_percent": round(100.0 * float(np.mean(log[:, 2] < -0.05)), 1)}
    if outcome != "pass":
        save_plot(result, log, line, config)
    return result


def save_plot(result, log, line, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = ROOT / config["paths"]["evaluation_dir"] / f"retrain_{config['retrain']['version']}" / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 9))
    axis.plot(line[:, 0], line[:, 1], color="0.8", linewidth=6)
    points = axis.scatter(log[:, 0], log[:, 1], c=log[:, 2], cmap="coolwarm", vmin=-0.52, vmax=0.52, s=3)
    axis.plot(log[0, 0], log[0, 1], "g^", markersize=10, label="start")
    axis.plot(log[-1, 0], log[-1, 1], "kx", markersize=12, label=result["outcome"])
    tail = log[-90:]  # zoom on the last 3 s
    pad = 3.0
    axis.set_xlim(tail[:, 0].min() - pad, tail[:, 0].max() + pad)
    axis.set_ylim(tail[:, 1].min() - pad, tail[:, 1].max() + pad)
    axis.set_aspect("equal")
    axis.legend()
    fig.colorbar(points, label="steer rad (+ = left)")
    axis.set_title(f"{result['method']} {result['variant']} {result['direction']} start {result['start_index']} "
                   f"delay {result['delay']}: {result['outcome']} at lap {result['lap_fraction']}")
    name = f"{result['method']}_{result['variant']}_{result['direction']}_e{result['start_index']}_d{result['delay']}.png"
    fig.savefig(folder / name, dpi=90)
    plt.close(fig)


def summarize(results, skipped):
    lines = []

    def table(title, key):
        lines.append(f"\n== {title}")
        groups = {}
        for r in results:
            groups.setdefault((r["method"], key(r)), []).append(r)
        for (method, group), items in sorted(groups.items(), key=lambda kv: (str(kv[0][1]), kv[0][0])):
            passed = [r for r in items if r["outcome"] == "pass"]
            lap = np.mean([r["time_s"] for r in passed]) if passed else float("nan")
            lines.append(
                f"  {str(group):<22} {method:<8} pass {len(passed):>3}/{len(items):<3} "
                f"collision {sum(r['outcome'] == 'collision' for r in items):>3} "
                f"timeout {sum(r['outcome'] == 'timeout' for r in items):>3} "
                f"mean lap {np.mean([r['lap_fraction'] for r in items]):.2f} "
                f"lap time {lap:6.1f}s  speed {np.mean([r['mean_speed_mps'] for r in items]):.2f}")

    table("overall", lambda r: "all")
    table("by width (m)", lambda r: r["width"])
    table("by direction", lambda r: r["direction"])
    table("by delay (control steps)", lambda r: r["delay"])
    table("by track", lambda r: r["track"])
    if skipped:
        lines.append("\n== not tested (no usable center lane)")
        lines += [f"  {item}" for item in skipped]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=["retrain", "ppo", "expert"])
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    config = C.load_config()
    runs, skipped = build_runs(config, args.methods)
    workers = args.workers or config["retrain"].get("workers", 4)
    output_dir = ROOT / config["paths"]["evaluation_dir"] / f"retrain_{config['retrain']['version']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(runs)} runs with {workers} workers -> {output_dir}", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(run_one, run, config): run for run in runs}
        for count, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if count % 20 == 0 or count == len(runs):
                print(f"  {count}/{len(runs)} done", flush=True)

    results.sort(key=lambda r: (r["method"], r["variant"], r["direction"], r["start_index"], r["delay"]))
    (output_dir / "results.json").write_text(json.dumps({"skipped": skipped, "runs": results}, indent=2) + "\n")
    text = summarize(results, skipped)
    (output_dir / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
