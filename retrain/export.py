"""Export a BC checkpoint to TorchScript for the ROS 2 policy_node (step 4 of retrain/run_all.sh).

Same interface as end2race_ros2/scripts/export_model.py:
    (lidar [B,180] m, speed [B,1] m/s, h_in [1,B,420]) -> (action [B,2], h_out [1,B,420])
action = [steering rad (positive = left), speed m/s].

Checks:
  1. Stepping the exported module 240 times gives the same actions as running
     imitation.model.End2Race on the whole sequence (max difference printed).
  2. The saved file loads back and gives the same output.

Usage: python retrain/export.py [--checkpoint checkpoint/bc_retrain_v1.pt] [--fp16]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import yaml

from imitation.model import End2Race


class SingleStepPolicy(nn.Module):
    """One control step of End2Race (eval mode, no speed masking)."""

    def __init__(self, model: End2Race):
        super().__init__()
        self.model = model

    def forward(self, lidar, speed, h_in):
        actions, h_out = self.model(lidar.unsqueeze(1), speed.unsqueeze(1), h_in)
        return actions.squeeze(1), h_out


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    version = config["retrain"]["version"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoint" / f"bc_retrain_{version}.pt"))
    parser.add_argument("--fp16", action="store_true", help="also write an FP16 copy (for Jetson CUDA)")
    args = parser.parse_args()

    model = End2Race()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    policy = SingleStepPolicy(model).eval()

    lidar0 = torch.full((1, 180), 5.0)
    speed0 = torch.full((1, 1), 2.0)
    h0 = torch.zeros((1, 1, End2Race.GRU_HIDDEN_SIZE))
    with torch.no_grad():
        traced = torch.jit.freeze(torch.jit.trace(policy, (lidar0, speed0, h0)).eval())

    # Check 1: step by step == full sequence, on random but realistic inputs
    torch.manual_seed(0)
    lidar_seq = torch.rand(1, 240, 180) * 9.5 + 0.3
    speed_seq = torch.rand(1, 240, 1) * 5.0
    with torch.no_grad():
        full, _ = model(lidar_seq, speed_seq)
        hidden, steps = h0, []
        for t in range(240):
            action, hidden = traced(lidar_seq[:, t], speed_seq[:, t], hidden)
            steps.append(action)
    difference = (torch.stack(steps, 1) - full).abs().max().item()
    print(f"step-by-step vs full sequence: max difference {difference:.2e}")
    if difference > 1e-4:
        raise RuntimeError("Exported module does not match the trained model")

    output_dir = ROOT / "checkpoint" / "export"
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / f"end2race_retrain_{version}_jit_fp32.pt"
    traced.save(str(fp32_path))
    loaded = torch.jit.load(str(fp32_path), map_location="cpu")
    with torch.no_grad():
        check = (loaded(lidar0, speed0, h0)[0] - traced(lidar0, speed0, h0)[0]).abs().max().item()
    print(f"Saved {fp32_path} (reload difference {check:.1e}), sample action {loaded(lidar0, speed0, h0)[0].tolist()}")

    if args.fp16:
        if not torch.cuda.is_available():
            raise RuntimeError("FP16 export needs CUDA to verify")
        half = SingleStepPolicy(model).cuda().half().eval()
        with torch.no_grad():
            traced_half = torch.jit.freeze(torch.jit.trace(
                half, (lidar0.cuda().half(), speed0.cuda().half(), h0.cuda().half())).eval())
            out = traced_half(lidar0.cuda().half(), speed0.cuda().half(), h0.cuda().half())[0].float().cpu()
        fp16_path = output_dir / f"end2race_retrain_{version}_jit_fp16.pt"
        traced_half.save(str(fp16_path))
        print(f"Saved {fp16_path}, FP16 vs FP32 sample difference {(out - traced(lidar0, speed0, h0)[0]).abs().max().item():.1e}")


if __name__ == "__main__":
    main()
