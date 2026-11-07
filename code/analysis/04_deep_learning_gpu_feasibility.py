#!/usr/bin/env python3
"""
Biohub Comprehensive Discovery Pass: Part 4
3D Deep Learning Architecture Benchmarking & Multi-GPU Feasibility on NVIDIA A100 80GB.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = os.environ.get("BIOHUB_CODE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class Simple3DUNet(nn.Module):
    """Representative 3D U-Net with 4 resolution levels for cell detection/heatmap regression."""
    def __init__(self, in_channels=1, out_channels=1, base_channels=16):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))  # preserve Z resolution in early layers

        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.enc3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool3d(kernel_size=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv3d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Output head
        self.out_head = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_head(d1)


def benchmark_configuration(device, shape, batch_size, base_channels=16, amp=True, warmup=3, runs=10):
    """Benchmarks memory and throughput for a 3D U-Net configuration."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = Simple3DUNet(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Input tensor
    x = torch.randn(batch_size, 1, shape[0], shape[1], shape[2], device=device)
    target = torch.randn(batch_size, 1, shape[0], shape[1], shape[2], device=device)

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(x)
            loss = criterion(out, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize(device)
    t_start = time.time()

    for _ in range(runs):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(x)
            loss = criterion(out, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize(device)
    dur = (time.time() - t_start) / runs
    peak_vram_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    throughput_vols_sec = batch_size / dur

    return {
        "shape": shape,
        "batch_size": batch_size,
        "base_channels": base_channels,
        "step_time_ms": round(dur * 1000, 2),
        "peak_vram_gib": round(peak_vram_gib, 2),
        "throughput_vols_sec": round(throughput_vols_sec, 2),
    }


def main():
    print("================================================================================")
    print("BIOHUB DISCOVERY PASS — PART 4: 3D DEEP LEARNING GPU BENCHMARKING (A100 80GB)")
    print("================================================================================")

    if not torch.cuda.is_available():
        print("CUDA not available! Skipping GPU benchmarking.")
        return

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    total_mem_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Device: {device_name} | Total VRAM: {total_mem_gib:.1f} GiB\n")

    configs = [
        # Full volume: 64 x 256 x 256
        {"shape": (64, 256, 256), "batch_size": 1, "base_channels": 16},
        {"shape": (64, 256, 256), "batch_size": 2, "base_channels": 16},
        {"shape": (64, 256, 256), "batch_size": 4, "base_channels": 16},
        # High capacity full volume: base_channels=32
        {"shape": (64, 256, 256), "batch_size": 2, "base_channels": 32},
        # Medium patch: 64 x 128 x 128
        {"shape": (64, 128, 128), "batch_size": 4, "base_channels": 16},
        {"shape": (64, 128, 128), "batch_size": 8, "base_channels": 16},
        # Small cubic patch: 32 x 64 x 64
        {"shape": (32, 64, 64), "batch_size": 16, "base_channels": 16},
    ]

    results = []
    print(f"{'Input Shape':16s} | {'Batch':5s} | {'BaseCh':6s} | {'VRAM (GiB)':10s} | {'Step Time':10s} | {'Throughput':12s}")
    print("-" * 75)

    for cfg in configs:
        try:
            res = benchmark_configuration(
                device=device,
                shape=cfg["shape"],
                batch_size=cfg["batch_size"],
                base_channels=cfg["base_channels"],
                amp=True
            )
            results.append(res)
            print(f"{str(res['shape']):16s} | {res['batch_size']:5d} | {res['base_channels']:6d} | "
                  f"{res['peak_vram_gib']:8.2f} G | {res['step_time_ms']:7.1f} ms | {res['throughput_vols_sec']:7.2f} vol/s")
        except RuntimeError as e:
            print(f"{str(cfg['shape']):16s} | {cfg['batch_size']:5d} | OOM / Error: {e}")

    # Theoretical epoch calculations
    # Total dataset: 199 datasets * 100 frames = 19,900 full frames
    # If using full frames at batch size 2 (throughput ~12 vol/s on 1 A100):
    best_full_res = [r for r in results if r["shape"] == (64, 256, 256) and r["base_channels"] == 16][-1]
    single_gpu_throughput = best_full_res["throughput_vols_sec"]

    total_frames = 199 * 100
    epoch_times = {
        "1_GPU (A100 80GB)": f"{round(total_frames / single_gpu_throughput / 60, 1)} minutes",
        "2_GPUs (Node rpgpu018)": f"{round(total_frames / (single_gpu_throughput * 1.95) / 60, 1)} minutes",
        "4_GPUs (2 Nodes)": f"{round(total_frames / (single_gpu_throughput * 3.8) / 60, 1)} minutes",
        "8_GPUs (Full Cluster Allocation)": f"{round(total_frames / (single_gpu_throughput * 7.4) / 60, 1)} minutes",
    }

    summary_part4 = {
        "hardware": {
            "device_name": device_name,
            "total_vram_gib": round(total_mem_gib, 1),
            "allocated_cluster_gpus": "8 x NVIDIA A100 80GB across 4 nodes",
        },
        "benchmarks": results,
        "full_volume_epoch_times_19900_frames": epoch_times,
    }

    json_path = os.path.join(OUT_DIR, "analysis_part4_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_part4, f, indent=2)
    print(f"\nSaved GPU feasibility benchmarks to: {json_path}")
    print("\n--- Multi-GPU Scaling Estimates for Full-Frame 3D Training (19,900 frames / epoch) ---")
    for k, v in epoch_times.items():
        print(f"  {k:35s}: {v}")


if __name__ == "__main__":
    main()
