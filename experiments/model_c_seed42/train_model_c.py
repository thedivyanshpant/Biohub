#!/usr/bin/env python
"""
train_model_c.py — Model C (Seed 42) Clean Reproduction Training Pipeline

Architecture: TemporalUNet3D + SimpleNodeTransformer
Hardware: 4x NVIDIA A100 80GB (2 nodes × 2 GPUs) via PyTorch DDP
Seed: 42
Validation: 8 held-out volumes (clean separation, 0 test leakage)
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import yaml

# Fix for PyTorch 2.7 SDPA CUDA kernel grid limit with T=2 and large spatial volume
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

PROJECT_ROOT = Path(os.environ.get("BIOHUB_ROOT", Path(__file__).resolve().parents[2]))
REPO_DIR = Path(os.environ.get("BIOHUB_TRACKING_REPO",
    PROJECT_ROOT / "kaggle_source" / "tracking_repo" if (PROJECT_ROOT / "kaggle_source" / "tracking_repo").exists()
    else PROJECT_ROOT / "code" / "leader_reproduction" / "working" / "tracking_repo"
))
EXP_DIR = PROJECT_ROOT / "experiments" / "model_c_seed42"

sys.path.insert(0, str(REPO_DIR / "scripts"))
sys.path.insert(0, str(REPO_DIR / "src"))

from biohub_tracking.models import TemporalUNet3D
from train_unet_transformer import (
    UNetNodeTransformer,
    load_dataset_windows,
    FrameWindowDataset,
    compute_detection_loss,
    detect_and_match,
    build_matched_edge_targets,
    compute_batch_loss,
    _evaluate_pair,
    DEFAULT_AUGMENTATIONS,
    _POS_EMBED_DIM,
)


class DDPUNetNodeTransformer(UNetNodeTransformer):
    """UNetNodeTransformer wrapper with forward method for PyTorch DDP."""

    def __init__(self, unet, unet_out_channels, pos_feat_dim, pool_kernel_um=5.0):
        super().__init__(unet=unet, unet_out_channels=unet_out_channels, pos_feat_dim=pos_feat_dim)
        self.pool_kernel_um = pool_kernel_um

    def forward(
        self,
        imgs: torch.Tensor,
        coords: torch.Tensor,
        masks: torch.Tensor,
        targets: torch.Tensor,
        image_shape: tuple,
        voxel_size: tuple,
        ds_scale: torch.Tensor,
        det_loss_weight: float = 1.0,
        det_neg_weight: float = 0.01,
    ):
        window = imgs.unsqueeze(2)  # (B, W, 1, *spatial)
        unet_out = self.unet(window)
        W = unet_out.shape[1]
        det_logits = [self.detect_head(unet_out[:, i]) for i in range(W)]

        det_losses = [
            compute_detection_loss(det_logits[i], coords[:, i], masks[:, i], det_neg_weight)
            for i in range(W)
        ]
        det_loss = sum(det_losses) / W

        frame_det = []
        for i in range(W):
            det_c, det_p, det_m, matches = detect_and_match(
                det_logits[i], coords[:, i], masks[:, i],
                image_shape, voxel_size=voxel_size,
                pool_kernel_um=self.pool_kernel_um,
                frame_index=i, window_size=W,
            )
            unet_feat = self._index_features(unet_out[:, i], det_c, det_m)
            frame_det.append((det_c, det_p, det_m, matches, unet_feat))

        block_losses = []
        for i in range(W - 1):
            ns = frame_det[i][0].shape[1]
            nt = frame_det[i + 1][0].shape[1]
            pair_target = build_matched_edge_targets(
                frame_det[i][3], frame_det[i + 1][3],
                targets[:, i], ns, nt,
            )
            edge_logits = self.predict_edges(
                frame_det[i][4], frame_det[i + 1][4],
                frame_det[i][0] * ds_scale, frame_det[i + 1][0] * ds_scale,
                frame_det[i][1], frame_det[i + 1][1],
                frame_det[i][2], frame_det[i + 1][2],
            )
            block_losses.append(
                compute_batch_loss(edge_logits, pair_target, frame_det[i][2], frame_det[i + 1][2])
            )
        edge_loss = sum(block_losses) / len(block_losses)
        total_loss = edge_loss + det_loss_weight * det_loss
        return total_loss, edge_loss, det_loss


@torch.no_grad()
def evaluate_ddp(model, loader, device, is_distributed=False, pool_kernel_um=5.0):
    model.eval()
    total_loss, correct, total, n_pairs = 0.0, 0, 0, 0
    gt_matched, gt_total = 0, 0

    m = model.module if hasattr(model, "module") else model

    for batch in loader:
        imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        pos_feats = batch["pos_feats"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        image_shape = tuple(batch["image_shape"][0].tolist())
        voxel_size = tuple(batch["voxel_size"][0].tolist())
        ds_scale = batch["downsample"][0].to(device)

        B, W = imgs.shape[:2]
        unet_out, det_logits = m.encode(imgs)
        frame_det = []
        for i in range(W):
            det_c, det_p, det_m, matches = detect_and_match(
                det_logits[i], coords[:, i], masks[:, i],
                image_shape, voxel_size=voxel_size,
                pool_kernel_um=pool_kernel_um,
                frame_index=i, window_size=W,
            )
            unet_feat = m._index_features(unet_out[:, i], det_c, det_m)
            frame_det.append((det_c, det_p, det_m, matches, unet_feat))

            for b in range(B):
                n_gt = int(masks[b, i].sum().item())
                n_matched = (matches[b] >= 0).sum().item()
                gt_total += n_gt
                gt_matched += n_matched

        for i in range(W - 1):
            ns = frame_det[i][0].shape[1]
            nt = frame_det[i + 1][0].shape[1]
            pair_target = build_matched_edge_targets(
                frame_det[i][3], frame_det[i + 1][3],
                targets[:, i], ns, nt,
            )
            pair_logits = m.predict_edges(
                frame_det[i][4], frame_det[i + 1][4],
                frame_det[i][0] * ds_scale, frame_det[i + 1][0] * ds_scale,
                frame_det[i][1], frame_det[i + 1][1],
                frame_det[i][2], frame_det[i + 1][2],
            )

            for b in range(B):
                ns_b = int(frame_det[i][2][b].sum().item())
                nt_b = int(frame_det[i + 1][2][b].sum().item())
                pair_loss, pair_correct, pair_total = _evaluate_pair(
                    pair_logits[b, :ns_b, :nt_b], pair_target[b, :ns_b, :nt_b],
                )
                total_loss += pair_loss
                correct += pair_correct
                total += pair_total
                n_pairs += 1

    if is_distributed:
        metrics_tensor = torch.tensor(
            [total_loss, correct, total, n_pairs, gt_matched, gt_total],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(metrics_tensor)
        total_loss, correct, total, n_pairs, gt_matched, gt_total = metrics_tensor.tolist()

    avg_loss = total_loss / max(n_pairs, 1)
    acc = correct / max(total, 1)
    recall = gt_matched / max(gt_total, 1)
    return avg_loss, acc, recall


def main():
    parser = argparse.ArgumentParser(description="Train Model C Seed 42")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16, help="Effective batch size across all GPUs")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--det-loss-weight", type=float, default=1.0)
    parser.add_argument("--det-neg-weight", type=float, default=0.01)
    parser.add_argument("--downsample", type=str, default="1,4,4")
    parser.add_argument("--unet-layers", type=str, default="32,64,128")
    parser.add_argument("--unet-out-channels", type=int, default=32)
    parser.add_argument("--pool-kernel-um", type=float, default=5.0)
    parser.add_argument("--master-port", type=str, default="29520")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if available")
    args = parser.parse_args()

    # Setup distributed environment
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    is_distributed = world_size > 1

    if is_distributed:
        os.environ["MASTER_ADDR"] = "172.6.3.7"
        os.environ["MASTER_PORT"] = args.master_port
        os.environ["NCCL_IB_DISABLE"] = "0"
        os.environ["NCCL_SOCKET_IFNAME"] = "ib0"
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set seeds
    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    is_rank0 = (rank == 0)
    log_file = EXP_DIR / "train.log"
    history_file = EXP_DIR / "training_history.csv"
    best_ckpt = EXP_DIR / "best.pt"
    last_ckpt = EXP_DIR / "last.pt"

    def log(msg):
        if is_rank0:
            print(msg, flush=True)
            with open(log_file, "a") as f:
                f.write(msg + "\n")

    if is_rank0:
        log("=" * 70)
        log("STARTING MODEL C (SEED 42) TRAINING")
        log(f"Nodes / GPUs: {world_size} GPUs across cluster ({os.uname().nodename})")
        log(f"Effective batch size: {args.batch_size} (per-GPU batch: {max(1, args.batch_size // world_size)})")
        log(f"Learning rate: {args.lr} | Optimizer: AdamW | Epochs: {args.epochs}")
        log(f"Detection loss weight: {args.det_loss_weight} | Negative weight: {args.det_neg_weight}")
        log(f"Downsample: {args.downsample} | Seed: {args.seed}")
        log("=" * 70)

        # Save config.yaml
        config_data = {
            "model": "TemporalUNet3D + SimpleNodeTransformer",
            "seed": args.seed,
            "epochs": args.epochs,
            "effective_batch_size": args.batch_size,
            "per_gpu_batch_size": max(1, args.batch_size // world_size),
            "world_size": world_size,
            "lr": args.lr,
            "optimizer": "AdamW",
            "det_loss_weight": args.det_loss_weight,
            "det_neg_weight": args.det_neg_weight,
            "downsample": [int(x) for x in args.downsample.split(",")],
            "unet_layers": [int(x) for x in args.unet_layers.split(",")],
            "unet_out_channels": args.unet_out_channels,
            "pool_kernel_um": args.pool_kernel_um,
            "validation_volumes": 8,
            "training_volumes": 187,
            "test_leakage_volumes": 0,
        }
        with open(EXP_DIR / "config.yaml", "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        # Initialize CSV header if not exists
        if not history_file.exists():
            with open(history_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "edge_loss", "det_loss", "train_loss",
                    "val_loss", "val_acc", "val_recall", "val_score", "epoch_time_s"
                ])

    # Load data from preloaded metadata cache
    cache_path = EXP_DIR / "metadata_cache.pt"
    if not cache_path.exists():
        if is_rank0:
            log(f"Metadata cache {cache_path} not found! Building it now...")
        # Fallback if needed
        time.sleep(5)

    log(f"[Rank {rank}] Loading metadata cache from {cache_path}...")
    cached = torch.load(cache_path, weights_only=False)
    train_video_data = cached["train_data"]
    val_video_data = cached["val_data"]
    max_nodes = cached["max_nodes"]

    train_ds = FrameWindowDataset(train_video_data, max_nodes=max_nodes, augmentations=DEFAULT_AUGMENTATIONS)
    val_ds = FrameWindowDataset(val_video_data, max_nodes=max_nodes)

    per_gpu_batch = max(1, args.batch_size // world_size)

    if is_distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(
            train_ds, batch_size=per_gpu_batch, sampler=train_sampler,
            num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None, pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=per_gpu_batch, sampler=val_sampler,
            num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None, pin_memory=False,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    log(f"[Rank {rank}] Dataset ready: {len(train_ds)} train windows, {len(val_ds)} val windows. max_nodes={max_nodes}")

    # Build model
    unet_layers = [int(x) for x in args.unet_layers.split(",")]
    unet = TemporalUNet3D(in_channels=1, out_channels=args.unet_out_channels, layers=unet_layers)
    pos_feat_dim = 4 * _POS_EMBED_DIM
    model = DDPUNetNodeTransformer(
        unet=unet,
        unet_out_channels=args.unet_out_channels,
        pos_feat_dim=pos_feat_dim,
        pool_kernel_um=args.pool_kernel_um,
    ).to(device)

    if is_distributed:
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    else:
        ddp_model = model

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)

    best_score = 0.0
    start_epoch = 0

    if args.resume and last_ckpt.exists():
        log(f"Resuming from checkpoint {last_ckpt}...")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        m = ddp_model.module if hasattr(ddp_model, "module") else ddp_model
        m.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_score = ckpt.get("best_score", 0.0)
        log(f"Resumed from epoch {start_epoch} (best_score={best_score:.4f})")
    elif is_rank0 and not args.resume:
        with open(history_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "edge_loss", "det_loss", "train_loss",
                "val_loss", "val_acc", "val_recall", "val_score", "epoch_time_s"
            ])

    log(f"[Rank {rank}] Model instantiated with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
    log(f"Ready. Commencing training from epoch {start_epoch+1} to {args.epochs}...")

    n_batches = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        ddp_model.train()
        total_edge_loss, total_det_loss = 0.0, 0.0
        n_samples = 0

        for b_idx, batch in enumerate(train_loader):
            imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
            coords = batch["coords"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            image_shape = tuple(batch["image_shape"][0].tolist())
            voxel_size = tuple(batch["voxel_size"][0].tolist())
            ds_scale = batch["downsample"][0].to(device)
            B = imgs.shape[0]

            optimizer.zero_grad()
            total_loss, edge_loss, det_loss = ddp_model(
                imgs, coords, masks, targets, image_shape, voxel_size, ds_scale,
                det_loss_weight=args.det_loss_weight, det_neg_weight=args.det_neg_weight,
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
            optimizer.step()

            if is_rank0 and (b_idx + 1) % 100 == 0:
                print(f"  [Epoch {epoch+1:02d}/{args.epochs} | Batch {b_idx+1:04d}/{n_batches:04d}] edge={edge_loss.item():.4f} det={det_loss.item():.4f} tot={total_loss.item():.4f}", flush=True)

            total_edge_loss += edge_loss.item() * B
            total_det_loss += det_loss.item() * B
            n_samples += B

        # Synchronize train metrics across GPUs
        if is_distributed:
            train_stats = torch.tensor([total_edge_loss, total_det_loss, n_samples], dtype=torch.float64, device=device)
            dist.all_reduce(train_stats)
            ep_edge_loss = train_stats[0].item() / max(train_stats[2].item(), 1)
            ep_det_loss = train_stats[1].item() / max(train_stats[2].item(), 1)
        else:
            ep_edge_loss = total_edge_loss / max(n_samples, 1)
            ep_det_loss = total_det_loss / max(n_samples, 1)

        ep_train_loss = ep_edge_loss + args.det_loss_weight * ep_det_loss

        # Evaluation
        val_loss, val_acc, val_recall = evaluate_ddp(
            ddp_model, val_loader, device, is_distributed=is_distributed, pool_kernel_um=args.pool_kernel_um
        )
        val_score = val_acc * val_recall
        epoch_time = time.time() - t0

        is_best = val_score > best_score
        if is_best:
            best_score = val_score

        marker = "★ BEST" if is_best else ""
        log(
            f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"Train: edge={ep_edge_loss:.4f} det={ep_det_loss:.4f} (tot={ep_train_loss:.4f}) | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} rec={val_recall:.4f} score={val_score:.4f} | "
            f"{epoch_time:.1f}s {marker}"
        )

        # Checkpoint saving & history logging on Rank 0
        if is_rank0:
            with open(history_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, round(ep_edge_loss, 5), round(ep_det_loss, 5), round(ep_train_loss, 5),
                    round(val_loss, 5), round(val_acc, 5), round(val_recall, 5), round(val_score, 5),
                    round(epoch_time, 2)
                ])

            raw_model = ddp_model.module if hasattr(ddp_model, "module") else ddp_model
            clean_state = {k.replace("module.", "", 1): v.cpu() for k, v in raw_model.state_dict().items()}

            checkpoint_meta = {
                "epoch": epoch + 1,
                "model_state_dict": clean_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_score": val_score,
                "val_acc": val_acc,
                "val_recall": val_recall,
                "best_score": best_score,
                "seed": args.seed,
            }
            torch.save(checkpoint_meta, last_ckpt)

            if is_best:
                torch.save(checkpoint_meta, best_ckpt)
                # Also save standard format edge_predictor_best.pth
                torch.save(clean_state, EXP_DIR / "edge_predictor_best.pth")

        if is_distributed:
            dist.barrier()

    log("=" * 70)
    log(f"TRAINING COMPLETE! Best Validation Score: {best_score:.4f}")
    log(f"Best checkpoint saved to: {best_ckpt}")
    log("=" * 70)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
