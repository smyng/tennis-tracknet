#!/usr/bin/env python3
"""Quick benchmark: batch_size=4 vs batch_size=8 for TrackNet training."""

import os
import sys
import time
import gc

# Add project root to path when running from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import Shuttlecock_Trajectory_Dataset
from utils.general import get_model
from utils.metric import WBCELoss


def bench(bs, num_batches=20):
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"Benchmarking batch_size={bs}  (device={device})")
    print(f"{'='*60}")

    dataset = Shuttlecock_Trajectory_Dataset(
        split='train', seq_len=8, sliding_step=1,
        data_mode='heatmap', bg_mode='subtract_concat',
        frame_alpha=0.5, debug=False,
    )
    # Use num_workers=0 to avoid macOS multiprocessing fork issues
    loader = DataLoader(
        dataset, batch_size=bs, shuffle=True,
        num_workers=0, drop_last=True,
    )

    model = get_model('TrackNet', 8, 'subtract_concat').to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    times = []
    load_times = []
    mem_peak = 0

    t_load_start = time.time()
    for step, (_, x, y, c, _) in enumerate(loader):
        if step >= num_batches:
            break

        t_load = time.time() - t_load_start
        load_times.append(t_load)

        t0 = time.time()
        optimizer.zero_grad()
        x, y = x.float().to(device), y.float().to(device)
        y_pred = model(x)
        loss = WBCELoss(y_pred, y)
        loss.backward()
        optimizer.step()
        if device == 'mps':
            torch.mps.synchronize()
        dt = time.time() - t0
        times.append(dt)

        # Check MPS memory
        if device == 'mps' and hasattr(torch.mps, 'current_allocated_memory'):
            mem = torch.mps.current_allocated_memory() / 1024**3
            mem_peak = max(mem_peak, mem)

        print(f"  batch {step+1:3d}: load={t_load:.2f}s  gpu={dt:.2f}s  total={t_load+dt:.2f}s  loss={loss.item():.4f}", flush=True)
        t_load_start = time.time()

    # Skip first 2 batches (warmup)
    steady_gpu = times[2:] if len(times) > 2 else times
    steady_load = load_times[2:] if len(load_times) > 2 else load_times
    avg_gpu = np.mean(steady_gpu)
    avg_load = np.mean(steady_load)
    avg_total = avg_gpu + avg_load
    total_batches = len(dataset) // bs
    epoch_est = total_batches * avg_total / 3600

    print(f"\n  Results for batch_size={bs}:")
    print(f"    Avg data load time:      {avg_load:.2f}s")
    print(f"    Avg GPU time/batch:      {avg_gpu:.2f}s")
    print(f"    Avg total time/batch:    {avg_total:.2f}s")
    print(f"    Samples/sec:             {bs/avg_total:.1f}")
    print(f"    Batches/epoch:           {total_batches}")
    print(f"    Est. epoch time:         {epoch_est:.1f} hrs")
    print(f"    Est. 30-epoch time:      {epoch_est*30:.1f} hrs ({epoch_est*30/24:.1f} days)")
    if mem_peak > 0:
        print(f"    Peak GPU memory:         {mem_peak:.2f} GB")

    # Clean up
    del model, optimizer, loader, dataset
    if device == 'mps':
        torch.mps.empty_cache()
    gc.collect()
    time.sleep(2)

    return {
        'batch_size': bs,
        'samples_per_sec': bs / avg_total,
        'avg_total': avg_total,
        'avg_gpu': avg_gpu,
        'avg_load': avg_load,
        'epoch_hrs': epoch_est,
        'mem_peak_gb': mem_peak,
    }


if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = []
    for bs in [4, 8]:
        r = bench(bs)
        results.append(r)

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<28s}  {'BS=4':>10s}  {'BS=8':>10s}")
    print(f"{'-'*52}")
    r4, r8 = results[0], results[1]
    print(f"{'Samples/sec':<28s}  {r4['samples_per_sec']:>10.1f}  {r8['samples_per_sec']:>10.1f}")
    print(f"{'Avg total time/batch (s)':<28s}  {r4['avg_total']:>10.2f}  {r8['avg_total']:>10.2f}")
    print(f"{'  - Data loading (s)':<28s}  {r4['avg_load']:>10.2f}  {r8['avg_load']:>10.2f}")
    print(f"{'  - GPU compute (s)':<28s}  {r4['avg_gpu']:>10.2f}  {r8['avg_gpu']:>10.2f}")
    print(f"{'Est. epoch time (hrs)':<28s}  {r4['epoch_hrs']:>10.1f}  {r8['epoch_hrs']:>10.1f}")
    print(f"{'Est. 30 epochs (days)':<28s}  {r4['epoch_hrs']*30/24:>10.1f}  {r8['epoch_hrs']*30/24:>10.1f}")
    if r4['mem_peak_gb'] > 0:
        print(f"{'Peak GPU memory (GB)':<28s}  {r4['mem_peak_gb']:>10.2f}  {r8['mem_peak_gb']:>10.2f}")

    winner = 4 if r4['samples_per_sec'] > r8['samples_per_sec'] else 8
    speedup = max(r4['samples_per_sec'], r8['samples_per_sec']) / min(r4['samples_per_sec'], r8['samples_per_sec'])
    print(f"\nWinner: batch_size={winner} ({speedup:.1f}x faster throughput)")
