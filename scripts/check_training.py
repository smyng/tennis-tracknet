#!/usr/bin/env python3
"""Monitor TrackNetV3 Training Progress"""

import os
import re
import subprocess
from datetime import datetime

LOG_FILE = "/tmp/training_tennis_v1.log"
CHECKPOINT_DIR = "exps/tennis_v1"

def check_process():
    """Check if training is running"""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if 'train.py' in line and 'grep' not in line:
                parts = line.split()
                return {
                    'running': True,
                    'cpu': parts[2] + '%',
                    'mem': parts[3] + '%',
                    'time': parts[9]
                }
        return {'running': False}
    except:
        return {'running': False}

def parse_log():
    """Parse training log for progress"""
    if not os.path.exists(LOG_FILE):
        return None

    with open(LOG_FILE, 'r') as f:
        log_content = f.read()

    # Find current epoch
    epoch_matches = re.findall(r'Epoch \[(\d+) / (\d+)\]', log_content)
    current_epoch = epoch_matches[-1] if epoch_matches else None

    # Find loss values
    loss_matches = re.findall(r'Train Loss: ([\d.]+), Val Loss: ([\d.]+)', log_content)

    # Find epoch runtimes
    runtime_matches = re.findall(r'Epoch runtime: ([\d.]+) hrs', log_content)

    return {
        'current_epoch': current_epoch,
        'losses': loss_matches,
        'runtimes': runtime_matches
    }

def format_losses(losses):
    """Format loss history as a simple ASCII chart"""
    if not losses:
        return "No loss data yet (training in progress...)"

    output = []
    output.append(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Trend'}")
    output.append("-" * 50)

    for i, (train, val) in enumerate(losses[-10:], 1):  # Last 10 epochs
        train_f = float(train)
        val_f = float(val)

        # Simple trend indicator
        if i > 1:
            prev_val = float(losses[i-2][1])
            trend = "↓ improving" if val_f < prev_val else "↑ rising" if val_f > prev_val else "→ stable"
        else:
            trend = "—"

        output.append(f"{i:<8} {train_f:<12.4f} {val_f:<12.4f} {trend}")

    return "\n".join(output)

def main():
    print("=" * 60)
    print("TrackNetV3 Tennis Training Monitor")
    print("=" * 60)
    print()

    # Check process
    proc = check_process()
    if proc['running']:
        print(f"✅ Status: TRAINING ACTIVE")
        print(f"   Runtime: {proc['time']}")
        print(f"   CPU: {proc['cpu']}, Memory: {proc['mem']}")
    else:
        print("⚠️  Status: NOT RUNNING")

    print()
    print("=" * 60)

    # Parse log
    data = parse_log()
    if data and data['current_epoch']:
        current, total = data['current_epoch']
        progress = int(current) * 100 // int(total)
        print(f"Progress: Epoch {current}/{total} ({progress}%)")

        if data['runtimes']:
            avg_time = sum(float(t) for t in data['runtimes']) / len(data['runtimes'])
            remaining = (int(total) - int(current)) * avg_time
            print(f"Avg epoch time: {avg_time:.2f} hrs")
            print(f"Est. remaining: {remaining:.1f} hrs (~{remaining/24:.1f} days)")
    else:
        print("Progress: Initializing...")

    print()
    print("=" * 60)
    print("Loss History")
    print("=" * 60)

    if data and data['losses']:
        print(format_losses(data['losses']))
        print()

        # Latest loss
        latest_train, latest_val = data['losses'][-1]
        print(f"Latest - Train: {float(latest_train):.4f}, Val: {float(latest_val):.4f}")

        # Check for overfitting
        if float(latest_val) > float(latest_train) * 1.5:
            print("⚠️  Warning: Validation loss much higher than training (possible overfitting)")
        elif float(latest_val) < float(latest_train):
            print("✅ Healthy: Validation loss lower than training")
    else:
        print("No loss data yet (epoch 1 in progress...)")

    print()
    print("=" * 60)
    print("Checkpoints")
    print("=" * 60)

    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')]
        if checkpoints:
            for ckpt in sorted(checkpoints):
                path = os.path.join(CHECKPOINT_DIR, ckpt)
                size = os.path.getsize(path) / (1024**2)  # MB
                mtime = datetime.fromtimestamp(os.path.getmtime(path))
                print(f"  {ckpt}: {size:.1f}MB (saved {mtime.strftime('%H:%M:%S')})")
        else:
            print("  No checkpoints yet (saved every 5 epochs)")
    else:
        print("  Checkpoint directory not created yet")

    print()
    print("=" * 60)
    print("Quick Commands")
    print("=" * 60)
    print(f"  Monitor:  python {__file__}")
    print(f"  Watch:    tail -f {LOG_FILE}")
    print(f"  Stop:     pkill -f train.py")
    print("=" * 60)

if __name__ == '__main__':
    main()
