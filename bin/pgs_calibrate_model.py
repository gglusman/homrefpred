#!/usr/bin/env python3
"""
pgs_calibrate_model.py

Fit a logistic regression model for P(callable) from the interval table
produced by pgs_calibration_intervals.py.

Uses all chromosomes for training.
(or user-specified splits).  Outputs:
  - Learned model coefficients (JSON) for use in pgs_score.py
  - Calibration table and plots data
  - Discrimination metrics (AUC, log-loss)

The model predicts callable_fraction from four features:
  - log(dp_mean + 1)         : flanking read depth
  - log(length + 1)          : interval length (gap size)
  - log(local_density + 1e-6): local variant density
  - dp_mean * length interaction (optional)

Each interval is weighted by its length (longer intervals = more bases
= more information).

Usage
-----
    python pgs_calibrate_model.py \\
        --intervals calibration_intervals.tsv \\
        --out-model model.json \\
        --out-eval  calibration_eval.tsv
"""

import argparse
import json
import math
import sys


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_intervals(path, min_length=10):
    """Load interval table, skipping very short intervals."""
    intervals = []
    with open(path) as fh:
        header = fh.readline().strip().split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line in fh:
            fields = line.strip().split("\t")
            try:
                row = {
                    "chrom": fields[col["chrom"]],
                    "start": int(fields[col["start"]]),
                    "end": int(fields[col["end"]]),
                    "length": int(fields[col["length"]]),
                    "dp_left": float(fields[col["dp_left"]]),
                    "dp_right": float(fields[col["dp_right"]]),
                    "dp_mean": float(fields[col["dp_mean"]]),
                    "local_density": float(fields[col["local_density"]]),
                    "n_callable": int(fields[col["n_callable"]]),
                    "n_uncallable": int(fields[col["n_uncallable"]]),
                    "n_variant": int(fields[col["n_variant"]]),
                    "callable_fraction": float(fields[col["callable_fraction"]]),
                }
            except (ValueError, IndexError, KeyError):
                continue

            if row["length"] >= min_length:
                intervals.append(row)

    return intervals


def chrom_number(chrom):
    """Extract numeric chromosome for train/test splitting."""
    c = chrom.replace("chr", "")
    try:
        return int(c)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(row):
    """
    Extract feature vector from an interval row.

    Features (all log-transformed for better linearity):
      0: log(dp_mean + 1)
      1: log(length + 1)
      2: log(local_density + 1e-6)
      3: interaction: log(dp_mean+1) * log(length+1)
      4: bias (intercept)
    """
    f0 = math.log(row["dp_mean"] + 1)
    f1 = math.log(row["length"] + 1)
    f2 = math.log(row["local_density"] + 1e-6)
    f3 = f0 * f1  # interaction: high DP + long gap = still callable
    return [f0, f1, f2, f3, 1.0]  # last is intercept


FEATURE_NAMES = ["log_dp_mean", "log_length", "log_density",
                 "dp_x_length", "intercept"]


# ---------------------------------------------------------------------------
# Logistic regression (hand-rolled, no dependencies)
# ---------------------------------------------------------------------------

def sigmoid(z):
    if z > 500:
        return 1.0
    if z < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def predict(features, weights):
    z = sum(f * w for f, w in zip(features, weights))
    return sigmoid(z)


def weighted_log_loss(data, weights):
    """Compute base-weighted log-loss."""
    total_loss = 0.0
    total_weight = 0.0
    eps = 1e-10
    for row in data:
        x = extract_features(row)
        p = predict(x, weights)
        n = row["length"]
        y = row["callable_fraction"]
        # Binary cross-entropy weighted by interval length
        loss = -(y * math.log(max(p, eps)) +
                 (1 - y) * math.log(max(1 - p, eps)))
        total_loss += loss * n
        total_weight += n
    return total_loss / total_weight if total_weight > 0 else 0.0


def fit_logistic_regression(train_data, lr=0.01, epochs=200,
                            verbose=True):
    """
    Fit weighted logistic regression using gradient descent.

    Each interval contributes a gradient weighted by its length.
    The target is callable_fraction (a continuous value in [0,1]),
    treated as the probability in a Bernoulli log-likelihood.
    This is equivalent to a quasi-binomial GLM.
    """
    n_features = len(extract_features(train_data[0]))
    weights = [0.0] * n_features

    total_bases = sum(r["length"] for r in train_data)

    for epoch in range(epochs):
        grad = [0.0] * n_features

        for row in train_data:
            x = extract_features(row)
            p = predict(x, weights)
            y = row["callable_fraction"]
            n = row["length"]

            # Gradient of cross-entropy: (p - y) * x * weight
            err = (p - y) * (n / total_bases)
            for j in range(n_features):
                grad[j] += err * x[j]

        # Update
        for j in range(n_features):
            weights[j] -= lr * grad[j]

        if verbose and (epoch + 1) % 50 == 0:
            ll = weighted_log_loss(train_data, weights)
            print(f"    Epoch {epoch+1:4d}: log-loss = {ll:.6f}",
                  file=sys.stderr)

    return weights


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(data, weights, label=""):
    """
    Compute evaluation metrics and calibration table.

    Returns dict with metrics and calibration bins.
    """
    n_bins = 20
    bin_bases = [0] * n_bins
    bin_callable = [0] * n_bins
    bin_predicted_sum = [0.0] * n_bins
    bin_count = [0] * n_bins

    total_loss = 0.0
    total_bases = 0
    eps = 1e-10

    # For AUC: collect (predicted, observed, weight) per interval
    scored = []

    for row in data:
        x = extract_features(row)
        p = predict(x, weights)
        y = row["callable_fraction"]
        n = row["length"]

        loss = -(y * math.log(max(p, eps)) +
                 (1 - y) * math.log(max(1 - p, eps)))
        total_loss += loss * n
        total_bases += n

        b = min(int(p * n_bins), n_bins - 1)
        bin_bases[b] += n
        bin_callable[b] += row["n_callable"]
        bin_predicted_sum[b] += p * n
        bin_count[b] += 1

        scored.append((p, y, n))

    avg_loss = total_loss / total_bases if total_bases > 0 else 0.0

    # Calibration table
    calibration = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if bin_bases[b] > 0:
            obs = bin_callable[b] / bin_bases[b]
            pred = bin_predicted_sum[b] / bin_bases[b]
        else:
            obs = pred = 0.0
        calibration.append({
            "bin_lo": lo, "bin_hi": hi,
            "n_intervals": bin_count[b],
            "n_bases": bin_bases[b],
            "pred_mean": pred,
            "obs_mean": obs,
        })

    # Weighted AUC (approximate, using interval-level binary threshold)
    # Treat intervals with callable_fraction >= 0.5 as "callable"
    scored.sort(key=lambda x: -x[0])  # sort by descending predicted
    tp = 0.0
    fp = 0.0
    total_pos = sum(w for _, y, w in scored if y >= 0.5)
    total_neg = sum(w for _, y, w in scored if y < 0.5)
    auc = 0.0
    prev_fp = 0.0

    if total_pos > 0 and total_neg > 0:
        for p, y, w in scored:
            if y >= 0.5:
                tp += w
            else:
                fp += w
                auc += tp * w  # trapezoidal area contribution
        auc /= (total_pos * total_neg)

    result = {
        "label": label,
        "n_intervals": len(data),
        "n_bases": total_bases,
        "log_loss": avg_loss,
        "auc": auc,
        "calibration": calibration,
    }

    return result


def print_eval(result, file=sys.stderr):
    label = result["label"]
    print(f"\n{'='*60}", file=file)
    print(f"Evaluation: {label}", file=file)
    print(f"{'='*60}", file=file)
    print(f"  Intervals: {result['n_intervals']:,}", file=file)
    print(f"  Bases:     {result['n_bases']:,}", file=file)
    print(f"  Log-loss:  {result['log_loss']:.6f}", file=file)
    print(f"  AUC:       {result['auc']:.4f}", file=file)

    print(f"\n  Calibration:", file=file)
    print(f"  {'Bin':>14} {'N_int':>8} {'Bases':>12} "
          f"{'Pred':>8} {'Obs':>8} {'Diff':>8}", file=file)
    print(f"  {'-'*14} {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*8}",
          file=file)
    for c in result["calibration"]:
        label_str = f"[{c['bin_lo']:.2f},{c['bin_hi']:.2f})"
        diff = c["pred_mean"] - c["obs_mean"]
        print(f"  {label_str:>14} {c['n_intervals']:>8,} "
              f"{c['n_bases']:>12,} {c['pred_mean']:>8.4f} "
              f"{c['obs_mean']:>8.4f} {diff:>+8.4f}", file=file)


# ---------------------------------------------------------------------------
# Hand-tuned model (for comparison baseline)
# ---------------------------------------------------------------------------

def predict_handtuned(row, genome_variant_rate=0.001, min_dp_callable=5):
    dp = row["dp_mean"]
    dp_score = 1.0 / (1.0 + math.exp(-0.5 * (dp - min_dp_callable)))

    expected_gap = 1.0 / genome_variant_rate if genome_variant_rate > 0 else 1000
    span_ratio = row["length"] / (3.0 * expected_gap)
    gap_score = 1.0 / (1.0 + span_ratio ** 2)

    density_ratio = (row["local_density"] / genome_variant_rate
                     if genome_variant_rate > 0 else 1.0)
    density_score = min(1.0, density_ratio)

    raw = (dp_score ** 0.25) * (gap_score ** 0.35) * (density_score ** 0.25)
    return max(0.01, min(0.99, raw))


def evaluate_handtuned(data, label="Hand-tuned model"):
    """Evaluate the hand-tuned model on the same data for comparison."""
    n_bins = 20
    bin_bases = [0] * n_bins
    bin_callable = [0] * n_bins
    bin_predicted_sum = [0.0] * n_bins
    bin_count = [0] * n_bins

    total_loss = 0.0
    total_bases = 0
    eps = 1e-10
    scored = []

    for row in data:
        p = predict_handtuned(row)
        y = row["callable_fraction"]
        n = row["length"]

        loss = -(y * math.log(max(p, eps)) +
                 (1 - y) * math.log(max(1 - p, eps)))
        total_loss += loss * n
        total_bases += n

        b = min(int(p * n_bins), n_bins - 1)
        bin_bases[b] += n
        bin_callable[b] += row["n_callable"]
        bin_predicted_sum[b] += p * n
        bin_count[b] += 1
        scored.append((p, y, n))

    avg_loss = total_loss / total_bases if total_bases > 0 else 0.0

    calibration = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if bin_bases[b] > 0:
            obs = bin_callable[b] / bin_bases[b]
            pred = bin_predicted_sum[b] / bin_bases[b]
        else:
            obs = pred = 0.0
        calibration.append({
            "bin_lo": lo, "bin_hi": hi,
            "n_intervals": bin_count[b],
            "n_bases": bin_bases[b],
            "pred_mean": pred,
            "obs_mean": obs,
        })

    scored.sort(key=lambda x: -x[0])
    tp = fp = 0.0
    total_pos = sum(w for _, y, w in scored if y >= 0.5)
    total_neg = sum(w for _, y, w in scored if y < 0.5)
    auc = 0.0
    if total_pos > 0 and total_neg > 0:
        for p, y, w in scored:
            if y >= 0.5:
                tp += w
            else:
                fp += w
                auc += tp * w
        auc /= (total_pos * total_neg)

    return {
        "label": label,
        "n_intervals": len(data),
        "n_bases": total_bases,
        "log_loss": avg_loss,
        "auc": auc,
        "calibration": calibration,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate P(hom-ref) model from interval data.",
    )
    parser.add_argument("--intervals", required=True,
                        help="Interval table from pgs_calibration_intervals.py")
    parser.add_argument("--out-model", required=True,
                        help="Output model coefficients (JSON)")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="Learning rate (default: 0.5)")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Training epochs (default: 500)")
    parser.add_argument("--min-length", type=int, default=10,
                        help="Min interval length to include (default: 10)")

    args = parser.parse_args()

    # Load data
    print("Loading intervals...", file=sys.stderr)
    data = load_intervals(args.intervals, args.min_length)
    print(f"  {len(data):,} intervals loaded", file=sys.stderr)

    # Don't split by chromosome
    train = [r for r in data if chrom_number(r["chrom"]) is not None ]

    print(f"  Train (all chroms):  {len(train):,} intervals, "
          f"{sum(r['length'] for r in train):,} bases", file=sys.stderr)

    # Fit logistic regression
    print("\nFitting logistic regression...", file=sys.stderr)
    weights = fit_logistic_regression(train, lr=args.lr, epochs=args.epochs)

    print(f"\n  Learned coefficients:", file=sys.stderr)
    for name, w in zip(FEATURE_NAMES, weights):
        print(f"    {name:>15}: {w:+.6f}", file=sys.stderr)

    # Save model
    model = {
        "type": "logistic_regression",
        "features": FEATURE_NAMES,
        "weights": weights,
        "feature_transforms": {
            "log_dp_mean": "log(dp_mean + 1)",
            "log_length": "log(interval_length + 1)",
            "log_density": "log(local_density + 1e-6)",
            "dp_x_length": "log(dp_mean+1) * log(length+1)",
            "intercept": "1.0",
        },
        "training": {
            "n_intervals": len(train),
            "n_bases": sum(r["length"] for r in train),
            "chromosomes": "all",
        },
    }

    with open(args.out_model, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nModel saved to {args.out_model}", file=sys.stderr)



if __name__ == "__main__":
    main()

