#!/usr/bin/env python3
"""
pgs_calibration_intervals.py

For each inter-variant interval in a standard VCF, determine the actual
callable vs. uncallable base counts from a gVCF.  Outputs a table suitable
for fitting and calibrating the P(hom-ref) model.

Approach
--------
1. Parse the standard VCF to extract variant positions and per-sample DP.
2. Define intervals between consecutive variant positions on each chromosome.
3. For each interval, compute features: flanking DP (left, right), interval
   length, local variant density.
4. Stream through the gVCF to count, within each interval, how many bases
   are "callable" (hom-ref with sufficient depth and genotype quality) vs.
   "not callable" (no coverage, low depth, or low GQ).
5. Output one row per interval with features and observed callable fraction.

The output can be used to:
  - Fit a logistic/beta regression of callable_fraction ~ features
  - Generate calibration plots (predicted P(hom-ref) vs. observed fraction)
  - Evaluate the current hand-tuned model

Usage
-----
    python pgs_calibration_intervals.py \\
        --vcf sample.vcf.gz \\
        --gvcf sample.g.vcf.gz \\
        --out intervals.tsv \\
        [--sample SAMPLE_ID] \\
        [--min-dp 5] \\
        [--min-gq 20] \\
        [--density-window 50000]

Dependencies: Python 3 standard library only.
"""

import argparse
import gzip
import sys
from collections import defaultdict


def open_maybe_gzipped(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


# ---------------------------------------------------------------------------
# VCF parsing: extract variant positions + DP per chromosome
# ---------------------------------------------------------------------------

def parse_vcf_positions(vcf_path, sample_id=None):
    """
    Parse VCF to get sorted variant positions and their DP per chromosome.

    Returns dict[chrom] -> list of (pos, dp) sorted by pos.
    """
    chrom_variants = defaultdict(list)

    with open_maybe_gzipped(vcf_path) as fh:
        sample_col = None
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.strip().split("\t")
                sample_names = header[9:]
                if sample_id and sample_id in sample_names:
                    sample_col = 9 + sample_names.index(sample_id)
                else:
                    sample_col = 9
                continue

            fields = line.rstrip("\n\r").split("\t")
            if len(fields) < sample_col + 1:
                continue

            chrom = fields[0]
            try:
                pos = int(fields[1])
            except ValueError:
                continue

            # Extract sample-level DP
            fmt_keys = fields[8].split(":")
            sample_vals = fields[sample_col].split(":")
            dp = None
            for i, key in enumerate(fmt_keys):
                if key == "DP" and i < len(sample_vals):
                    try:
                        dp = int(sample_vals[i])
                    except ValueError:
                        pass
                    break

            # Fall back to INFO DP
            if dp is None:
                for entry in fields[7].split(";"):
                    if entry.startswith("DP="):
                        try:
                            dp = int(entry[3:])
                        except ValueError:
                            pass
                        break

            chrom_variants[chrom].append((pos, dp if dp is not None else 0))

    # Sort by position
    for chrom in chrom_variants:
        chrom_variants[chrom].sort()

    return dict(chrom_variants)


# ---------------------------------------------------------------------------
# Build inter-variant intervals with features
# ---------------------------------------------------------------------------

def build_intervals(chrom_variants):
    """
    For each chromosome, build the list of inter-variant intervals.

    Each interval is defined by the gap between consecutive VCF variant
    positions.  Features are derived from the bounding variants.

    Returns dict[chrom] -> list of interval dicts, sorted by start.
    Each dict: {start, end, length, dp_left, dp_right, dp_mean}
    """
    chrom_intervals = {}

    for chrom, variants in chrom_variants.items():
        intervals = []
        for i in range(len(variants) - 1):
            pos_left, dp_left = variants[i]
            pos_right, dp_right = variants[i + 1]

            # The interval is the open region (pos_left, pos_right),
            # i.e., positions pos_left+1 through pos_right-1 inclusive.
            start = pos_left + 1
            end = pos_right  # exclusive (the right variant itself)
            length = end - start  # number of bases in the interval

            if length <= 0:
                continue  # adjacent or overlapping variants

            intervals.append({
                "chrom": chrom,
                "start": start,
                "end": end,
                "length": length,
                "dp_left": dp_left,
                "dp_right": dp_right,
                "dp_mean": (dp_left + dp_right) / 2.0,
                "idx_left": i,
                "idx_right": i + 1,
            })

        chrom_intervals[chrom] = intervals

    return chrom_intervals


def add_density_features(chrom_variants, chrom_intervals, window=50000):
    """
    Add local variant density feature to each interval.
    Density = number of VCF variants within ±window of interval midpoint,
    divided by (2 * window).
    """
    from bisect import bisect_left, bisect_right

    for chrom, intervals in chrom_intervals.items():
        if chrom not in chrom_variants:
            for iv in intervals:
                iv["local_density"] = 0.0
            continue

        positions = [p for p, _ in chrom_variants[chrom]]

        for iv in intervals:
            mid = (iv["start"] + iv["end"]) // 2
            lo = bisect_left(positions, mid - window)
            hi = bisect_right(positions, mid + window)
            iv["local_density"] = (hi - lo) / (2.0 * window)


# ---------------------------------------------------------------------------
# gVCF parsing: count callable bases per interval
# ---------------------------------------------------------------------------

def count_callable_in_intervals(gvcf_path, chrom_intervals, sample_id=None,
                                min_dp=5, min_gq=20):
    """
    Stream through the gVCF and count callable vs. not-callable bases
    within each inter-variant interval.

    A base is "callable" if it falls within a gVCF record (either a
    reference block or a variant) with:
      - GT = 0/0 (hom-ref) for reference blocks
      - DP >= min_dp (or MIN_DP for ref blocks)
      - GQ >= min_gq

    For each interval, we record:
      - n_callable: bases meeting the criteria
      - n_uncallable: bases not meeting the criteria (low depth, low GQ,
        or not covered by any gVCF record)
      - n_variant: bases that are actually variant calls in the gVCF
        (these should be rare within inter-VCF-variant intervals, but
        can occur if the gVCF has variants not in the standard VCF)

    Modifies interval dicts in place.
    """
    # Build a lookup: for each chromosome, sorted list of intervals
    # with pointers for streaming
    chrom_interval_idx = {}
    for chrom, intervals in chrom_intervals.items():
        for iv in intervals:
            iv["n_callable"] = 0
            iv["n_uncallable"] = 0
            iv["n_variant"] = 0
        chrom_interval_idx[chrom] = 0

    current_chrom = None
    intervals = None
    iv_idx = 0

    with open_maybe_gzipped(gvcf_path) as fh:
        sample_col = None
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.strip().split("\t")
                sample_names = header[9:]
                if sample_id and sample_id in sample_names:
                    sample_col = 9 + sample_names.index(sample_id)
                else:
                    sample_col = 9
                continue

            fields = line.rstrip("\n\r").split("\t")
            if len(fields) < sample_col + 1:
                continue

            chrom = fields[0]
            try:
                pos = int(fields[1])
            except ValueError:
                continue

            ref = fields[3]
            alt = fields[4]

            # Determine the span of this gVCF record
            # For reference blocks, INFO contains END=<pos>
            # For variant records, the span is just the REF length
            end_pos = pos  # default: single base
            info = fields[7]

            is_ref_block = (alt == "<NON_REF>" or alt == "." or
                            (alt.startswith("<") and "NON_REF" in alt))

            if is_ref_block:
                # Look for END in INFO
                for entry in info.split(";"):
                    if entry.startswith("END="):
                        try:
                            end_pos = int(entry[4:])
                        except ValueError:
                            pass
                        break
            else:
                end_pos = pos + len(ref) - 1

            # Parse FORMAT fields
            fmt_keys = fields[8].split(":")
            sample_vals = fields[sample_col].split(":")
            fmt_dict = {}
            for i, key in enumerate(fmt_keys):
                if i < len(sample_vals):
                    fmt_dict[key] = sample_vals[i]

            gt = fmt_dict.get("GT", "./.")

            # Determine depth: use MIN_DP for ref blocks, DP otherwise
            dp = None
            if "MIN_DP" in fmt_dict:
                try:
                    dp = int(fmt_dict["MIN_DP"])
                except ValueError:
                    pass
            if dp is None and "DP" in fmt_dict:
                try:
                    dp = int(fmt_dict["DP"])
                except ValueError:
                    pass

            # GQ
            gq = None
            if "GQ" in fmt_dict:
                try:
                    gq = int(fmt_dict["GQ"])
                except ValueError:
                    pass

            # Classify this record
            if is_ref_block:
                is_callable = (dp is not None and dp >= min_dp and
                               gq is not None and gq >= min_gq and
                               gt in ("0/0", "0|0"))
                record_type = "ref"
            else:
                # Variant record in gVCF — this is a real variant
                record_type = "variant"
                is_callable = False  # not hom-ref

            # Switch chromosome if needed
            if chrom != current_chrom:
                current_chrom = chrom
                intervals = chrom_intervals.get(chrom, [])
                iv_idx = 0

            # Assign bases to intervals
            # The gVCF record covers [pos, end_pos] inclusive.
            # We need to find which intervals overlap this range.
            while iv_idx < len(intervals):
                iv = intervals[iv_idx]

                # Interval is [iv.start, iv.end) — bases start..end-1
                if end_pos < iv["start"]:
                    # gVCF record is entirely before this interval
                    break

                if pos >= iv["end"]:
                    # gVCF record is entirely after this interval,
                    # move to next interval
                    iv_idx += 1
                    continue

                # There is overlap.  Compute the overlapping range.
                overlap_start = max(pos, iv["start"])
                overlap_end = min(end_pos, iv["end"] - 1)  # inclusive
                n_bases = overlap_end - overlap_start + 1

                if n_bases > 0:
                    if record_type == "variant":
                        iv["n_variant"] += n_bases
                    elif is_callable:
                        iv["n_callable"] += n_bases
                    else:
                        iv["n_uncallable"] += n_bases

                # If the gVCF record extends beyond this interval,
                # move to the next interval but don't advance the gVCF
                if end_pos >= iv["end"]:
                    iv_idx += 1
                    continue
                else:
                    break

    # Any interval bases not covered by any gVCF record are uncallable.
    # Compute this as: length - (n_callable + n_uncallable + n_variant)
    for chrom, intervals in chrom_intervals.items():
        for iv in intervals:
            accounted = iv["n_callable"] + iv["n_uncallable"] + iv["n_variant"]
            gap = iv["length"] - accounted
            if gap > 0:
                iv["n_uncallable"] += gap
            iv["callable_fraction"] = (iv["n_callable"] / iv["length"]
                                       if iv["length"] > 0 else 0.0)


# ---------------------------------------------------------------------------
# Apply the current hand-tuned model for comparison
# ---------------------------------------------------------------------------

def predict_p_homref(iv, genome_variant_rate=0.001, min_dp_callable=5):
    """
    Apply the current hand-tuned model to an interval to get P(hom-ref).
    This allows direct comparison with the observed callable_fraction.
    """
    import math

    fl = iv["dp_left"]
    fr = iv["dp_right"]
    avg_dp = (fl + fr) / 2.0
    dp_score = 1.0 / (1.0 + math.exp(-0.5 * (avg_dp - min_dp_callable)))

    expected_gap = 1.0 / genome_variant_rate if genome_variant_rate > 0 else 1000
    total_span = iv["length"]
    span_ratio = total_span / (3.0 * expected_gap)
    gap_score = 1.0 / (1.0 + span_ratio ** 2)

    density_ratio = (iv["local_density"] / genome_variant_rate
                     if genome_variant_rate > 0 else 1.0)
    density_score = min(1.0, density_ratio)

    raw = (dp_score ** 0.25) * (gap_score ** 0.35) * (density_score ** 0.25)
    return max(0.01, min(0.99, raw))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_intervals(chrom_intervals, out_path, genome_variant_rate=0.001):
    """Write the interval table with features, observed outcomes, and
    current model prediction."""

    columns = [
        "chrom", "start", "end", "length",
        "dp_left", "dp_right", "dp_mean", "local_density",
        "n_callable", "n_uncallable", "n_variant",
        "callable_fraction",
        "predicted_p_homref",
    ]

    n_written = 0
    with gzip.open(out_path + '.gz', "w") as fh:
        fh.write("\t".join(columns) + "\n")

        for chrom in sorted(chrom_intervals.keys(),
                            key=lambda c: (len(c), c)):
            for iv in chrom_intervals[chrom]:
                iv["predicted_p_homref"] = predict_p_homref(
                    iv, genome_variant_rate)

                vals = []
                for c in columns:
                    v = iv.get(c)
                    if v is None:
                        vals.append(".")
                    elif isinstance(v, float):
                        vals.append(f"{v:.6g}")
                    else:
                        vals.append(str(v))
                fh.write("\t".join(vals) + "\n")
                n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(chrom_intervals):
    """Print summary statistics about the intervals and calibration."""
    import math

    total_intervals = 0
    total_bases = 0
    total_callable = 0
    total_uncallable = 0
    total_variant = 0

    # For calibration analysis: bin by predicted P(hom-ref)
    n_bins = 10
    bin_counts = [0] * n_bins      # number of intervals
    bin_bases = [0] * n_bins       # total bases
    bin_callable = [0] * n_bins    # callable bases
    bin_predicted = [0.0] * n_bins # sum of predicted * length

    for chrom, intervals in chrom_intervals.items():
        for iv in intervals:
            total_intervals += 1
            total_bases += iv["length"]
            total_callable += iv["n_callable"]
            total_uncallable += iv["n_uncallable"]
            total_variant += iv["n_variant"]

            p = iv.get("predicted_p_homref", 0.5)
            b = min(int(p * n_bins), n_bins - 1)
            bin_counts[b] += 1
            bin_bases[b] += iv["length"]
            bin_callable[b] += iv["n_callable"]
            bin_predicted[b] += p * iv["length"]

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Interval summary", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Total intervals:  {total_intervals:,}", file=sys.stderr)
    print(f"  Total bases:      {total_bases:,}", file=sys.stderr)
    print(f"  Callable bases:   {total_callable:,} "
          f"({100*total_callable/total_bases:.1f}%)", file=sys.stderr)
    print(f"  Uncallable bases: {total_uncallable:,} "
          f"({100*total_uncallable/total_bases:.1f}%)", file=sys.stderr)
    print(f"  Variant bases:    {total_variant:,} "
          f"({100*total_variant/total_bases:.1f}%)", file=sys.stderr)

    # Calibration table
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Calibration: predicted P(hom-ref) vs observed", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  {'Bin':>12} {'Intervals':>10} {'Bases':>12} "
          f"{'Pred':>8} {'Obs':>8} {'Diff':>8}", file=sys.stderr)
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*8} {'-'*8} {'-'*8}",
          file=sys.stderr)

    total_log_loss = 0.0
    total_weight = 0

    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        label = f"[{lo:.1f},{hi:.1f})"

        if bin_bases[b] > 0:
            obs = bin_callable[b] / bin_bases[b]
            pred = bin_predicted[b] / bin_bases[b]
            diff = pred - obs

            # Log loss contribution (base-weighted)
            eps = 1e-10
            ll = -(bin_callable[b] * math.log(max(pred, eps)) +
                   (bin_bases[b] - bin_callable[b]) *
                   math.log(max(1 - pred, eps)))
            total_log_loss += ll
            total_weight += bin_bases[b]
        else:
            obs = pred = diff = 0.0

        print(f"  {label:>12} {bin_counts[b]:>10,} {bin_bases[b]:>12,} "
              f"{pred:>8.3f} {obs:>8.3f} {diff:>+8.3f}", file=sys.stderr)

    if total_weight > 0:
        avg_ll = total_log_loss / total_weight
        print(f"\n  Base-weighted log-loss: {avg_ll:.4f}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract inter-variant intervals and callable fractions "
                    "for P(hom-ref) model calibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--vcf", required=True,
                        help="Standard VCF file (plain or .gz)")
    parser.add_argument("--gvcf", required=True,
                        help="gVCF file with reference blocks (plain or .gz)")
    parser.add_argument("--out", required=True,
                        help="Output TSV with interval features and outcomes")
    parser.add_argument("--sample", default=None,
                        help="Sample ID for multi-sample files")
    parser.add_argument("--min-dp", type=int, default=5,
                        help="Min depth for callable in gVCF (default: 5)")
    parser.add_argument("--min-gq", type=int, default=20,
                        help="Min genotype quality for callable (default: 20)")
    parser.add_argument("--density-window", type=int, default=50000,
                        help="Window for local density (default: 50000)")
    parser.add_argument("--genome-variant-rate", type=float, default=0.001,
                        help="Expected variants/bp for model (default: 0.001)")

    args = parser.parse_args()

    # Step 1: Parse VCF
    print("Parsing VCF...", file=sys.stderr)
    chrom_variants = parse_vcf_positions(args.vcf, args.sample)
    total_vars = sum(len(v) for v in chrom_variants.values())
    print(f"  {total_vars:,} variants across "
          f"{len(chrom_variants)} chromosomes", file=sys.stderr)

    # Step 2: Build intervals
    print("Building inter-variant intervals...", file=sys.stderr)
    chrom_intervals = build_intervals(chrom_variants)
    total_intervals = sum(len(v) for v in chrom_intervals.values())
    total_bases = sum(iv["length"] for ivs in chrom_intervals.values()
                      for iv in ivs)
    print(f"  {total_intervals:,} intervals covering "
          f"{total_bases:,} bases", file=sys.stderr)

    # Step 3: Add density features
    print("Computing density features...", file=sys.stderr)
    add_density_features(chrom_variants, chrom_intervals, args.density_window)

    # Step 4: Count callable bases from gVCF
    print("Streaming gVCF to count callable bases...", file=sys.stderr)
    count_callable_in_intervals(args.gvcf, chrom_intervals,
                                args.sample, args.min_dp, args.min_gq)

    # Step 5: Write output
    print("Writing output...", file=sys.stderr)
    n = write_intervals(chrom_intervals, args.out, args.genome_variant_rate)
    print(f"  {n:,} intervals written to {args.out}", file=sys.stderr)

    # Step 6: Print summary and calibration
    print_summary(chrom_intervals)

    print(f"\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()

