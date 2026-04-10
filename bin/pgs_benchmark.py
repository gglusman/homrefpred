#!/usr/bin/env python3
"""
pgs_benchmark.py

Benchmark PGS scoring strategies against gVCF ground truth.

For each PGS variant, computes the "true" dosage from the gVCF (which
knows whether every position is callable or not), then compares four
scoring strategies:
  1. Assume all absent = hom-ref
  2. Mean-impute all absent (using population frequency)
  3. Coverage-weighted with hand-tuned model
  4. Coverage-weighted with calibrated model (from pgs_calibrate_model.py)

Reports per-variant and aggregate error metrics.

Usage
-----
    python pgs_benchmark.py \\
        --vcf sample.vcf.gz \\
        --gvcf sample.g.vcf.gz \\
        --pgs PGS000001.annotated.txt \\
        --model model.json \\
        --out benchmark_results.tsv \\
        [--population EUR] \\
        [--min-dp 5] \\
        [--min-gq 20]
"""

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from bisect import bisect_left, bisect_right


def open_maybe_gzipped(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


# ---------------------------------------------------------------------------
# Parse the calibrated model
# ---------------------------------------------------------------------------

def load_model(path):
    with open(path) as f:
        return json.load(f)


def predict_calibrated(dp_mean, length, local_density, model):
    """Apply calibrated logistic regression model."""
    weights = model["weights"]
    f0 = math.log(dp_mean + 1)
    f1 = math.log(length + 1)
    f2 = math.log(local_density + 1e-6)
    f3 = f0 * f1
    z = weights[0]*f0 + weights[1]*f1 + weights[2]*f2 + weights[3]*f3 + weights[4]
    if z > 500:
        return 1.0
    if z < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def predict_handtuned(dp_mean, length, local_density,
                      genome_variant_rate=0.001, min_dp_callable=5):
    """Apply the hand-tuned model."""
    dp_score = 1.0 / (1.0 + math.exp(-0.5 * (dp_mean - min_dp_callable)))

    expected_gap = 1.0 / genome_variant_rate if genome_variant_rate > 0 else 1000
    span_ratio = length / (3.0 * expected_gap)
    gap_score = 1.0 / (1.0 + span_ratio ** 2)

    density_ratio = (local_density / genome_variant_rate
                     if genome_variant_rate > 0 else 1.0)
    density_score = min(1.0, density_ratio)

    raw = (dp_score ** 0.25) * (gap_score ** 0.35) * (density_score ** 0.25)
    return max(0.01, min(0.99, raw))


# ---------------------------------------------------------------------------
# VCF parsing (same as pgs_score.py)
# ---------------------------------------------------------------------------

def parse_vcf(vcf_path, sample_id=None):
    chrom_variants = defaultdict(list)
    vcf_genotypes = {}

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
            ref = fields[3]
            alts = fields[4].split(",")

            fmt_keys = fields[8].split(":")
            sample_vals = fields[sample_col].split(":")
            fmt_dict = {fmt_keys[i]: sample_vals[i]
                        for i in range(min(len(fmt_keys), len(sample_vals)))}

            dp = None
            if "DP" in fmt_dict:
                try:
                    dp = int(fmt_dict["DP"])
                except ValueError:
                    pass
            if dp is None:
                for entry in fields[7].split(";"):
                    if entry.startswith("DP="):
                        try:
                            dp = int(entry[3:])
                        except ValueError:
                            pass
                        break

            gt = fmt_dict.get("GT", "./.")
            if dp is not None and dp >= 0:
                chrom_variants[chrom].append((pos, dp))
            vcf_genotypes[(chrom, pos)] = {
                "ref": ref, "alts": alts, "gt": gt, "dp": dp,
            }

    for chrom in chrom_variants:
        chrom_variants[chrom].sort()
    return chrom_variants, vcf_genotypes


# ---------------------------------------------------------------------------
# gVCF: look up ground truth for specific positions
# ---------------------------------------------------------------------------

def load_gvcf_index(gvcf_path, positions_needed, sample_id=None,
                    min_dp=5, min_gq=20):
    """
    For a set of (chrom, pos) positions, determine from the gVCF whether
    each is callable (hom-ref with sufficient quality), variant, or
    uncallable.

    Returns dict[(chrom, pos)] -> {"status": "callable"/"uncallable"/"variant",
                                    "dp": ..., "gq": ..., "gt": ...}
    """
    # Organize needed positions by chrom for efficient lookup
    needed_by_chrom = defaultdict(set)
    for chrom, pos in positions_needed:
        needed_by_chrom[chrom].add(pos)

    results = {}
    current_chrom = None
    current_needed = None

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

            if chrom != current_chrom:
                current_chrom = chrom
                current_needed = needed_by_chrom.get(chrom, set())
                if not current_needed:
                    continue

            ref = fields[3]
            alt = fields[4]
            info = fields[7]

            # Determine record span
            is_ref_block = (alt == "<NON_REF>" or alt == "." or
                            (alt.startswith("<") and "NON_REF" in alt))
            end_pos = pos
            if is_ref_block:
                for entry in info.split(";"):
                    if entry.startswith("END="):
                        try:
                            end_pos = int(entry[4:])
                        except ValueError:
                            pass
                        break
            else:
                end_pos = pos + len(ref) - 1

            # Check if any needed positions fall in this record
            # (For efficiency, only process if there could be overlap)
            for qpos in list(current_needed):
                if pos <= qpos <= end_pos:
                    # Parse FORMAT
                    fmt_keys = fields[8].split(":")
                    sample_vals = fields[sample_col].split(":")
                    fmt_dict = {fmt_keys[i]: sample_vals[i]
                                for i in range(min(len(fmt_keys),
                                                   len(sample_vals)))}

                    gt = fmt_dict.get("GT", "./.")
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
                    gq = None
                    if "GQ" in fmt_dict:
                        try:
                            gq = int(fmt_dict["GQ"])
                        except ValueError:
                            pass

                    if is_ref_block:
                        if (dp is not None and dp >= min_dp and
                                gq is not None and gq >= min_gq and
                                gt in ("0/0", "0|0")):
                            status = "callable"
                        else:
                            status = "uncallable"
                    else:
                        status = "variant"

                    results[(chrom, qpos)] = {
                        "status": status,
                        "dp": dp,
                        "gq": gq,
                        "gt": gt,
                    }
                    current_needed.discard(qpos)

    # Positions not covered by any gVCF record are uncallable
    for chrom, positions in needed_by_chrom.items():
        for qpos in positions:
            if (chrom, qpos) not in results:
                results[(chrom, qpos)] = {
                    "status": "uncallable",
                    "dp": None, "gq": None, "gt": None,
                }

    return results


# ---------------------------------------------------------------------------
# Annotated PGS parsing (simplified from pgs_score.py)
# ---------------------------------------------------------------------------

def parse_annotated_pgs(pgs_path, population="FREQ"):
    target_col = f"tgp_{population}".lower()
    variants = []

    with open_maybe_gzipped(pgs_path) as fh:
        header_line = None
        for line in fh:
            if line.startswith("#"):
                continue
            header_line = line.rstrip("\n\r")
            break

        header = header_line.split("\t")
        col_map = {name.strip().lower(): i for i, name in enumerate(header)}

        def get_col(names, req=True):
            for n in names:
                if n in col_map:
                    return col_map[n]
            if req:
                print(f"Error: column {names} not found.", file=sys.stderr)
                sys.exit(1)
            return None

        ci = get_col(["chr_name", "chrom", "chr"])
        pi = get_col(["chr_position", "pos", "bp"])
        ei = get_col(["effect_allele", "a1"])
        ri = get_col(["other_allele", "ref", "a2"], req=False)
        bi = get_col(["effect_weight", "beta"])
        fi = col_map.get(target_col)
        if fi is None:
            fi = get_col(["effect_allele_freq", "eaf"], req=False)

        for line in fh:
            if line.startswith("#") or line.rstrip() == "":
                continue
            fields = line.rstrip("\n\r").split("\t")
            try:
                chrom = fields[ci].strip()
                pos = int(fields[pi].strip())
                ea = fields[ei].strip()
                beta = float(fields[bi].strip())
            except (ValueError, IndexError):
                continue

            ref = fields[ri].strip() if ri is not None and ri < len(fields) \
                else None

            eaf = None
            if fi is not None and fi < len(fields):
                val = fields[fi].strip()
                if val and val != ".":
                    try:
                        eaf = float(val)
                    except ValueError:
                        pass

            if not chrom.startswith("chr"):
                chrom_lookup = [chrom, "chr" + chrom]
            else:
                chrom_lookup = [chrom, chrom[3:]]

            variants.append({
                "chrom": chrom, "chrom_lookup": chrom_lookup,
                "pos": pos, "ref": ref, "effect_allele": ea,
                "beta": beta, "eaf": eaf,
            })

    return variants


# ---------------------------------------------------------------------------
# Helper: flanking DP and density for a position
# ---------------------------------------------------------------------------

def get_flanking_features(chrom_variants, chrom, pos, flank_k=10, window=50000):
    """Get flanking DP, gap, and density for a single position."""
    cv = chrom_variants.get(chrom, [])
    if not cv:
        return 0.0, 0.0, float("inf"), float("inf"), 0.0

    positions = [p for p, _ in cv]
    idx = bisect_left(positions, pos)

    left_start = max(0, idx - flank_k)
    left_dps = [cv[i][1] for i in range(left_start, idx)]
    right_end = min(len(cv), idx + flank_k)
    right_dps = [cv[i][1] for i in range(idx, right_end)]

    def median(vals):
        if not vals:
            return 0.0
        s = sorted(vals)
        n = len(s)
        return float(s[n//2]) if n % 2 == 1 else (s[n//2-1] + s[n//2]) / 2.0

    fl = median(left_dps)
    fr = median(right_dps)
    gl = pos - positions[idx - 1] if idx > 0 else float("inf")
    gr = positions[idx] - pos if idx < len(positions) else float("inf")

    lo = bisect_left(positions, pos - window)
    hi = bisect_right(positions, pos + window)
    density = (hi - lo) / (2.0 * window) if window > 0 else 0.0

    return fl, fr, gl, gr, density


# ---------------------------------------------------------------------------
# Allele helpers
# ---------------------------------------------------------------------------

def effect_allele_is_ref(pgs_ref, ea, vcf_ref=None):
    if ea == pgs_ref:
        return True
    if vcf_ref is not None and ea == vcf_ref:
        return True
    return False


def homref_dosage(pgs_ref, ea, vcf_ref=None):
    return 2.0 if effect_allele_is_ref(pgs_ref, ea, vcf_ref) else 0.0


def compute_dosage_for_gt(gt_str, ref, ea, alts):
    if gt_str in ("./.", ".|.", "."):
        return None
    sep = "|" if "|" in gt_str else "/"
    allele_list = [ref] + alts
    dosage = 0
    for ai in gt_str.split(sep):
        if ai == ".":
            return None
        idx = int(ai)
        if idx < len(allele_list) and allele_list[idx] == ea:
            dosage += 1
    return dosage


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PGS scoring strategies against gVCF truth.",
    )
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--gvcf", required=True)
    parser.add_argument("--pgs", required=True,
                        help="Annotated PGS file(s), comma-separated")
    parser.add_argument("--model", required=True,
                        help="Calibrated model JSON from pgs_calibrate_model.py")
    parser.add_argument("--out", required=True,
                        help="Output benchmark results TSV")
    parser.add_argument("--population", default="FREQ")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--min-dp", type=int, default=5)
    parser.add_argument("--min-gq", type=int, default=20)

    args = parser.parse_args()

    # Load model
    print("Loading calibrated model...", file=sys.stderr)
    model = load_model(args.model)

    # Parse VCF
    print("Parsing VCF...", file=sys.stderr)
    chrom_variants, vcf_genotypes = parse_vcf(args.vcf, args.sample)

    # Process each PGS file
    pgs_files = [p.strip() for p in args.pgs.split(",")]

    all_results = []

    for pgs_path in pgs_files:
        pgs_name = pgs_path.split("/")[-1].replace(".annotated.txt", "")
        print(f"\nProcessing PGS: {pgs_name}...", file=sys.stderr)

        pgs_variants = parse_annotated_pgs(pgs_path, args.population)
        print(f"  {len(pgs_variants)} variants", file=sys.stderr)

        # Determine which positions we need gVCF truth for
        # (only those absent from the standard VCF)
        positions_needed = []
        for var in pgs_variants:
            found = False
            for cl in var["chrom_lookup"]:
                if (cl, var["pos"]) in vcf_genotypes:
                    found = True
                    break
            if not found:
                for cl in var["chrom_lookup"]:
                    positions_needed.append((cl, var["pos"]))

        print(f"  {len(positions_needed)} absent from VCF, "
              f"querying gVCF...", file=sys.stderr)

        gvcf_truth = load_gvcf_index(
            args.gvcf, positions_needed, args.sample,
            args.min_dp, args.min_gq)

        # Score each variant under all strategies
        scores = {"homref": 0.0, "meanimpute": 0.0,
                  "handtuned": 0.0, "calibrated": 0.0, "truth": 0.0}
        n_found = 0
        n_absent = 0
        n_absent_callable = 0
        n_absent_uncallable = 0
        n_absent_variant = 0
        n_missing_eaf = 0
        per_variant = []

        for var in pgs_variants:
            beta = var["beta"]
            eaf = var["eaf"]
            ea = var["effect_allele"]

            # Check if in VCF
            found = False
            matched_chrom = None
            for cl in var["chrom_lookup"]:
                if (cl, var["pos"]) in vcf_genotypes:
                    matched_chrom = cl
                    found = True
                    break

            if found:
                n_found += 1
                vcf_entry = vcf_genotypes[(matched_chrom, var["pos"])]
                dosage = compute_dosage_for_gt(
                    vcf_entry["gt"], vcf_entry["ref"], ea, vcf_entry["alts"])
                if dosage is None:
                    dosage = 0.0  # missing GT, treat as 0 for simplicity
                d = float(dosage)
                for strategy in scores:
                    scores[strategy] += beta * d
                per_variant.append({
                    "pgs": pgs_name, "chrom": var["chrom"],
                    "pos": var["pos"], "beta": beta, "eaf": eaf,
                    "ea": ea, "status": "found", "gvcf_status": "in_vcf",
                    "dosage_truth": d, "dosage_homref": d,
                    "dosage_meanimpute": d, "dosage_handtuned": d,
                    "dosage_calibrated": d, "p_handtuned": 1.0,
                    "p_calibrated": 1.0,
                })
                continue

            # Absent from VCF
            n_absent += 1

            # gVCF ground truth
            truth_entry = None
            for cl in var["chrom_lookup"]:
                if (cl, var["pos"]) in gvcf_truth:
                    truth_entry = gvcf_truth[(cl, var["pos"])]
                    matched_chrom = cl
                    break
            if truth_entry is None:
                truth_entry = {"status": "uncallable"}

            gvcf_status = truth_entry["status"]
            if gvcf_status == "callable":
                n_absent_callable += 1
            elif gvcf_status == "variant":
                n_absent_variant += 1
            else:
                n_absent_uncallable += 1

            # True dosage from gVCF
            d_hr = homref_dosage(var["ref"], ea)
            if gvcf_status == "callable":
                # Site was callable and hom-ref → dosage is hom-ref dosage
                dosage_truth = d_hr
            elif gvcf_status == "variant":
                # gVCF has a variant here not in the standard VCF (rare)
                dosage_truth = d_hr  # approximate
            else:
                # Truly uncallable → best we can do is mean-impute
                if eaf is not None:
                    dosage_truth = 2.0 * eaf
                else:
                    dosage_truth = 1.0
                    n_missing_eaf += 1

            # Strategy 1: assume hom-ref
            dosage_homref = d_hr

            # Strategy 2: mean-impute
            if eaf is not None:
                dosage_meanimpute = 2.0 * eaf
            else:
                dosage_meanimpute = 1.0

            # Get coverage features
            fl, fr, gl, gr, density = get_flanking_features(
                chrom_variants, matched_chrom or var["chrom_lookup"][0],
                var["pos"])
            dp_mean = (fl + fr) / 2.0
            length = gl + gr if gl != float("inf") and gr != float("inf") \
                else max(gl, gr) if max(gl, gr) != float("inf") else 100000

            # Strategy 3: hand-tuned
            p_ht = predict_handtuned(dp_mean, length, density)
            dosage_handtuned = p_ht * d_hr + (1 - p_ht) * dosage_meanimpute

            # Strategy 4: calibrated
            p_cal = predict_calibrated(dp_mean, length, density, model)
            dosage_calibrated = p_cal * d_hr + (1 - p_cal) * dosage_meanimpute

            scores["truth"] += beta * dosage_truth
            scores["homref"] += beta * dosage_homref
            scores["meanimpute"] += beta * dosage_meanimpute
            scores["handtuned"] += beta * dosage_handtuned
            scores["calibrated"] += beta * dosage_calibrated

            per_variant.append({
                "pgs": pgs_name, "chrom": var["chrom"],
                "pos": var["pos"], "beta": beta, "eaf": eaf,
                "ea": ea, "status": "absent", "gvcf_status": gvcf_status,
                "dosage_truth": dosage_truth,
                "dosage_homref": dosage_homref,
                "dosage_meanimpute": dosage_meanimpute,
                "dosage_handtuned": dosage_handtuned,
                "dosage_calibrated": dosage_calibrated,
                "p_handtuned": p_ht, "p_calibrated": p_cal,
            })

        # Report
        print(f"\n  {'='*55}", file=sys.stderr)
        print(f"  Results for {pgs_name}", file=sys.stderr)
        print(f"  {'='*55}", file=sys.stderr)
        print(f"  Found in VCF: {n_found}", file=sys.stderr)
        print(f"  Absent from VCF: {n_absent}", file=sys.stderr)
        print(f"    Callable (hom-ref) in gVCF: {n_absent_callable}",
              file=sys.stderr)
        print(f"    Uncallable in gVCF:         {n_absent_uncallable}",
              file=sys.stderr)
        print(f"    Variant in gVCF (rare):     {n_absent_variant}",
              file=sys.stderr)
        if n_missing_eaf:
            print(f"    Missing EAF (used 0.5):     {n_missing_eaf}",
                  file=sys.stderr)

        print(f"\n  Scores:", file=sys.stderr)
        print(f"    {'Strategy':<25} {'Score':>12} {'Error vs truth':>15}",
              file=sys.stderr)
        print(f"    {'-'*25} {'-'*12} {'-'*15}", file=sys.stderr)
        truth_score = scores["truth"]
        for strategy in ["homref", "meanimpute", "handtuned", "calibrated",
                         "truth"]:
            s = scores[strategy]
            err = s - truth_score
            label = {"homref": "Assume hom-ref",
                     "meanimpute": "Mean-impute all",
                     "handtuned": "Hand-tuned weighted",
                     "calibrated": "Calibrated weighted",
                     "truth": "gVCF truth"}[strategy]
            err_str = f"{err:+.6f}" if strategy != "truth" else "---"
            print(f"    {label:<25} {s:>12.6f} {err_str:>15}",
                  file=sys.stderr)

        # Per-variant squared error for absent variants
        absent = [r for r in per_variant if r["status"] == "absent"]
        if absent:
            for strategy in ["homref", "meanimpute", "handtuned", "calibrated"]:
                mse = sum((r[f"dosage_{strategy}"] - r["dosage_truth"])**2
                          * abs(r["beta"])
                          for r in absent) / sum(abs(r["beta"])
                                                 for r in absent)
                key = f"dosage_{strategy}"
                label = {"dosage_homref": "Hom-ref",
                         "dosage_meanimpute": "Mean-impute",
                         "dosage_handtuned": "Hand-tuned",
                         "dosage_calibrated": "Calibrated"}[key]

            print(f"\n  Beta-weighted MSE of dosage (absent variants):",
                  file=sys.stderr)
            print(f"    {'Strategy':<25} {'MSE':>12}", file=sys.stderr)
            print(f"    {'-'*25} {'-'*12}", file=sys.stderr)
            for strategy in ["homref", "meanimpute", "handtuned", "calibrated"]:
                mse = sum((r[f"dosage_{strategy}"] - r["dosage_truth"])**2
                          * abs(r["beta"])
                          for r in absent) / sum(abs(r["beta"])
                                                 for r in absent)
                label = {"homref": "Hom-ref", "meanimpute": "Mean-impute",
                         "handtuned": "Hand-tuned",
                         "calibrated": "Calibrated"}[strategy]
                print(f"    {label:<25} {mse:>12.6f}", file=sys.stderr)

        all_results.extend(per_variant)

    # Write per-variant output
    columns = ["pgs", "chrom", "pos", "beta", "eaf", "ea",
               "status", "gvcf_status",
               "dosage_truth", "dosage_homref", "dosage_meanimpute",
               "dosage_handtuned", "dosage_calibrated",
               "p_handtuned", "p_calibrated"]

    with open(args.out, "w") as f:
        f.write("\t".join(columns) + "\n")
        for r in all_results:
            vals = []
            for c in columns:
                v = r.get(c)
                if v is None:
                    vals.append(".")
                elif isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")

    print(f"\nPer-variant results written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

