#!/usr/bin/env python3
"""
pgs_score.py

Compute a polygenic risk score from a VCF, using an annotated PGS file
(produced by pgs_annotate_freq.py) that includes per-population allele
frequencies from 1000 Genomes.

For each PGS variant absent from the VCF, estimates the probability that
the site was truly homozygous-reference (well-covered) vs. a no-call
(poorly covered / unmappable), using DP and variant density from
neighboring VCF entries.  This probability weights a blend between
dosage=hom-ref and dosage=mean-imputed.

Usage
-----
    python pgs_score.py \\
        --vcf sample.vcf.gz \\
        --pgs PGS000001.annotated.txt \\
        --out results.tsv \\
        [--population EUR] \\
        [--sample SAMPLE_ID] \\
        [--mappability callable_regions.bed] \\
        [--flank-k 10] \\
        [--window 50000] \\
        [--genome-variant-rate 0.001] \\
        [--min-dp-callable 5]

The --population parameter selects which tgp_* column from the annotated
PGS file to use for allele frequencies.  Default: FREQ (global).
Examples: FREQ, AFR, AMR, EUR, SAS, EAS, CLM, GBR, JPT, YRI, ...

Dependencies: Python 3 standard library only.
"""

import argparse
import gzip
import math
import sys
from collections import defaultdict
from bisect import bisect_left, bisect_right


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def open_maybe_gzipped(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


# ---------------------------------------------------------------------------
# VCF parsing
# ---------------------------------------------------------------------------

def parse_vcf(vcf_path, sample_id=None):
    """
    Parse a VCF and extract per-chromosome variant positions with DP,
    plus genotype information for PGS matching.
    """
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
                    if sample_id:
                        print(f"Warning: sample '{sample_id}' not found, "
                              f"using first.", file=sys.stderr)
                    sample_col = 9
                continue

            fields = line.strip().split("\t")
            if len(fields) < sample_col + 1:
                continue

            chrom = fields[0]
            pos = int(fields[1])
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
        chrom_variants[chrom].sort(key=lambda x: x[0])

    return chrom_variants, vcf_genotypes


# ---------------------------------------------------------------------------
# Annotated PGS parsing
# ---------------------------------------------------------------------------

def parse_annotated_pgs(pgs_path, population="FREQ"):
    """
    Parse an annotated PGS file (output of pgs_annotate_freq.py).

    Selects the tgp_{population} column for EAF.  Falls back to the
    PGS file's own EAF column if no tgp_ column matches.

    Returns list of variant dicts with 'eaf' set from the chosen population.
    """
    variants = []
    skipped = 0

    # The annotated column we want
    target_col = f"tgp_{population}".lower()

    with open_maybe_gzipped(pgs_path) as fh:
        header_line = None
        for line in fh:
            if line.startswith("#"):
                continue
            header_line = line.rstrip("\n\r")
            break

        if header_line is None:
            print("Error: PGS file has no data.", file=sys.stderr)
            sys.exit(1)

        header = header_line.split("\t")
        col_map = {name.strip().lower(): i for i, name in enumerate(header)}

        def get_col(names, required=True):
            for n in names:
                if n in col_map:
                    return col_map[n]
            if required:
                print(f"Error: column not found for {names} in header.",
                      file=sys.stderr)
                sys.exit(1)
            return None

        ci = get_col(["chr_name", "chrom", "chr", "chromosome"])
        pi = get_col(["chr_position", "pos", "position", "bp"])
        ei = get_col(["effect_allele", "alt", "allele", "a1"])
        ri = get_col(["other_allele", "ref", "reference_allele", "a2"], required=False)
        bi = get_col(["effect_weight", "beta", "weight"])

        # Find the population frequency column
        freq_col = None
        if target_col in col_map:
            freq_col = col_map[target_col]
        else:
            # List available tgp_ columns for error message
            tgp_cols = sorted([k for k in col_map if k.startswith("tgp_")])
            if tgp_cols:
                avail = [c.replace("tgp_", "") for c in tgp_cols]
                print(f"Error: population '{population}' not found.",
                      file=sys.stderr)
                print(f"Available: {', '.join(avail)}", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"Warning: no tgp_* frequency columns found. "
                      f"Was this PGS file annotated with pgs_annotate_freq.py?",
                      file=sys.stderr)
                # Fall back to any EAF column in the PGS itself
                freq_col = get_col(["effect_allele_freq", "eaf",
                                    "allelefrequency"], required=False)

        max_idx = max(x for x in [ci, pi, ei, bi] if x is not None)

        for line in fh:
            if line.startswith("#") or line.rstrip() == "":
                continue

            fields = line.rstrip("\n\r").split("\t")
            if len(fields) <= max_idx:
                skipped += 1
                continue

            chrom = fields[ci].strip()
            pos_str = fields[pi].strip()
            if not chrom or not pos_str:
                skipped += 1
                continue
            try:
                pos = int(pos_str)
            except ValueError:
                skipped += 1
                continue

            if not chrom.startswith("chr"):
                chrom_lookup = [chrom, "chr" + chrom]
            else:
                chrom_lookup = [chrom, chrom[3:]]

            effect_allele = fields[ei].strip()
            try:
                beta = float(fields[bi].strip())
            except ValueError:
                skipped += 1
                continue

            ref = fields[ri].strip() if ri is not None and ri < len(fields) \
                else None

            eaf = None
            eaf_source = None
            if freq_col is not None and freq_col < len(fields):
                val = fields[freq_col].strip()
                if val and val != ".":
                    try:
                        eaf = float(val)
                        eaf_source = f"tgp_{population}"
                    except ValueError:
                        pass

            variants.append({
                "chrom": chrom,
                "chrom_lookup": chrom_lookup,
                "pos": pos,
                "ref": ref,
                "effect_allele": effect_allele,
                "beta": beta,
                "eaf": eaf,
                "eaf_source": eaf_source,
            })

    if skipped:
        print(f"PGS: skipped {skipped} unparseable variants.",
              file=sys.stderr)

    n_with_eaf = sum(1 for v in variants if v["eaf"] is not None)
    print(f"PGS: {len(variants)} variants, {n_with_eaf} with "
          f"{population} frequency.", file=sys.stderr)

    return variants


# ---------------------------------------------------------------------------
# Coverage confidence model
# ---------------------------------------------------------------------------

def get_flanking_dp(positions_dp, query_pos, k=10):
    positions = [p for p, _ in positions_dp]
    idx = bisect_left(positions, query_pos)

    left_start = max(0, idx - k)
    left_dps = [positions_dp[i][1] for i in range(left_start, idx)]
    gap_left = query_pos - positions[idx - 1] if idx > 0 else float("inf")

    right_end = min(len(positions_dp), idx + k)
    right_dps = [positions_dp[i][1] for i in range(idx, right_end)]
    gap_right = positions[idx] - query_pos if idx < len(positions) \
        else float("inf")

    def median(vals):
        if not vals:
            return 0.0
        s = sorted(vals)
        n = len(s)
        return float(s[n // 2]) if n % 2 == 1 else (s[n//2-1] + s[n//2]) / 2.0

    return median(left_dps), median(right_dps), gap_left, gap_right


def local_variant_density(positions_dp, query_pos, window=50000):
    positions = [p for p, _ in positions_dp]
    lo = bisect_left(positions, query_pos - window)
    hi = bisect_right(positions, query_pos + window)
    count = hi - lo
    density = count / (2 * window) if window > 0 else 0.0
    return count, density


def estimate_p_homref_handtuned(fl, fr, gl, gr, local_density,
                                genome_variant_rate, mappable=None,
                                min_dp_callable=5):
    """Original hand-tuned model (default fallback)."""
    avg_dp = (fl + fr) / 2.0
    dp_score = 1.0 / (1.0 + math.exp(-0.5 * (avg_dp - min_dp_callable)))

    expected_gap = 1.0 / genome_variant_rate if genome_variant_rate > 0 else 1000
    total_span = gl + gr if gl != float("inf") and gr != float("inf") \
        else max(gl, gr)
    span_ratio = total_span / (3.0 * expected_gap)
    gap_score = 1.0 / (1.0 + span_ratio ** 2)

    density_ratio = local_density / genome_variant_rate \
        if genome_variant_rate > 0 else 1.0
    density_score = min(1.0, density_ratio)

    if mappable is True:
        map_score = 1.0
    elif mappable is False:
        map_score = 0.1
    else:
        map_score = 1.0

    raw = (dp_score ** 0.25) * (gap_score ** 0.35) * \
          (density_score ** 0.25) * (map_score ** 0.15)

    return max(0.01, min(0.99, raw))


def estimate_p_homref_calibrated(fl, fr, gl, gr, local_density, model):
    """
    Apply a calibrated logistic regression model loaded from a JSON file
    produced by pgs_calibrate_model.py.

    Features (must match those used during training):
      - log(dp_mean + 1)
      - log(length + 1)
      - log(local_density + 1e-6)
      - dp_x_length interaction
      - intercept
    """
    weights = model["weights"]
    dp_mean = (fl + fr) / 2.0
    length = gl + gr if gl != float("inf") and gr != float("inf") \
        else max(gl, gr) if max(gl, gr) != float("inf") else 100000

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


def estimate_p_homref(fl, fr, gl, gr, local_density, genome_variant_rate,
                      mappable=None, min_dp_callable=5, model=None):
    """
    Dispatch to either the calibrated model (if provided) or the hand-tuned
    fallback.  When a calibrated model is used, mappability is currently
    ignored (the calibration training did not include it).
    """
    if model is not None:
        return estimate_p_homref_calibrated(fl, fr, gl, gr, local_density,
                                            model)
    return estimate_p_homref_handtuned(fl, fr, gl, gr, local_density,
                                       genome_variant_rate, mappable,
                                       min_dp_callable)


# ---------------------------------------------------------------------------
# Mappability BED (optional)
# ---------------------------------------------------------------------------

def parse_mappability_bed(bed_path):
    if bed_path is None:
        return None
    regions = defaultdict(list)
    with open_maybe_gzipped(bed_path) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("track"):
                continue
            fields = line.strip().split("\t")
            regions[fields[0]].append((int(fields[1]), int(fields[2])))
    for chrom in regions:
        regions[chrom].sort()
    return dict(regions)


def in_mappable_region(regions, chrom, pos):
    if regions is None:
        return None
    intervals = regions.get(chrom, [])
    if not intervals:
        return None
    idx = bisect_right(intervals, (pos,)) - 1
    if idx >= 0 and intervals[idx][0] <= pos < intervals[idx][1]:
        return True
    if idx + 1 < len(intervals) and intervals[idx+1][0] <= pos < intervals[idx+1][1]:
        return True
    return False


# ---------------------------------------------------------------------------
# Allele helpers
# ---------------------------------------------------------------------------

def effect_allele_is_ref(pgs_ref, pgs_effect_allele, vcf_ref=None):
    if pgs_effect_allele == pgs_ref:
        return True
    if vcf_ref is not None and pgs_effect_allele == vcf_ref:
        return True
    return False


def homref_dosage(pgs_ref, pgs_effect_allele, vcf_ref=None):
    return 2.0 if effect_allele_is_ref(pgs_ref, pgs_effect_allele, vcf_ref) \
        else 0.0


def compute_dosage_for_gt(gt_str, ref, effect_allele, alts):
    if gt_str in ("./.", ".|.", "."):
        return None
    sep = "|" if "|" in gt_str else "/"
    allele_list = [ref] + alts
    dosage = 0
    for ai in gt_str.split(sep):
        if ai == ".":
            return None
        idx = int(ai)
        if idx < len(allele_list) and allele_list[idx] == effect_allele:
            dosage += 1
    return dosage


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

def score_pgs(vcf_path, pgs_path, population="FREQ", sample_id=None,
              flank_k=10, window=50000, genome_variant_rate=0.001,
              mappability_path=None, min_dp_callable=5, model=None):

    print("Parsing VCF...", file=sys.stderr)
    chrom_variants, vcf_genotypes = parse_vcf(vcf_path, sample_id)

    print(f"Parsing annotated PGS (population: {population})...",
          file=sys.stderr)
    pgs_variants = parse_annotated_pgs(pgs_path, population)

    mappable_regions = parse_mappability_bed(mappability_path)

    if model is not None:
        print(f"Using calibrated model "
              f"(trained on {model.get('training', {}).get('n_intervals', '?')} "
              f"intervals).", file=sys.stderr)
    else:
        print("Using hand-tuned model (no --model provided).",
              file=sys.stderr)

    # Genome-wide median DP
    all_dps = sorted(dp for chrom in chrom_variants
                     for _, dp in chrom_variants[chrom])
    genome_median_dp = all_dps[len(all_dps) // 2] if all_dps else 0
    print(f"VCF: {len(all_dps)} variants, median DP: {genome_median_dp}",
          file=sys.stderr)

    results = []
    score_homref = 0.0
    score_meanimpute = 0.0
    score_weighted = 0.0
    n_found = 0
    n_missing = 0
    n_missing_eaf = 0

    for var in pgs_variants:
        chrom = var["chrom"]
        pos = var["pos"]
        beta = var["beta"]
        eaf = var["eaf"]
        eaf_source = var["eaf_source"]

        # Find variant in VCF
        found = False
        matched_chrom = None
        for cl in var["chrom_lookup"]:
            if (cl, pos) in vcf_genotypes:
                matched_chrom = cl
                found = True
                break

        if found:
            n_found += 1
            vcf_entry = vcf_genotypes[(matched_chrom, pos)]
            dosage = compute_dosage_for_gt(
                vcf_entry["gt"], vcf_entry["ref"],
                var["effect_allele"], vcf_entry["alts"])

            if dosage is None:
                d_hr = homref_dosage(var["ref"], var["effect_allele"],
                                     vcf_entry["ref"])
                dosage_homref = d_hr
                if eaf is not None:
                    dosage_mean = 2.0 * eaf
                else:
                    dosage_mean = 1.0
                    eaf_source = "default_0.5"
                    n_missing_eaf += 1
                dosage_weighted = dosage_mean
                p_homref = 0.0
                status = "missing_gt"
            else:
                dosage_homref = dosage_mean = dosage_weighted = float(dosage)
                p_homref = 1.0
                status = "found"

            ea_is_ref = effect_allele_is_ref(
                var["ref"], var["effect_allele"], vcf_entry["ref"])

            results.append({
                "chrom": chrom, "pos": pos, "ref": var["ref"],
                "effect_allele": var["effect_allele"],
                "beta": beta, "eaf": eaf, "eaf_source": eaf_source,
                "status": status, "ea_is_ref": ea_is_ref,
                "p_homref": p_homref,
                "dosage_homref": dosage_homref,
                "dosage_meanimpute": dosage_mean,
                "dosage_weighted": dosage_weighted,
                "flanking_dp_left": vcf_entry.get("dp"),
                "flanking_dp_right": vcf_entry.get("dp"),
                "gap_left": 0, "gap_right": 0, "local_density": None,
            })
            score_homref += beta * dosage_homref
            score_meanimpute += beta * dosage_mean
            score_weighted += beta * dosage_weighted
            continue

        # --- Variant absent from VCF ---
        n_missing += 1

        cv = None
        used_chrom = None
        for cl in var["chrom_lookup"]:
            if cl in chrom_variants and chrom_variants[cl]:
                cv = chrom_variants[cl]
                used_chrom = cl
                break

        if cv is None or len(cv) == 0:
            p_homref = 0.01
            fl, fr, gl, gr = 0.0, 0.0, float("inf"), float("inf")
            ld_density = 0.0
        else:
            fl, fr, gl, gr = get_flanking_dp(cv, pos, k=flank_k)
            _, ld_density = local_variant_density(cv, pos, window)

            mappable = None
            if mappable_regions is not None:
                mappable = in_mappable_region(mappable_regions, used_chrom, pos)

            p_homref = estimate_p_homref(
                fl, fr, gl, gr, ld_density, genome_variant_rate,
                mappable=mappable, min_dp_callable=min_dp_callable,
                model=model)

        d_hr = homref_dosage(var["ref"], var["effect_allele"])
        dosage_homref = d_hr

        if eaf is not None:
            dosage_mean = 2.0 * eaf
        else:
            dosage_mean = 1.0
            eaf_source = "default_0.5"
            n_missing_eaf += 1

        dosage_weighted = p_homref * dosage_homref + \
                          (1.0 - p_homref) * dosage_mean

        ea_is_ref = effect_allele_is_ref(var["ref"], var["effect_allele"])

        results.append({
            "chrom": chrom, "pos": pos, "ref": var["ref"],
            "effect_allele": var["effect_allele"],
            "beta": beta, "eaf": eaf, "eaf_source": eaf_source,
            "status": "absent", "ea_is_ref": ea_is_ref,
            "p_homref": p_homref,
            "dosage_homref": dosage_homref,
            "dosage_meanimpute": dosage_mean,
            "dosage_weighted": dosage_weighted,
            "flanking_dp_left": fl, "flanking_dp_right": fr,
            "gap_left": gl, "gap_right": gr, "local_density": ld_density,
        })
        score_homref += beta * dosage_homref
        score_meanimpute += beta * dosage_mean
        score_weighted += beta * dosage_weighted

    # Summary
    print(f"\nPGS variants found in VCF: {n_found}", file=sys.stderr)
    print(f"PGS variants absent from VCF: {n_missing}", file=sys.stderr)
    if n_missing_eaf > 0:
        print(f"  WARNING: {n_missing_eaf} variants had no {population} "
              f"frequency; used EAF=0.5.", file=sys.stderr)
    print(f"\nScores:", file=sys.stderr)
    print(f"  Assume all hom-ref:  {score_homref:.6f}", file=sys.stderr)
    print(f"  Mean-impute all:     {score_meanimpute:.6f}", file=sys.stderr)
    print(f"  Coverage-weighted:   {score_weighted:.6f}", file=sys.stderr)

    return results, score_homref, score_meanimpute, score_weighted


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(results, out_path):
    columns = [
        "chrom", "pos", "ref", "effect_allele", "beta", "eaf", "eaf_source",
        "status", "ea_is_ref", "p_homref",
        "dosage_homref", "dosage_meanimpute", "dosage_weighted",
        "flanking_dp_left", "flanking_dp_right",
        "gap_left", "gap_right", "local_density",
    ]
    with open(out_path, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for r in results:
            vals = []
            for c in columns:
                v = r.get(c)
                if v is None:
                    vals.append(".")
                elif isinstance(v, float):
                    vals.append("Inf" if v == float("inf") else f"{v:.6g}")
                elif isinstance(v, bool):
                    vals.append(str(v))
                else:
                    vals.append(str(v))
            fh.write("\t".join(vals) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PGS scoring with coverage-confidence weighting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--vcf", required=True,
                        help="Input VCF file (plain or .gz)")
    parser.add_argument("--pgs", required=True,
                        help="Annotated PGS file (from pgs_annotate_freq.py)")
    parser.add_argument("--out", required=True,
                        help="Output TSV with per-variant results")
    parser.add_argument("--population", default="FREQ",
                        help="Population for allele frequency "
                             "(default: FREQ). Reads tgp_{POP} column.")
    parser.add_argument("--sample", default=None,
                        help="Sample ID in multi-sample VCF (default: first)")
    parser.add_argument("--mappability", default=None,
                        help="BED file of callable/mappable regions")
    parser.add_argument("--flank-k", type=int, default=10,
                        help="Flanking variants to examine each side "
                             "(default: 10)")
    parser.add_argument("--window", type=int, default=50000,
                        help="Window for local density (default: 50000)")
    parser.add_argument("--genome-variant-rate", type=float, default=0.001,
                        help="Expected variants/bp (default: 0.001)")
    parser.add_argument("--min-dp-callable", type=int, default=5,
                        help="Min DP for callable (default: 5; "
                             "only used by hand-tuned model)")
    parser.add_argument("--model", default=None,
                        help="Calibrated model JSON from "
                             "pgs_calibrate_model.py. If omitted, the "
                             "hand-tuned default model is used.")

    args = parser.parse_args()

    # Load calibrated model if provided
    model = None
    if args.model:
        import json
        with open(args.model) as f:
            model = json.load(f)

    results, s_hr, s_mi, s_w = score_pgs(
        vcf_path=args.vcf,
        pgs_path=args.pgs,
        population=args.population,
        sample_id=args.sample,
        flank_k=args.flank_k,
        window=args.window,
        genome_variant_rate=args.genome_variant_rate,
        mappability_path=args.mappability,
        min_dp_callable=args.min_dp_callable,
        model=model,
    )

    write_results(results, args.out)
    print(f"\nResults written to {args.out}", file=sys.stderr)
    print(f"homref\t{s_hr:.6f}\tmeanimpute\t{s_mi:.6f}\tweighted\t{s_w:.6f}")


if __name__ == "__main__":
    main()

