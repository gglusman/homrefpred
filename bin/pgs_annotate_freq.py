#!/usr/bin/env python3
"""
pgs_annotate_freq.py

Annotate a PGS Catalog scoring file with population allele frequencies
from 1000 Genomes Project (TGP) per-chromosome frequency tables.

For each variant in the PGS file, looks up the matching position in the
TGP frequency files and appends all population frequency columns.
Handles allele orientation: if the PGS effect allele is the TGP ALT,
frequencies are used directly; if it is the TGP REF, frequencies are
flipped (100 - freq).

Input
-----
  --pgs       PGS scoring file (PGS Catalog format, plain or .gz)
  --freq-dir  Directory with per-chromosome files: 1.tsv.gz .. 22.tsv.gz
  --out       Output annotated PGS file (TSV)

Output
------
The original PGS file with comment header preserved, plus additional
columns for each population frequency (FREQ, AFR, AMR, EUR, SAS, EAS,
and all subpopulations).  Frequencies are stored as proportions (0-1),
converted from the percentage-based TGP files.  Variants not found in
TGP get "." in all frequency columns.

Usage
-----
    python pgs_annotate_freq.py \\
        --pgs PGS000001.txt.gz \\
        --freq-dir TGP37.tableAllPop/ \\
        --out PGS000001.annotated.txt
"""

import argparse
import gzip
import os
import sys


def open_maybe_gzipped(path):
    """Open a file that may or may not be gzip-compressed."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def discover_freq_columns(freq_dir):
    """
    Read the header of one frequency file to discover population columns.

    Returns the ordered list of population column names (everything except
    CHROM, POS, ID, REF, ALT, GC, CONTEXT).
    """
    for chrom in range(1, 23):
        for ext in (".tsv.gz", ".tsv"):
            path = os.path.join(freq_dir, f"{chrom}{ext}")
            if os.path.exists(path):
                with open_maybe_gzipped(path) as fh:
                    header = fh.readline().rstrip("\n\r").split("\t")
                    skip = {"CHROM", "POS", "ID", "REF", "ALT", "GC", "CONTEXT"}
                    pop_cols = [h for h in header if h not in skip]
                    return header, pop_cols
    print("Error: no frequency files found in " + freq_dir, file=sys.stderr)
    sys.exit(1)


def load_chrom_freqs(freq_dir, chrom, pop_cols):
    """
    Load all variants for one chromosome from the TGP frequency file.

    Returns dict[(pos, ref, alt)] -> list of freq values (as strings,
    already converted to proportions).
    """
    table = {}

    for ext in (".tsv.gz", ".tsv"):
        path = os.path.join(freq_dir, f"{chrom}{ext}")
        if os.path.exists(path):
            break
    else:
        return table

    with open_maybe_gzipped(path) as fh:
        header = fh.readline().rstrip("\n\r").split("\t")
        col_map = {name: i for i, name in enumerate(header)}

        pos_idx = col_map["POS"]
        ref_idx = col_map["REF"]
        alt_idx = col_map["ALT"]
        pop_indices = []
        for pc in pop_cols:
            if pc in col_map:
                pop_indices.append(col_map[pc])
            else:
                pop_indices.append(None)

        for line in fh:
            fields = line.rstrip("\n\r").split("\t")
            try:
                pos = int(fields[pos_idx])
            except (ValueError, IndexError):
                continue

            ref = fields[ref_idx].strip()
            alt = fields[alt_idx].strip()

            freqs = []
            for pi in pop_indices:
                if pi is not None and pi < len(fields):
                    try:
                        freqs.append(float(fields[pi]) / 100.0)
                    except ValueError:
                        freqs.append(None)
                else:
                    freqs.append(None)

            table[(pos, ref, alt)] = freqs

    return table


def lookup_freqs(table, pos, pgs_effect_allele, pgs_other_allele, n_pops):
    """
    Look up frequencies for a PGS variant, handling allele orientation.

    Returns (list of freq floats or None, orientation_str).
    orientation_str is "direct" if effect=ALT, "flipped" if effect=REF,
    or "missing" if not found.
    """
    # Case 1: effect allele is ALT, other allele is REF
    if pgs_other_allele:
        key = (pos, pgs_other_allele, pgs_effect_allele)
        if key in table:
            return table[key], "direct"
        # Case 2: effect allele is REF, other allele is ALT
        key_rev = (pos, pgs_effect_allele, pgs_other_allele)
        if key_rev in table:
            raw = table[key_rev]
            flipped = [(1.0 - f) if f is not None else None for f in raw]
            return flipped, "flipped"

    # Fallback: no other_allele known, try all bases
    for base in ("A", "C", "G", "T"):
        if base == pgs_effect_allele:
            continue
        # effect is ALT?
        key = (pos, base, pgs_effect_allele)
        if key in table:
            return table[key], "direct"
        # effect is REF?
        key = (pos, pgs_effect_allele, base)
        if key in table:
            raw = table[key]
            flipped = [(1.0 - f) if f is not None else None for f in raw]
            return flipped, "flipped"

    return [None] * n_pops, "missing"


def parse_pgs_header_and_columns(pgs_path):
    """
    Read a PGS file and return (comment_lines, header_fields, data_lines).
    Each data_line is the raw string (rstripped).
    """
    comments = []
    header = None
    data_lines = []

    with open_maybe_gzipped(pgs_path) as fh:
        for line in fh:
            raw = line.rstrip("\n\r")
            if raw.startswith("#"):
                comments.append(raw)
                continue
            if header is None:
                header = raw
                continue
            data_lines.append(raw)

    return comments, header, data_lines


def main():
    parser = argparse.ArgumentParser(
        description="Annotate a PGS scoring file with TGP population "
                    "allele frequencies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pgs", required=True,
                        help="PGS scoring file (PGS Catalog format)")
    parser.add_argument("--freq-dir", required=True,
                        help="Directory with per-chrom TGP frequency files "
                             "(1.tsv.gz .. 22.tsv.gz)")
    parser.add_argument("--out", required=True,
                        help="Output annotated PGS file")

    args = parser.parse_args()

    # Discover population columns from frequency files
    print("Discovering frequency columns...", file=sys.stderr)
    freq_header, pop_cols = discover_freq_columns(args.freq_dir)
    print(f"  Found {len(pop_cols)} population columns: "
          f"{', '.join(pop_cols[:8])}{'...' if len(pop_cols) > 8 else ''}",
          file=sys.stderr)

    # Parse the PGS file
    print("Reading PGS file...", file=sys.stderr)
    comments, header_line, data_lines = parse_pgs_header_and_columns(args.pgs)
    header_fields = header_line.split("\t")
    col_map = {name.strip().lower(): i for i, name in enumerate(header_fields)}

    # Find required columns
    def get_col(names):
        for n in names:
            if n in col_map:
                return col_map[n]
        return None

    ci = get_col(["chr_name", "chrom", "chr", "chromosome"])
    pi = get_col(["chr_position", "pos", "position", "bp"])
    ei = get_col(["effect_allele", "alt", "allele", "a1"])
    ri = get_col(["other_allele", "ref", "reference_allele"])

    if ci is None or pi is None or ei is None:
        print("Error: PGS file must have chr_name, chr_position, and "
              "effect_allele columns.", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(data_lines)} variants in PGS file", file=sys.stderr)

    # Determine which chromosomes we need
    chroms_needed = set()
    for line in data_lines:
        fields = line.split("\t")
        if ci < len(fields):
            c = fields[ci].strip().replace("chr", "")
            if c:
                chroms_needed.add(c)

    print(f"  Chromosomes: {chroms_needed}", file=sys.stderr)

    # Prefixed column names for the output
    freq_col_names = [f"tgp_{pc}" for pc in pop_cols]

    # Process chromosome by chromosome to limit memory
    # First, group data lines by chromosome
    lines_by_chrom = {}
    line_indices_by_chrom = {}
    for idx, line in enumerate(data_lines):
        fields = line.split("\t")
        if ci < len(fields):
            c = fields[ci].strip().replace("chr", "")
            if c not in lines_by_chrom:
                lines_by_chrom[c] = []
                line_indices_by_chrom[c] = []
            lines_by_chrom[c].append(fields)
            line_indices_by_chrom[c].append(idx)

    # Prepare output array (one freq list per data line)
    freq_annotations = [None] * len(data_lines)
    n_found = 0
    n_flipped = 0
    n_missing = 0

    for chrom in sorted(chroms_needed):
        print(f"  Processing chrom {chrom}...", file=sys.stderr,
              end="", flush=True)

        chrom_table = load_chrom_freqs(args.freq_dir, chrom, pop_cols)
        print(f" {len(chrom_table)} TGP variants loaded...",
              file=sys.stderr, end="", flush=True)

        if chrom not in lines_by_chrom:
            print(" no PGS variants", file=sys.stderr)
            continue

        chrom_found = 0
        for fields, line_idx in zip(lines_by_chrom[chrom],
                                     line_indices_by_chrom[chrom]):
            pos_str = fields[pi].strip() if pi < len(fields) else ""
            ea = fields[ei].strip() if ei < len(fields) else ""
            oa = fields[ri].strip() if ri is not None and ri < len(fields) else None

            try:
                pos = int(pos_str)
            except ValueError:
                freq_annotations[line_idx] = [None] * len(pop_cols)
                n_missing += 1
                continue

            freqs, orient = lookup_freqs(
                chrom_table, pos, ea, oa, len(pop_cols))

            freq_annotations[line_idx] = freqs
            if orient == "missing":
                n_missing += 1
            elif orient == "flipped":
                n_flipped += 1
                chrom_found += 1
            else:
                chrom_found += 1

        n_found += chrom_found
        print(f" {chrom_found} matched", file=sys.stderr)

        # Free memory
        del chrom_table

    print(f"\nAnnotation summary:", file=sys.stderr)
    print(f"  Matched (direct): {n_found - n_flipped}", file=sys.stderr)
    print(f"  Matched (flipped, effect=ref): {n_flipped}", file=sys.stderr)
    print(f"  Not found in TGP: {n_missing}", file=sys.stderr)

    # Write output
    print(f"Writing annotated PGS to {args.out}...", file=sys.stderr)
    with open(args.out, "w") as fh:
        # Preserve comment lines
        for c in comments:
            fh.write(c + "\n")

        # Write header with new columns
        fh.write(header_line + "\t" + "\t".join(freq_col_names) + "\n")

        # Write data with frequency annotations
        for idx, line in enumerate(data_lines):
            freqs = freq_annotations[idx]
            if freqs is None:
                freqs = [None] * len(pop_cols)

            freq_strs = []
            for f in freqs:
                if f is None:
                    freq_strs.append(".")
                else:
                    freq_strs.append(f"{f:.6g}")

            fh.write(line + "\t" + "\t".join(freq_strs) + "\n")

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()

