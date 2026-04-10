#!/usr/bin/env python3
"""
pgs_annotate_freq_many.py

Annotate multiple PGS Catalog scoring files with population allele
frequencies from 1000 Genomes (TGP) per-chromosome frequency tables.

This is an efficient batch version of pgs_annotate_freq.py.  Instead of
re-reading the (large) TGP frequency files once per PGS file, it:

  1. Scans all PGS files in the input directory to collect the union
     of (chrom, pos) variant positions needed.
  2. Streams through each TGP chromosome file ONCE, building a single
     in-memory frequency table for just the positions needed.
  3. Re-reads each PGS file and emits an annotated version with all
     32 population frequency columns appended.

For N PGS files this turns N expensive freq-dir passes into 1.

Input
-----
  --pgs-dir   Directory containing PGS scoring files (.txt or .txt.gz)
  --freq-dir  Directory with per-chromosome TGP frequency files
              (1.tsv.gz .. 22.tsv.gz)
  --out-dir   Output directory for annotated PGS files

Output
------
For each input PGS file, a corresponding annotated file with the same
name in --out-dir.  The output preserves comment headers and adds 32
tgp_<POP> columns (one per population in the TGP files).

Usage
-----
    python pgs_annotate_freq_many.py \\
        --pgs-dir scoring_files/ \\
        --freq-dir TGP37.tableAllPop/ \\
        --out-dir annotated_lists/
"""

import argparse
import gzip
import os
import sys
from collections import defaultdict


def open_maybe_gzipped(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


# ---------------------------------------------------------------------------
# Discover PGS files in directory
# ---------------------------------------------------------------------------

def discover_pgs_files(pgs_dir):
    """Return list of PGS files in the directory (txt, tsv, or .gz versions)."""
    files = []
    for fname in sorted(os.listdir(pgs_dir)):
        if fname.startswith("."):
            continue
        path = os.path.join(pgs_dir, fname)
        if not os.path.isfile(path):
            continue
        # Common PGS file extensions
        lower = fname.lower()
        if (lower.endswith(".txt") or lower.endswith(".txt.gz") or
                lower.endswith(".prs") or lower.endswith(".prs.gz") or
                lower.endswith(".tsv") or lower.endswith(".tsv.gz") or
                lower.endswith(".pgs") or lower.endswith(".pgs.gz")):
            files.append(path)
    return files


# ---------------------------------------------------------------------------
# PGS file parsing
# ---------------------------------------------------------------------------

def parse_pgs_structure(pgs_path):
    """
    Read a PGS file and return its structure for both variant collection
    and later re-emission.

    Returns dict with:
      - comments: list of comment lines (starting with #)
      - header_line: the column header line (raw, no newline)
      - header_fields: parsed header fields
      - col_map: lowercase column name -> index
      - data_lines: list of raw data lines (rstripped)
      - col_indices: dict with ci, pi, ei, ri keys for the relevant columns
    """
    comments = []
    header_line = None
    data_lines = []

    with open_maybe_gzipped(pgs_path) as fh:
        for line in fh:
            raw = line.rstrip("\n\r")
            if raw.startswith("#"):
                comments.append(raw)
                continue
            if header_line is None:
                header_line = raw
                continue
            data_lines.append(raw)

    if header_line is None:
        return None

    header_fields = header_line.split("\t")
    col_map = {name.strip().lower(): i for i, name in enumerate(header_fields)}

    def get_col(names, required=True):
        for n in names:
            if n in col_map:
                return col_map[n]
        if required:
            return None
        return None

    ci = get_col(["chr_name", "chrom", "chr", "chromosome"])
    pi = get_col(["chr_position", "pos", "position", "bp"])
    ei = get_col(["effect_allele", "alt", "allele", "a1"])
    ri = get_col(["other_allele", "ref", "reference_allele", "a2"], required=False)

    if ci is None or pi is None or ei is None:
        return None

    return {
        "path": pgs_path,
        "comments": comments,
        "header_line": header_line,
        "header_fields": header_fields,
        "col_map": col_map,
        "data_lines": data_lines,
        "ci": ci, "pi": pi, "ei": ei, "ri": ri,
    }


# ---------------------------------------------------------------------------
# Collect needed positions across all PGS files
# ---------------------------------------------------------------------------

def collect_positions_needed(pgs_structures):
    """
    Walk all PGS files and collect the union of (chrom, pos) positions
    needed.  Chromosomes are normalized to bare numbers (no "chr" prefix).

    Returns dict[chrom (bare)] -> set of positions.
    """
    positions_by_chrom = defaultdict(set)

    for pgs in pgs_structures:
        ci, pi = pgs["ci"], pgs["pi"]
        for line in pgs["data_lines"]:
            fields = line.split("\t")
            if ci >= len(fields) or pi >= len(fields):
                continue
            chrom = fields[ci].strip()
            pos_str = fields[pi].strip()
            if not chrom or not pos_str:
                continue
            try:
                pos = int(pos_str)
            except ValueError:
                continue
            bare = chrom.replace("chr", "") if chrom.startswith("chr") else chrom
            positions_by_chrom[bare].add(pos)

    return dict(positions_by_chrom)


# ---------------------------------------------------------------------------
# Discover frequency columns from one TGP file
# ---------------------------------------------------------------------------

def discover_freq_columns(freq_dir):
    """Read header of any one frequency file to discover population columns."""
    for chrom in range(1, 23):
        for ext in (".tsv.gz", ".tsv"):
            path = os.path.join(freq_dir, f"{chrom}{ext}")
            if os.path.exists(path):
                with open_maybe_gzipped(path) as fh:
                    header = fh.readline().rstrip("\n\r").split("\t")
                    skip = {"CHROM", "POS", "ID", "REF", "ALT", "GC", "CONTEXT"}
                    pop_cols = [h for h in header if h not in skip]
                    return pop_cols
    print(f"Error: no frequency files found in {freq_dir}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Single-pass frequency loading: only positions we need
# ---------------------------------------------------------------------------

def load_needed_frequencies(freq_dir, positions_by_chrom, pop_cols):
    """
    Stream through each chromosome's TGP file ONCE and build a frequency
    table for only the positions we need.

    Returns dict[(chrom_bare, pos, ref, alt)] -> list of freq values
    (as proportions, converted from percentages).
    """
    freq_table = {}
    chroms_sorted = sorted(positions_by_chrom.keys(),
                           key=lambda x: int(x) if x.isdigit() else 1000)

    for chrom in chroms_sorted:
        positions_needed = positions_by_chrom[chrom]
        if not positions_needed:
            continue

        # Find the file
        path = None
        for ext in (".tsv.gz", ".tsv"):
            candidate = os.path.join(freq_dir, f"{chrom}{ext}")
            if os.path.exists(candidate):
                path = candidate
                break

        if path is None:
            print(f"  Warning: no frequency file for chrom {chrom} "
                  f"(needed {len(positions_needed)} positions)",
                  file=sys.stderr)
            continue

        print(f"  Chrom {chrom}: scanning {path} for "
              f"{len(positions_needed):,} positions...",
              file=sys.stderr, end="", flush=True)

        n_matched = 0
        with open_maybe_gzipped(path) as fh:
            header = fh.readline().rstrip("\n\r").split("\t")
            col_map = {name: i for i, name in enumerate(header)}

            pos_idx = col_map["POS"]
            ref_idx = col_map["REF"]
            alt_idx = col_map["ALT"]
            pop_indices = [col_map.get(pc) for pc in pop_cols]

            max_idx = max(pos_idx, ref_idx, alt_idx,
                          max(i for i in pop_indices if i is not None))

            for line in fh:
                fields = line.rstrip("\n\r").split("\t")
                if len(fields) <= max_idx:
                    continue

                try:
                    pos = int(fields[pos_idx])
                except ValueError:
                    continue

                if pos not in positions_needed:
                    continue

                ref = fields[ref_idx].strip()
                alt = fields[alt_idx].strip()

                freqs = []
                for pi in pop_indices:
                    if pi is not None:
                        try:
                            freqs.append(float(fields[pi]) / 100.0)
                        except ValueError:
                            freqs.append(None)
                    else:
                        freqs.append(None)

                freq_table[(chrom, pos, ref, alt)] = freqs
                n_matched += 1

        print(f" {n_matched:,} entries", file=sys.stderr)

    return freq_table


# ---------------------------------------------------------------------------
# Look up a PGS variant in the frequency table (handle allele orientation)
# ---------------------------------------------------------------------------

def lookup_freqs(freq_table, chrom_bare, pos, pgs_effect_allele,
                 pgs_other_allele, n_pops):
    """
    Look up frequencies for a PGS variant.  Handles allele orientation:
      - If effect_allele is the TGP ALT: use frequencies directly.
      - If effect_allele is the TGP REF: flip frequencies (1 - freq).

    Returns (list of floats or None, orientation).
    """
    if pgs_other_allele:
        # Try effect=ALT, other=REF
        key = (chrom_bare, pos, pgs_other_allele, pgs_effect_allele)
        if key in freq_table:
            return freq_table[key], "direct"
        # Try effect=REF, other=ALT
        key_rev = (chrom_bare, pos, pgs_effect_allele, pgs_other_allele)
        if key_rev in freq_table:
            raw = freq_table[key_rev]
            return [(1.0 - f) if f is not None else None for f in raw], "flipped"

    # Fallback: try all bases
    for base in ("A", "C", "G", "T"):
        if base == pgs_effect_allele:
            continue
        key = (chrom_bare, pos, base, pgs_effect_allele)
        if key in freq_table:
            return freq_table[key], "direct"
        key = (chrom_bare, pos, pgs_effect_allele, base)
        if key in freq_table:
            raw = freq_table[key]
            return [(1.0 - f) if f is not None else None for f in raw], "flipped"

    return [None] * n_pops, "missing"


# ---------------------------------------------------------------------------
# Write an annotated PGS file
# ---------------------------------------------------------------------------

def write_annotated_pgs(pgs, freq_table, pop_cols, out_path):
    """
    Write an annotated version of a PGS file: original content with
    tgp_<POP> columns appended.

    Returns (n_direct, n_flipped, n_missing) match counts.
    """
    n_pops = len(pop_cols)
    freq_col_names = [f"tgp_{pc}" for pc in pop_cols]

    n_direct = 0
    n_flipped = 0
    n_missing = 0

    ci, pi, ei, ri = pgs["ci"], pgs["pi"], pgs["ei"], pgs["ri"]

    with open(out_path, "w") as fh:
        # Comments
        for c in pgs["comments"]:
            fh.write(c + "\n")
        # Header with new columns
        fh.write(pgs["header_line"] + "\t" + "\t".join(freq_col_names) + "\n")

        # Data lines
        for line in pgs["data_lines"]:
            fields = line.split("\t")

            chrom = fields[ci].strip() if ci < len(fields) else ""
            pos_str = fields[pi].strip() if pi < len(fields) else ""
            ea = fields[ei].strip() if ei < len(fields) else ""
            oa = fields[ri].strip() if ri is not None and ri < len(fields) \
                else None

            try:
                pos = int(pos_str)
            except ValueError:
                pos = None

            if pos is None or not chrom:
                freqs = [None] * n_pops
                orient = "missing"
            else:
                bare = chrom.replace("chr", "") if chrom.startswith("chr") \
                    else chrom
                freqs, orient = lookup_freqs(
                    freq_table, bare, pos, ea, oa, n_pops)

            if orient == "direct":
                n_direct += 1
            elif orient == "flipped":
                n_flipped += 1
            else:
                n_missing += 1

            freq_strs = []
            for f in freqs:
                if f is None:
                    freq_strs.append(".")
                else:
                    freq_strs.append(f"{f:.6g}")

            fh.write(line + "\t" + "\t".join(freq_strs) + "\n")

    return n_direct, n_flipped, n_missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch-annotate multiple PGS files with TGP "
                    "population allele frequencies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pgs-dir", required=True,
                        help="Directory containing PGS scoring files")
    parser.add_argument("--freq-dir", required=True,
                        help="Directory with per-chrom TGP frequency files")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for annotated PGS files")

    args = parser.parse_args()

    # Discover PGS files
    print(f"Discovering PGS files in {args.pgs_dir}...", file=sys.stderr)
    pgs_files = discover_pgs_files(args.pgs_dir)
    if not pgs_files:
        print(f"  No PGS files found.", file=sys.stderr)
        sys.exit(1)
    print(f"  Found {len(pgs_files)} PGS file(s)", file=sys.stderr)

    # Discover frequency columns from one TGP file
    print(f"Discovering frequency columns...", file=sys.stderr)
    pop_cols = discover_freq_columns(args.freq_dir)
    print(f"  Found {len(pop_cols)} population columns: "
          f"{', '.join(pop_cols[:8])}{'...' if len(pop_cols) > 8 else ''}",
          file=sys.stderr)

    # Parse all PGS files (preserving structure for re-emission)
    print(f"Parsing {len(pgs_files)} PGS file(s)...", file=sys.stderr)
    pgs_structures = []
    for path in pgs_files:
        pgs = parse_pgs_structure(path)
        if pgs is None:
            print(f"  Skipping {path}: missing required columns",
                  file=sys.stderr)
            continue
        pgs_structures.append(pgs)
        print(f"  {os.path.basename(path)}: {len(pgs['data_lines']):,} variants",
              file=sys.stderr)

    if not pgs_structures:
        print("No valid PGS files to process.", file=sys.stderr)
        sys.exit(1)

    # Collect union of positions needed
    print(f"Collecting union of variant positions...", file=sys.stderr)
    positions_by_chrom = collect_positions_needed(pgs_structures)
    total_positions = sum(len(s) for s in positions_by_chrom.values())
    chroms_str = ", ".join(sorted(positions_by_chrom.keys(),
                                  key=lambda x: int(x) if x.isdigit() else 1000))
    print(f"  {total_positions:,} unique positions across "
          f"{len(positions_by_chrom)} chromosomes: {chroms_str}",
          file=sys.stderr)

    # Single pass over frequency files
    print(f"Loading frequencies (single pass over freq-dir)...",
          file=sys.stderr)
    freq_table = load_needed_frequencies(
        args.freq_dir, positions_by_chrom, pop_cols)
    print(f"  {len(freq_table):,} total frequency entries loaded",
          file=sys.stderr)

    # Make sure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Write each annotated PGS file
    print(f"\nWriting annotated PGS files to {args.out_dir}/...",
          file=sys.stderr)
    total_direct = 0
    total_flipped = 0
    total_missing = 0

    print('pgs_name', 'matched', 'total', 'pct', 'direct', 'flipped', 'missing', sep='\t', file=sys.stdout)
    for pgs in pgs_structures:
        basename = os.path.basename(pgs["path"])
        out_path = os.path.join(args.out_dir, basename)

        n_d, n_f, n_m = write_annotated_pgs(pgs, freq_table, pop_cols, out_path)
        total_direct += n_d
        total_flipped += n_f
        total_missing += n_m

        n_total = n_d + n_f + n_m
        n_matched = n_d + n_f
        pct = 100 * n_matched / n_total if n_total > 0 else 0
        print(basename, n_matched, n_total, pct, n_d, n_f, n_m, sep='\t', file=sys.stdout)
        #print(f"  {basename}: {n_matched}/{n_total} matched ({pct:.1f}%) "
        #      f"[direct={n_d}, flipped={n_f}, missing={n_m}]",
        #      file=sys.stderr)

    # Summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Summary", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  PGS files processed: {len(pgs_structures)}", file=sys.stderr)
    grand_total = total_direct + total_flipped + total_missing
    print(f"  Total variants:      {grand_total:,}", file=sys.stderr)
    print(f"    Matched (direct):  {total_direct:,}", file=sys.stderr)
    print(f"    Matched (flipped): {total_flipped:,}", file=sys.stderr)
    print(f"    Not in TGP:        {total_missing:,}", file=sys.stderr)
    print(f"\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
