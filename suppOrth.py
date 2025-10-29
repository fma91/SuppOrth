#!/usr/bin/env python3

import argparse
import json
import pandas as pd
from pathlib import Path
from supportTableGenerator_json import load_dicts, get_unified_support, collapse_isoforms


def parse_args():
    parser = argparse.ArgumentParser(description="Generate orthologue support tables from multiple predictors.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-b", "--bulk", help="Directory containing multiple *_result_dict.json files.")
    group.add_argument("-p", "--predictors", help="Comma-separated paths to predictor JSON files.")

    parser.add_argument("-l", "--labels", help="Comma-separated labels matching the predictors. Required if not using bulk mode.")
    parser.add_argument("-B", "--Blast_dicts", required=True, help="Comma-separated paths to two BLAST pickle files.")
    parser.add_argument("-p2g", "--prot2gene", help="Optional comma-separated paths to two protein-to-gene mapping files (pickle).")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output TSV file.")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Predictor input ---
    if args.bulk:
        predictor_paths = sorted([str(p) for p in Path(args.bulk).glob("*_result_dict.json")])
        if not predictor_paths:
            raise FileNotFoundError("No *_result_dict.json files found in the specified directory.")
        labels = [f"T{i+1}" for i in range(len(predictor_paths))]
    else:
        predictor_paths = args.predictors.split(",")
        labels = args.labels.split(",") if args.labels else None
        if not labels or len(labels) != len(predictor_paths):
            raise ValueError("You must provide the same number of labels as predictor files.")

    # --- BLAST input ---
    blast_paths = args.Blast_dicts.split(",")
    if len(blast_paths) != 2:
        raise ValueError("Exactly two BLAST JSON files must be provided.")

    # --- Load and process ---
    tools_dicts, protein_set, blast1, blast2 = load_dicts(predictor_paths, labels, blast_paths)
    df = get_unified_support(protein_set, tools_dicts, blast1, blast2)

    # --- Optional collapse by gene ---
    if args.prot2gene:
        p2g_paths = args.prot2gene.split(",")
        if len(p2g_paths) != 2:
            raise ValueError("You must provide exactly two files for protein-to-gene mapping.")
        df = collapse_isoforms(df, p2g_paths[0], p2g_paths[1])

    # --- Save output ---
    df.to_csv(args.output, sep="\t", index=False)
    print("Output written to: {args.output}")


if __name__ == "__main__":
    main()
