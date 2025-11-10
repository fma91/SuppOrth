#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SuppOrth Parser – DEVELOPMENT BRANCH
# Version: 0.3-dev
# Status: In testing – do not push to main




"""
Created on Wed Oct  8 15:01:52 2025

@author: avitia
"""
import pandas as pd
from itertools import product
import re
from Bio import SeqIO
import io




def safe_read_table(path, sep='\t'):  # Support function
    """
    Safely load predictor TSVs that may contain comments, empty lines, or no headers.
    Returns a clean pandas DataFrame.
    """
    # First pass: read as raw text
    with open(path, 'r') as f:
        lines = [line for line in f if line.strip() and not line.startswith('#')]

    # Peek first line
    first_line = lines[0].strip().split(sep)
    has_header = not all(re.fullmatch(r'\S+', v) for v in first_line)  # rough test
    # If header looks like IDs (e.g., "XGW....") treat as no header
    if all(re.match(r'^[A-Za-z0-9_.-]+$', x) for x in first_line) and not any("orth" in x.lower() for x in first_line):
        has_header = False

    # Reconstruct cleaned text
    clean_text = ''.join(lines)
    df = pd.read_csv(io.StringIO(clean_text), sep=sep, header=0 if has_header else None, dtype=str)
    return df

def create_predictorsDict(labels =None, paths = None):  #This is need to be run 1st, it uses the safe_read_table function
    df_dict = {}
    for p,l in zip(paths,labels):
        df_dict[l] = safe_read_table(p)
    return df_dict

def cleaner_predictorsDF(df_dict):   #This need to be run 2nd
    valF=df_dict.get('F',None)
    if valF is not None:
        valF = valF.drop(index=0)
        valF = valF[[valF.columns[1],valF.columns[2]]]
        df_dict['F'] = valF
        
    valO=df_dict.get('O',None)
    if valO is not None:
        valO = valO[[valO.columns[2],valO.columns[3]]]
        df_dict['O'] = valO
        
    valS=df_dict.get('S',None)
    if valS is not None:
        valS = valS.drop(index=0)
        valS = valS[[valS.columns[2],valS.columns[3]]]
        df_dict['S'] = valS
    return df_dict

#create the parsersing functiosn for all the outputs
def split_and_clean(entry):
    #Split by commas or spaces, trim whitespace, and remove numbers
    if pd.isna(entry) or not isinstance(entry, str):
        return []
    tokens = re.split(r"[,\s]+", entry.strip())
    return [t for t in tokens if t and not re.fullmatch(r"\d*\.?\d+", t)]

def load_fastas(path):
    fasta_records = SeqIO.to_dict(SeqIO.parse( path,'fasta')) #this need to be run 3rd to create the fastas records
    return fasta_records


def get_fasta_ids(fasta_records): # Support function.  #Extract sequence IDs from a dict or iterable of SeqRecord objects
    if isinstance(fasta_records, dict):
        return set(fasta_records.keys())
    else:
        return {rec.id.split()[0] for rec in fasta_records}


def parse_orthology_dfs(df_dict, strain_records1, strain_records2):  #This need to be run 4th and uses some support functions above
    """
    Unified parser for orthology DataFrames with exactly two columns
    (handles variable value structures: commas, spaces, or numeric scores).
    Parameters: df_dict : dict {'F': finder_df, 'B': broccoli_df, 'S': sonic_df}
        Each DataFrame must have exactly two columns containing IDs.
    strain_records : dict or iterable
        FASTA records already loaded (dict of SeqRecord or list of SeqRecord).
    Returns:   dict {'F': set_of_tuples, 'B': set_of_tuples, 'S': set_of_tuples}
    """
    strain1_ids = get_fasta_ids(strain_records1)
    strain2_ids = get_fasta_ids(strain_records2)
    orthology_dict = {}

    for tool_id, df in df_dict.items():
        # Expect exactly two columns, regardless of names
        local_df = df.copy(deep=False)
        local_df.columns = [str(c) for c in local_df.columns]
        a_col, b_col = local_df.columns[:2]

        tuples_set = set()
        for a_entry, b_entry in zip(local_df[a_col], local_df[b_col]):
            try:
                a_list = split_and_clean(a_entry)
                b_list = split_and_clean(b_entry)
                for comb in product(a_list, b_list):
                    #check the ids belons to the right species
                    both_strains_ids = strain1_ids|strain2_ids
                    if comb[0] not in both_strains_ids and comb[1] not in both_strains_ids:
                        continue
                    #check their are not paralogs
                    if comb[0] in strain1_ids and comb[1] in strain1_ids:
                        continue
                    if comb[0] in strain2_ids and comb[1] in strain2_ids:
                        continue
                    # Orient by strain FASTA
                    if comb[0] not in strain1_ids:
                        comb = (comb[1], comb[0])
                    tuples_set.add(comb)
            except Exception:
                continue  # skip malformed rows safely

        orthology_dict[tool_id] = tuples_set

    return orthology_dict

