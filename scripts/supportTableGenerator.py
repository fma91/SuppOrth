# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 01:11:55 2025

@author: Avitia
"""


import pandas as pd
import pickle
import json


def load_blast_as_dict(path):
    """
    Read a BLAST tabular file (outfmt 6) and return a dictionary:
        {(query, subject): (identity, evalue)}
    """
    # BLAST default outfmt 6 columns:
    cols = [
        'qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen',
        'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'nident'
    ]
    try:
        df = pd.read_csv(path, sep='\t', comment='#', names=cols, usecols=['qseqid', 'sseqid', 'pident', 'evalue'])
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return {}

    # build dictionary directly
    blast_dict = {(r.qseqid, r.sseqid): (r.pident, r.evalue) for _, r in df.iterrows()}
    return blast_dict



def create_support(pairs_id, orthology_dict, blast_dict1,blast_dict2):
     #unique_orthologues = orthology_dict['F'] | orthology_dict['O'] | orthology_dict['B'] | orthology_dict['S']
     # For each orthologue, record which tools support it
    orthologues_summary = []

    supporting_tools = []
    for tag, tool_set in orthology_dict.items():
        if pairs_id in tool_set:
            supporting_tools.append(tag)

    support_count = len(supporting_tools)
    identity, e_value = get_blast_score(pairs_id, blast_dict1, blast_dict2)
    orthologues_summary = [pairs_id[0],pairs_id[1],support_count, supporting_tools, identity, e_value]

    return orthologues_summary
         
        

def get_blast_score(pair_prot, blast_dict1, blast_dict2):  
    """
    Extract the BLAST values for each protein pair.
    """
    output= blast_dict1.get(pair_prot)
    if not output:
        output = blast_dict2.get((pair_prot[1], pair_prot[0]),[0,0])
    return output



def get_unified_support(orthology_dict, blast_dict1, blast_dict2, sp1='Specie1', sp2='Specie2'):
    """
    Creates orthologue predictions df across multiple tools for a given set of proteins.
    Returns a dataframe: sp1, sp2, 'Lv_support', 'SupportedBy', 'Identity', 'e_val'
    """
    unique_orthologues = orthology_dict['F'] | orthology_dict['O'] | orthology_dict['B'] | orthology_dict['S']

    df_unified_support = pd.DataFrame(columns=[sp1, sp2, 'Lv_support', 'SupportedBy', 'Identity', 'e_val'])
    for pair_i in unique_orthologues:
        df_unified_support.loc[len(df_unified_support)] = create_support(pair_i, orthology_dict, blast_dict1, blast_dict2)

    return df_unified_support



def collapse_isoforms(df, prot2geneSp1_path, prot2geneSp2_path):
    """
    Collapse isoform-level orthologues to gene-level, aggregating support information.
    
    Args:
        df (pd.DataFrame): Output from get_unified_support
        prot2geneSp1_path (str): Path to JSON dict mapping Specie1 proteins to genes
        prot2geneSp2_path (str): Path to JSON dict mapping Specie2 proteins to genes
        
    Returns:
        pd.DataFrame: Collapsed gene-level orthologues with unified support info
    """
    # Load protein-to-gene dictionaries
    with open(prot2geneSp1_path, 'r') as handle:
        prot2geneSp1 = json.load(handle)
    with open(prot2geneSp2_path, 'r') as handle:
        prot2geneSp2 = json.load(handle)

    # Identify species columns
    sp1_prot_col = df.columns[0]
    sp2_prot_col = df.columns[1]

    # Map proteins to genes
    df['GeneID1'] = df[sp1_prot_col].map(prot2geneSp1)
    df['GeneID2'] = df[sp2_prot_col].map(prot2geneSp2)

    # Group by gene-gene pairs
    grouped = df.groupby(['GeneID1', 'GeneID2'])

    collapsed_data = []

    for (gene1, gene2), group in grouped:
        prots1 = group[sp1_prot_col].tolist()
        prots2 = group[sp2_prot_col].tolist()
        lv_supports = group['Lv_support'].tolist()
        supported_by = group['SupportedBy'].tolist()
        identities = group['Identity'].tolist()
        e_vals = group['e_val'].tolist()

        all_supported_by = sorted(set(tag for sublist in supported_by for tag in sublist))

        collapsed_data.append({
            'GeneID1': gene1,
            'GeneID2': gene2,
            'Prots1list': prots1,
            'Prots2list': prots2,
            'Lv_support_unified': max(lv_supports),
            'Lv_support_list': lv_supports,
            'SupportedBy_unified': all_supported_by,
            'SupportedBy_list': supported_by,
            'Identity_min': min(identities),
            'Identity_max': max(identities),
            'Identity_list': identities,
            'e_val_min': min(e_vals),
            'e_val_max': max(e_vals),
            'e_val_list': e_vals
        })

    collapsed_df = pd.DataFrame(collapsed_data)
    return collapsed_df

