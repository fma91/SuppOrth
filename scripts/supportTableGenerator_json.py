# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 01:11:55 2025

@author: Avitia
"""

import json
import pandas as pd
import pickle

def load_dicts(file_paths, predictor_labels, blast_paths):
    """
    Loads multiple JSON files into a labeled dictionary, and returns:
    - tools_dicts: a dict mapping each label to its loaded JSON dictionary
    - predictors_output_set: a set of all unique protein IDs across predictors
    - blast_paths: a list of the blast files
    Args:
        file_paths (list of str): Paths to predictor JSON files.
        predictor_labels (list of str): Labels for each predictor file.
        blast_output1 (str): Path to first BLAST JSON file.
        blast_output2 (str): Path to second BLAST JSON file.

    Returns:
        tuple: (tools_dicts, predictors_output_list, blats_paths)
    """
    if len(file_paths) != len(predictor_labels):
        raise ValueError("File_paths and predictor_labels must have the same length.")
    
    tools_dicts = {}
    for path, label in zip(file_paths, predictor_labels):
        with open(path, 'r') as handle:
            tools_dicts[label] = json.load(handle)

    # Getting Blast dicts 
    blast_dicts = []
    for path in blast_paths:
        with open(path, 'rb') as handle:
            blast_dicts.append(pickle.load(handle))

    blast_dict1 = blast_dicts[0]  # blast_homocont_dict
    blast_dict2 = blast_dicts[1]  # blast_conthomo_dict
    
    potential_orthologues_set = set().union(*(d.keys() for d in tools_dicts.values()))  # Get unique keys among dicts
    
    return tools_dicts, potential_orthologues_set, blast_dict1, blast_dict2



def create_support(protein_id, tool_dicts, blast_dict1, blast_dict2):  
    """
    Combine orthologue predictions across multiple tools for a given protein.
    Returns a dictionary: {orthologue_id: [support_count, support_tags, identity, e_value]}.
    """
    all_orthologues = []  # collect orthologues from all tools
    for tag, predictor_dict in tool_dicts:
        orthologue_list = predictor_dict.get(protein_id)
        if orthologue_list:
            all_orthologues += orthologue_list

    # Unique orthologues predicted by any tool
    unique_orthologues = list(set(all_orthologues))
    orthologue_summary = {}

    # For each orthologue, record which tools support it
    for orthologue_id in unique_orthologues:
        supporting_tools = []
        for tag, tool_dict in tool_dicts:
            orthologue_list = tool_dict.get(protein_id)
            if orthologue_list and orthologue_id in orthologue_list:
                supporting_tools.append(tag)

        support_count = len(supporting_tools)
        identity, e_value = get_blast_score(protein_id, orthologue_id, blast_dict1, blast_dict2)
        orthologue_summary[orthologue_id] = [support_count, supporting_tools, identity, e_value]
    
    return orthologue_summary



def get_blast_score(protein_id, orthologue_id, blast_dict1, blast_dict2):  
    """
    Extract the BLAST values for each protein pair.
    """
    output= blast_dict1.get((protein_id, orthologue_id))
    if not output:
        output = blast_dict2.get((protein_id, orthologue_id),[0,0])
    return output



def get_unified_support(potential_orthologues_set, tool_dicts, blast_dict1, blast_dict2, sp1='Specie1', sp2='Specie2'):
    """
    Creates orthologue predictions df across multiple tools for a given set of proteins.
    Returns a dataframe: sp1, sp2, 'Lv_support', 'SupportedBy', 'Identity', 'e_val'
    """
    df_unified_support = pd.DataFrame(columns=[sp1, sp2, 'Lv_support', 'SupportedBy', 'Identity', 'e_val'])

    for i in potential_orthologues_set:
        orthos = create_support(i, tool_dicts.items(), blast_dict1, blast_dict2)
        for j in orthos:
            a, b, c, d = orthos[j]
            df_unified_support.loc[len(df_unified_support)] = [i, j, a, b, c, d]

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

