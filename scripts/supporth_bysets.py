#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 00:43:06 2025

@author: avitia
"""

from itertools import combinations
import pandas as pd





def get_blast_score(pair_prot, blast_dict1, blast_dict2):  
    """
    Extract the BLAST values for each protein pair.
    """
    output= blast_dict1.get(pair_prot)
    if not output:
        output = blast_dict2.get((pair_prot[1], pair_prot[0]),[0,0])
    return output





def get_unified_support_dynamic(orthology_dict, blast_dict1, blast_dict2,
                                sp1='Specie1', sp2='Specie2'):
    """
    Dynamic set-based support builder for ANY number of predictors.
    orthology_dict: {'F': set(...), 'O': set(...), ...}
    Returns DataFrame: [sp1, sp2, 'Lv_support', 'SupportedBy', 'Identity', 'e_val']
    """
    # Normalize to sets (skip empty/malformed)
    tool_sets = {k: (v if isinstance(v, set) else set()) for k, v in orthology_dict.items()}
    tools = list(tool_sets.keys())
    n = len(tools)

    # Precompute unions for "others" to speed up exact-region calc
    # exact region for combo C = (∩ S[t] for t in C) \ (∪ S[u] for u not in C)
    records = []

    # Helper: safe union over an iterable of sets (handles empty)
    def union_all(iter_sets):
        sets_list = list(iter_sets)
        return set().union(*sets_list) if sets_list else set()

    # Iterate over all non-empty combinations
    for r in range(1, n + 1):
        for combo in combinations(tools, r):
            combo_set = tools_set_intersection(tool_sets, combo)
            if not combo_set:
                continue
            others = [t for t in tools if t not in combo]
            other_union = union_all(tool_sets[t] for t in others)
            region_pairs = combo_set - other_union  # exact region for this combo
            if not region_pairs:
                continue

            supported_by = list(combo)  # tags for this exact region
            lv_support = len(combo)

            for pair in region_pairs:
                identity, e_value = get_blast_score(pair, blast_dict1, blast_dict2)
                records.append([pair[0], pair[1], lv_support, supported_by, identity, e_value])

    df = pd.DataFrame(records, columns=[sp1, sp2, 'Lv_support', 'SupportedBy', 'Identity', 'e_val'])
    return df


def tools_set_intersection(tool_sets, combo):
    """Intersection of predictor sets in 'combo'. Empty-safe."""
    it = iter(combo)
    try:
        first = next(it)
    except StopIteration:
        return set()
    inter = set(tool_sets[first])  # copy
    for t in it:
        inter &= tool_sets[t]
        if not inter:
            break
    return inter
