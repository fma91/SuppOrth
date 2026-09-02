"""Building the consensus table from several predictor results.

Each predictor has its own 0/1 column (the tool name), so a table can be
filtered without parsing a string. ``SupportedBy`` records the same set as a
comma-separated list of short labels (``F,S,B,O``), for a compact agreement
pattern.

Two cautions belong with any use of these numbers. Support is not a
probability, and the predictors are not independent observers — they share
inputs and, in part, algorithmic lineage. Treat a high count as a filter that
trades recall for precision, not as a calibrated confidence.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .canonical import Pair, PredictorResult
from .proteomes import Proteome, as_proteomes
from .similarity import SimilarityIndex

# Comma, not plus: a plus-joined pattern is awkward to split in R, pandas, or Excel.
SUPPORT_SEPARATOR = ","
SPECIES_COLUMNS = ("Species1", "Species2", "Protein1", "Protein2")
ANNOTATION_COLUMNS = ("Identity", "e_val", "Bitscore")


def support_tokens(value: object) -> list[str]:
    """Predictor names recorded in a SupportedBy cell."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).replace("+", SUPPORT_SEPARATOR)
    return [token for token in text.split(SUPPORT_SEPARATOR) if token]


def tool_flag_columns(frame: pd.DataFrame) -> list[str]:
    """0/1 columns named after predictors, in table order.

    They sit between ``SupportedBy`` and the alignment columns, so a TSV can
    be read back without remembering which tools were used.
    """
    recorded = frame.attrs.get("tool_columns")
    if recorded:
        return [name for name in recorded if name in frame.columns]
    flags: list[str] = []
    collecting = False
    for name in frame.columns:
        if name == "SupportedBy":
            collecting = True
            continue
        if not collecting:
            continue
        if name in ANNOTATION_COLUMNS:
            break
        flags.append(str(name))
    return flags


def protein_columns(frame: pd.DataFrame) -> tuple[str, str]:
    """Columns that hold the two protein identifiers of a pair."""
    if "Protein1" in frame.columns and "Protein2" in frame.columns:
        return "Protein1", "Protein2"
    return str(frame.columns[0]), str(frame.columns[1])


def build_table(
    results: Sequence[PredictorResult],
    similarity: SimilarityIndex | None,
    proteomes: Sequence[Proteome] | Proteome,
    *rest: Proteome,
) -> pd.DataFrame:
    loaded = as_proteomes(proteomes, *rest)
    owner: dict[str, str] = {}
    for proteome in loaded:
        for gene in proteome.ids:
            owner[gene] = proteome.label

    tools: list[str] = []
    labels_for: dict[str, str] = {}
    by_tool: dict[str, set[Pair]] = {}
    for result in results:
        if result.tool not in by_tool:
            tools.append(result.tool)
            labels_for[result.tool] = result.label or result.tool
        by_tool.setdefault(result.tool, set()).update(result.pairs)

    universe: set[Pair] = set()
    for pairs in by_tool.values():
        universe |= pairs

    records = []
    for left, right in sorted(universe):
        flags = {tool: int((left, right) in by_tool[tool]) for tool in tools}
        supported_by = [labels_for[tool] for tool in tools if flags[tool]]
        hit = similarity.get(left, right) if similarity else None
        records.append(
            {
                "Species1": owner.get(left, ""),
                "Species2": owner.get(right, ""),
                "Protein1": left,
                "Protein2": right,
                "Lv_support": len(supported_by),
                "SupportedBy": SUPPORT_SEPARATOR.join(supported_by),
                **flags,
                "Identity": hit.identity if hit else pd.NA,
                "e_val": hit.evalue if hit else pd.NA,
                "Bitscore": hit.bitscore if hit else pd.NA,
            }
        )

    frame = pd.DataFrame.from_records(
        records,
        columns=[
            *SPECIES_COLUMNS,
            "Lv_support",
            "SupportedBy",
            *tools,
            *ANNOTATION_COLUMNS,
        ],
    )
    frame.attrs["tool_columns"] = tools
    return frame.sort_values(
        ["Lv_support", "Bitscore"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)


def agreement_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """How many pairs each exact combination of predictors accounts for."""
    counts = Counter(zip(frame["SupportedBy"], frame["Lv_support"]))
    rows = [
        {"SupportedBy": pattern, "Lv_support": level, "n_pairs": count}
        for (pattern, level), count in counts.items()
    ]
    summary = pd.DataFrame(rows, columns=["SupportedBy", "Lv_support", "n_pairs"])
    return summary.sort_values(["Lv_support", "n_pairs"], ascending=False).reset_index(drop=True)


def collapse_to_genes(
    frame: pd.DataFrame,
    map1: Path,
    map2: Path,
    *,
    conservative: bool = True,
) -> pd.DataFrame:
    """Collapse isoform-level pairs to gene-level pairs.

    ``conservative`` reports the *minimum* support across the isoform pairs of a
    gene pair as the headline number, with the maximum kept alongside. Taking
    the maximum alone lets one lucky isoform hit from one predictor present a
    gene pair as strongly supported, which flatters the result.
    """
    prot2gene1 = _load_mapping(map1)
    prot2gene2 = _load_mapping(map2)

    sp1_col, sp2_col = protein_columns(frame)
    work = frame.copy()
    work["GeneID1"] = work[sp1_col].map(prot2gene1)
    work["GeneID2"] = work[sp2_col].map(prot2gene2)

    unmapped = int(work["GeneID1"].isna().sum() + work["GeneID2"].isna().sum())
    work = work.dropna(subset=["GeneID1", "GeneID2"])

    rows = []
    for (gene1, gene2), group in work.groupby(["GeneID1", "GeneID2"], sort=True):
        supported_sets = [set(support_tokens(s)) for s in group["SupportedBy"]]
        union = sorted(set().union(*supported_sets)) if supported_sets else []
        intersection = sorted(set.intersection(*supported_sets)) if supported_sets else []
        supports = list(group["Lv_support"])
        flags = {name: int(group[name].max()) for name in tool_flag_columns(frame)}

        rows.append(
            {
                "GeneID1": gene1,
                "GeneID2": gene2,
                "Lv_support": min(supports) if conservative else max(supports),
                "Lv_support_min": min(supports),
                "Lv_support_max": max(supports),
                "SupportedBy_all": SUPPORT_SEPARATOR.join(intersection),
                "SupportedBy_any": SUPPORT_SEPARATOR.join(union),
                **flags,
                "n_isoform_pairs": len(group),
                "Prots1": ",".join(sorted(set(group[sp1_col]))),
                "Prots2": ",".join(sorted(set(group[sp2_col]))),
                "Identity_max": group["Identity"].max(skipna=True),
                "Identity_min": group["Identity"].min(skipna=True),
                "e_val_min": group["e_val"].min(skipna=True),
            }
        )

    collapsed = pd.DataFrame(rows)
    collapsed.attrs["unmapped_proteins"] = unmapped
    if collapsed.empty:
        return collapsed
    return collapsed.sort_values(
        ["Lv_support", "Identity_max"], ascending=False, na_position="last"
    ).reset_index(drop=True)


def _load_mapping(path: Path) -> dict[str, str]:
    path = Path(path)
    if path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)
    if path.suffix == ".json":
        with path.open() as handle:
            return json.load(handle)
    mapping: dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                mapping[fields[0].strip()] = fields[1].strip()
    return mapping


def filter_table(
    frame: pd.DataFrame,
    *,
    support_lv: int = 3,
    identity_filter: float = 80.0,
    require: Iterable[str] = (),
    max_evalue: float | None = None,
) -> pd.DataFrame:
    """Keep pairs with enough independent calls, or a strong alignment.

    Default is support ≥ 3, or identity strictly above 80% when fewer tools
    agree. Missing identities never satisfy the identity side of that or.
    """
    out = frame
    high_support = out["Lv_support"] >= support_lv
    aligned = out["Identity"].notna() & (out["Identity"] > identity_filter)
    out = out[high_support | aligned]
    for label in require:
        if label in out.columns and label not in {"SupportedBy", "Lv_support"}:
            out = out[out[label] == 1]
        else:
            out = out[out["SupportedBy"].map(lambda s, wanted=label: wanted in support_tokens(s))]
    if max_evalue is not None:
        out = out[out["e_val"].notna() & (out["e_val"] <= max_evalue)]
    return out.reset_index(drop=True)


def venn_sets(results: Sequence[PredictorResult]) -> Mapping[str, set[Pair]]:
    return {result.label: set(result.pairs) for result in results}
