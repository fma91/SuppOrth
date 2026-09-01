"""Why several predictors, and how they are kept from duplicating one another.

The four tools are not four copies of the same search. They were chosen because
they stand for different ways of calling an orthologue: graph inference from
sequence similarity (Broccoli, SonicParanoid), gene-tree reconciliation after
an alignment and a tree (OrthoFinder in MSA mode), and evolutionary-distance
consistency (OMA). Where a tool lets us pick the search program, the defaults
also differ — BLAST, DIAMOND, MMseqs2, OMA's own Smith-Waterman — so two tools
agreeing is not one BLAST counted twice.

SuppOrth then scores a pair by how many of those independent calls recovered
it. That is a filter, not a probability: the tools share input sequences and
some algorithmic ancestry, so a high count is "several methods said so", not
a calibrated confidence.

OMA is not launched here; the profile below is what a typical standalone run
uses, recorded when the user points ``--oma`` at an existing result.

SonicParanoid 2 does not call IQ-TREE. Its phylogenetic content is
domain-architecture (Pfam) analysis on top of an InParanoid-style graph. The
tree column therefore stays empty rather than claiming a program it never runs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stack:
    """One predictor's place in the complementary split."""

    search: str
    clustering: str
    alignment: str
    tree: str
    orthology: str


ORTHOFINDER = Stack(
    search="blast",
    clustering="MCL",
    alignment="mafft",
    tree="iqtree",
    orthology="gene-tree reconciliation",
)

BROCCOLI = Stack(
    search="diamond",
    clustering="Broccoli graph/network",
    alignment="Broccoli internal",
    tree="FastTree",
    orthology="hybrid graph and phylogeny",
)

SONICPARANOID = Stack(
    search="mmseqs",
    clustering="MCL",
    alignment="none",
    tree="none",
    orthology="InParanoid-like graph with Pfam domain analysis",
)

OMA = Stack(
    search="OMA native Smith-Waterman",
    clustering="OMA groups / HOGs",
    alignment="OMA internal",
    tree="OMA species-tree framework",
    orthology="evolutionary-distance consistency",
)
