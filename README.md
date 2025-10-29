![Logo](supporth_banner.png)


# SuppOrth

SuppOrth (Support-based Orthologue integration) is a lightweight, flexible Python tool to unify and compare orthologue predictions across multiple tools.

It aggregates orthologue calls from various predictors, enriches them with BLAST alignment data, and optionally collapses isoform-level matches to gene-level orthology relationships.

-----------------------
Features:
-----------------------
- Combine results from multiple orthologue predictors (e.g., Broccoli, SonicParanoid, OrthoFinder, etc.)
- Support-aware integration of orthologue calls
- Use BLAST identity and e-value for additional scoring
- Optional collapsing of isoforms using protein-to-gene mapping
- JSON and Pickle support for input formats

-----------------------
Usage:
-----------------------

Run the tool with specific predictor files:

    python suppOrth.py -p predictors.json,predictors2.json -l B,F -B blast1.pickle,blast2.pickle -o output.tsv

Use bulk mode to automatically load all predictor JSONs in a folder:

    python suppOrth.py --bulk predictors_folder/ -B blast1.pickle,blast2.pickle -o output.tsv

To collapse to gene-level (if isoforms exist and mapping dictionaries are provided):

    python suppOrth.py ... -p2g prot2gene_species1.pickle,prot2gene_species2.pickle

-----------------------
Output:
-----------------------
Tab-separated file (`.tsv`) including:
- Protein or gene pairs
- Level of support
- Supporting tools
- BLAST identity & e-value
- (Optional) Aggregated gene-level support
