"""Tests for the parts of the pipeline that do not need the predictors installed."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from supporth import canonical, consensus, manifest, pipeline
from supporth.adapters.base import Adapter, MethodProfile, StageContext
from supporth.adapters.tabular import pairs_from_table
from supporth.proteomes import Proteome, ProteomeError, load_pair
from supporth.similarity import Hit, SimilarityIndex

SP1_FASTA = """>a1 first protein
MAAAAAAAAK
>a2
MBBBBBBBBK
>a3
MCCCCCCCCK
"""

SP2_FASTA = """>b1
MAAAAAAAAK
>b2
MBBBBBBBBK
>b3
MDDDDDDDDK
"""


@pytest.fixture()
def proteomes(tmp_path: Path) -> tuple[Proteome, Proteome]:
    first = tmp_path / "speciesA.faa"
    second = tmp_path / "speciesB.faa"
    first.write_text(SP1_FASTA)
    second.write_text(SP2_FASTA)
    return load_pair(first, second)


def test_labels_come_from_filenames(proteomes):
    sp1, sp2 = proteomes
    assert (sp1.label, sp2.label) == ("speciesA", "speciesB")
    assert sp1.size == 3 and sp2.size == 3


def test_overlapping_identifiers_are_rejected(tmp_path: Path):
    same = ">x1\nMAAAK\n"
    first, second = tmp_path / "one.faa", tmp_path / "two.faa"
    first.write_text(same)
    second.write_text(same)
    with pytest.raises(ProteomeError, match="occur in both"):
        load_pair(first, second)


def test_three_proteomes_are_loaded_in_input_order(tmp_path: Path):
    from supporth.proteomes import load_proteomes

    paths = []
    for name, seqs in (
        ("one.faa", ">a1\nMAAAK\n"),
        ("two.faa", ">b1\nMBBBK\n"),
        ("three.faa", ">c1\nMCCCK\n"),
    ):
        path = tmp_path / name
        path.write_text(seqs)
        paths.append(path)
    proteomes = load_proteomes(paths)
    assert [p.label for p in proteomes] == ["one", "two", "three"]
    assert proteomes[2].ids == frozenset({"c1"})


def test_labels_must_match_the_number_of_fastas(tmp_path: Path):
    from supporth.proteomes import ProteomeError, load_proteomes

    first, second, third = tmp_path / "a.faa", tmp_path / "b.faa", tmp_path / "c.faa"
    first.write_text(">a1\nMAAAK\n")
    second.write_text(">b1\nMBBBK\n")
    third.write_text(">c1\nMCCCK\n")
    with pytest.raises(ProteomeError, match="3 FASTA"):
        load_proteomes((first, second, third), labels=("A", "B"))


def test_cli_accepts_more_than_two_fastas():
    from supporth.cli import build_parser

    args = build_parser().parse_args(
        ["run", "a.faa", "b.faa", "c.faa", "-o", "out"]
    )
    assert [p.name for p in args.fastas] == ["a.faa", "b.faa", "c.faa"]


def test_cli_accepts_a_directory_of_fastas(tmp_path: Path):
    from supporth.cli import _run_fastas, build_parser

    folder = tmp_path / "proteomes"
    folder.mkdir()
    (folder / "a.faa").write_text(">a1\nMAAAK\n")
    (folder / "b.faa").write_text(">b1\nMBBBK\n")
    (folder / "notes.txt").write_text("ignore me\n")
    args = build_parser().parse_args(["run", "-f", str(folder), "-o", "out"])
    paths = _run_fastas(args)
    assert [p.name for p in paths] == ["a.faa", "b.faa"]


def test_cli_rejects_files_and_directory_together():
    from supporth.cli import _run_fastas, build_parser
    from supporth.proteomes import ProteomeError

    args = build_parser().parse_args(
        ["run", "a.faa", "b.faa", "-f", "proteomes", "-o", "out"]
    )
    with pytest.raises(ProteomeError, match="not both"):
        _run_fastas(args)


def test_pairs_are_oriented_and_filtered(proteomes):
    sp1, sp2 = proteomes
    raw = [
        ("a1", "b1"),   # already oriented
        ("b2", "a2"),   # reversed, must be flipped
        ("a1", "a2"),   # same species, dropped
        ("a3", "zzz"),  # unknown identifier, dropped
    ]
    result = canonical.build_result("stub", "T", raw, sp1, sp2)

    assert result.pairs == {("a1", "b1"), ("a2", "b2")}
    assert result.dropped_same_species == 1
    assert result.dropped_unknown == 1


def test_tabular_parser_ignores_scores_and_headers(tmp_path: Path, proteomes):
    sp1, sp2 = proteomes
    table = tmp_path / "messy.tsv"
    table.write_text(
        "# a comment\n"
        "group_id\tspecies_a\tspecies_b\tscore\n"
        "OG1\ta1 1.000\tb1 0.850\t42\n"
        "OG2\ta2,a3\tb2\t7\n"
        "OG3\tonly_junk\t0.5\n"
    )
    pairs = set(pairs_from_table(table, sp1.ids, sp2.ids))
    assert pairs == {("a1", "b1"), ("a2", "b2"), ("a3", "b2")}


def test_tabular_parser_keeps_ncbi_pipe_identifiers(tmp_path: Path):
    first = tmp_path / "Mgenitalium.faa"
    second = tmp_path / "Mhyopneumoniae.faa"
    first.write_text(">gi|3844967|gb|AAC71606.1|\nMAAAK\n")
    second.write_text(">gi|144227418|gb|AAZ44097.2|\nMAAAK\n")
    sp1, sp2 = load_pair(first, second)
    table = tmp_path / "pairs.tsv"
    table.write_text("gi|144227418|gb|AAZ44097.2|\tgi|3844967|gb|AAC71606.1|\n")
    pairs = set(pairs_from_table(table, sp1.ids, sp2.ids))
    assert pairs == {("gi|3844967|gb|AAC71606.1|", "gi|144227418|gb|AAZ44097.2|")}


def test_tabular_parser_emits_all_cross_species_pairs_on_a_row(tmp_path: Path):
    from supporth.proteomes import load_proteomes

    paths = []
    for name, header in (("one.faa", "a1"), ("two.faa", "b1"), ("three.faa", "c1")):
        path = tmp_path / name
        path.write_text(f">{header}\nMAAAK\n")
        paths.append(path)
    proteomes = load_proteomes(paths)
    table = tmp_path / "og.tsv"
    table.write_text("OG1\ta1\tb1\tc1\n")
    pairs = set(pairs_from_table(table, *[p.ids for p in proteomes]))
    assert pairs == {("a1", "b1"), ("a1", "c1"), ("b1", "c1")}


def test_consensus_counts_support_and_records_the_pattern(proteomes):
    sp1, sp2 = proteomes
    results = [
        canonical.PredictorResult("t1", "F", {("a1", "b1"), ("a2", "b2")}),
        canonical.PredictorResult("t2", "S", {("a1", "b1")}),
        canonical.PredictorResult("t3", "B", {("a1", "b1"), ("a3", "b3")}),
    ]
    index = SimilarityIndex()
    index.add("a1", "b1", Hit(88.5, 1e-40, 200.0))

    table = consensus.build_table(results, index, sp1, sp2)

    top = table.iloc[0]
    assert (top["Species1"], top["Species2"]) == ("speciesA", "speciesB")
    assert (top["Protein1"], top["Protein2"]) == ("a1", "b1")
    assert top["Lv_support"] == 3
    assert top["SupportedBy"] == "F,S,B"
    assert (top["t1"], top["t2"], top["t3"]) == (1, 1, 1)
    assert top["Identity"] == 88.5

    unaligned = table[table["Protein1"] == "a2"].iloc[0]
    assert unaligned["Lv_support"] == 1
    assert unaligned["SupportedBy"] == "F"
    assert (unaligned["t1"], unaligned["t2"], unaligned["t3"]) == (1, 0, 0)
    assert pd.isna(unaligned["Identity"]), "missing alignments must be NA, not 0.0"


def test_agreement_summary_partitions_the_pairs(proteomes):
    sp1, sp2 = proteomes
    results = [
        canonical.PredictorResult("t1", "F", {("a1", "b1"), ("a2", "b2")}),
        canonical.PredictorResult("t2", "S", {("a1", "b1")}),
    ]
    table = consensus.build_table(results, None, sp1, sp2)
    summary = consensus.agreement_summary(table)

    assert summary["n_pairs"].sum() == len(table)
    assert set(summary["SupportedBy"]) == {"F,S", "F"}


def test_default_filter_keeps_high_support_or_high_identity():
    table = pd.DataFrame(
        {
            "Lv_support": [4, 3, 2, 2, 1],
            "Identity": [40.0, 50.0, 90.0, 80.0, pd.NA],
        }
    )
    kept = consensus.filter_table(table)
    # 4 and 3 pass on support; 90% passes identity > 80; 80% does not; NA does not.
    assert list(kept["Lv_support"]) == [4, 3, 2]


def test_supportLv_and_identity_filter_are_honoured():
    table = pd.DataFrame({"Lv_support": [2, 1], "Identity": [50.0, 90.0]})
    kept = consensus.filter_table(table, support_lv=2, identity_filter=40)
    assert len(kept) == 2
    strict = consensus.filter_table(table, support_lv=4, identity_filter=95)
    assert strict.empty


def test_collapse_is_conservative_by_default(tmp_path: Path, proteomes):
    sp1, sp2 = proteomes
    table = pd.DataFrame(
        {
            "speciesA": ["a1", "a2"],
            "speciesB": ["b1", "b2"],
            "Lv_support": [3, 1],
            "SupportedBy": ["t1,t2,t3", "t1"],
            "t1": [1, 1],
            "t2": [1, 0],
            "t3": [1, 0],
            "Identity": [90.0, 40.0],
            "e_val": [1e-50, 1e-3],
            "Bitscore": [300.0, 50.0],
        }
    )
    map1 = tmp_path / "m1.json"
    map2 = tmp_path / "m2.json"
    map1.write_text(json.dumps({"a1": "geneA", "a2": "geneA"}))
    map2.write_text(json.dumps({"b1": "geneB", "b2": "geneB"}))

    collapsed = consensus.collapse_to_genes(table, map1, map2)
    row = collapsed.iloc[0]

    assert row["n_isoform_pairs"] == 2
    assert row["Lv_support"] == 1, "conservative collapse reports the weakest isoform pair"
    assert row["Lv_support_max"] == 3
    assert row["SupportedBy_all"] == "t1"
    assert set(row["SupportedBy_any"].split(",")) == {"t1", "t2", "t3"}
    assert (row["t1"], row["t2"], row["t3"]) == (1, 1, 1)


def test_overlap_report_counts_exclusives_and_intersections():
    from supporth import figures

    results = [
        canonical.PredictorResult("orthofinder", "F", {("a1", "b1"), ("a2", "b2"), ("a3", "b3")}),
        canonical.PredictorResult("broccoli", "B", {("a1", "b1"), ("a4", "b4")}),
    ]
    report = figures.overlap_report(results)

    assert report.tools == ("orthofinder", "broccoli")
    assert report.matrix[0][0] == 3
    assert report.matrix[0][1] == 1
    by_tool = {item.tool: item for item in report.exclusives}
    assert by_tool["orthofinder"].count == 2
    assert by_tool["broccoli"].count == 1
    assert by_tool["orthofinder"].percent == pytest.approx(200 / 3)


def test_support_figure_writes_png_and_svg(tmp_path: Path):
    from supporth import figures

    pytest.importorskip("matplotlib")
    pytest.importorskip("venny4py")
    results = [
        canonical.PredictorResult("orthofinder", "F", {("a1", "b1"), ("a2", "b2")}),
        canonical.PredictorResult("broccoli", "B", {("a1", "b1")}),
        canonical.PredictorResult("sonicparanoid", "S", {("a1", "b1"), ("a3", "b3")}),
        canonical.PredictorResult("oma", "O", {("a3", "b3")}),
    ]
    destination = tmp_path / "suppOrth.png"
    written = figures.write_support_figure(results, destination, log=lambda *_: None)
    assert written == destination
    assert destination.is_file() and destination.stat().st_size > 0
    svg = destination.with_suffix(".svg")
    assert svg.is_file() and svg.stat().st_size > 0
    assert b"<svg" in svg.read_bytes()[:200]


def test_round_trip_through_canonical_json(tmp_path: Path):
    result = canonical.PredictorResult("t1", "F", {("a1", "b1"), ("a2", "b2")})
    path = canonical.write_json(result, tmp_path / "t1.json")
    assert canonical.read_json(path).pairs == result.pairs


def test_legacy_query_keyed_json_is_still_readable(tmp_path: Path):
    legacy = tmp_path / "old_result_dict.json"
    legacy.write_text(json.dumps({"a1": ["b1", "b2"], "a2": []}))
    assert canonical.read_json(legacy).pairs == {("a1", "b1"), ("a1", "b2")}


class StubAdapter(Adapter):
    name = "stub"
    label = "X"
    repo = "https://example.invalid/stub.git"
    default_ref = "v1"

    def check(self, options=None):
        return True, "ok (stub)"

    def run(self, context: StageContext) -> Path:
        out = context.work_dir / "results"
        out.mkdir(parents=True, exist_ok=True)
        (out / "pairs.tsv").write_text("a1\tb1\na2\tb2\n")
        return out

    def parse(self, native_root: Path, proteomes, *rest):
        loaded = (proteomes, *rest) if hasattr(proteomes, "ids") else tuple(proteomes)
        raw = list(pairs_from_table(native_root / "pairs.tsv", *[p.ids for p in loaded]))
        return canonical.build_result(self.name, self.label, raw, loaded)

    def method_profile(self, context: StageContext) -> MethodProfile:
        return MethodProfile(search="stub", clustering="stub")


def test_staged_inputs_use_the_extension_broccoli_requires(tmp_path, proteomes):
    from supporth.proteomes import stage_input_dir

    staged = stage_input_dir(tmp_path / "input", proteomes)
    names = sorted(p.name for p in staged.iterdir())

    assert names == ["speciesA.fasta", "speciesB.fasta"], (
        "Broccoli only scans for *.fasta and reports an empty input directory otherwise"
    )


def test_staging_replaces_inputs_from_an_earlier_run(tmp_path, proteomes):
    from supporth.proteomes import stage_input_dir

    destination = tmp_path / "input"
    destination.mkdir()
    (destination / "stale.fa").write_text(">old\nMK\n")

    staged = stage_input_dir(destination, proteomes)
    assert not (staged / "stale.fa").exists()


def test_orthofinder_default_stack_is_blast_mafft_iqtree():
    from supporth.adapters.orthofinder import OrthoFinder, parse_methods

    methods = parse_methods([])
    assert (methods.search, methods.mode, methods.aligner, methods.tree) == (
        "blast",
        "msa",
        "mafft",
        "iqtree",
    )
    assert set(methods.required_binaries()) == {"blastp", "makeblastdb", "mafft", "iqtree"}
    assert set(OrthoFinder().extra_binaries([])) == {"blastp", "makeblastdb", "mafft", "iqtree"}


def test_orthofinder_msa_mode_requires_the_selected_programs():
    from supporth.adapters.orthofinder import OrthoFinder, parse_methods

    options = "-S diamond -M msa -A mafft -T fasttree".split()
    methods = parse_methods(options)

    assert (methods.mode, methods.aligner, methods.tree) == ("msa", "mafft", "fasttree")
    assert set(OrthoFinder().extra_binaries(options)) == {"diamond", "mafft", "FastTree"}


def test_orthofinder_dendroblast_drops_the_aligner_and_tree():
    from supporth.adapters.orthofinder import parse_methods

    required = parse_methods("-M dendroblast".split()).required_binaries()
    assert set(required) == {"blastp", "makeblastdb"}


def test_orthofinder_method_profile_reports_the_configured_tree(tmp_path, proteomes):
    from supporth.adapters.orthofinder import OrthoFinder

    sp1, sp2 = proteomes
    context = StageContext(
        input_dir=tmp_path,
        work_dir=tmp_path,
        sp1=sp1,
        sp2=sp2,
        options="-M msa -A mafft -T iqtree".split(),
    )
    profile = OrthoFinder().method_profile(context).as_dict()

    assert profile["tree"] == "iqtree"
    assert profile["alignment"] == "mafft"
    assert profile["orthology"] == "gene-tree reconciliation"

    default = OrthoFinder().method_profile(
        StageContext(input_dir=tmp_path, work_dir=tmp_path, sp1=sp1, sp2=sp2)
    ).as_dict()
    assert default["tree"] == "iqtree"


def test_search_jobs_collapse_to_four_for_a_species_pair():
    from supporth.adapters import of_config

    assert of_config.search_job_count(2) == 4
    assert of_config.search_job_count(4) == 16
    assert of_config.search_job_count(2, double_blast=False) == 3
    # The reason the stock configuration wastes a big machine: four jobs.
    assert of_config.threads_per_job(32, of_config.search_job_count(2)) == 8
    assert of_config.threads_per_job(2, 4) == 1


def test_threaded_search_entry_is_registered_in_config(tmp_path: Path):
    from supporth.adapters import of_config

    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "__comment": "upstream comment",
        "diamond": {
            "program_type": "search",
            "db_cmd": "diamond makedb --ignore-warnings --in INPUT -d OUTPUT",
            "search_cmd": "diamond blastp -d DATABASE -q INPUT -o OUTPUT -p 1 --quiet",
        },
    }))

    name = of_config.register_threaded_search(config, "diamond", 8)
    written = json.loads(config.read_text())

    assert name == "supporth_diamond"
    assert "-p 8" in written[name]["search_cmd"]
    assert written["diamond"]["search_cmd"].endswith("--quiet"), "stock entry untouched"
    assert config.with_suffix(".json.supporth-orig").exists()
    # OrthoFinder skips the key named exactly __comment; anything else must be a dict.
    assert all(k == "__comment" or isinstance(v, dict) for k, v in written.items())


def test_threaded_blast_entry_gets_a_thread_flag(tmp_path: Path):
    from supporth.adapters import of_config

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"blast_gz": {"program_type": "search"}}))

    name = of_config.register_threaded_search(config, "blast", 12)
    command = json.loads(config.read_text())[name]["search_cmd"]

    # The built-in "blast" method is assembled in Python with no thread flag,
    # which is why it is replaced rather than patched.
    assert "-num_threads 12" in command
    assert name == "supporth_blast"


def test_unknown_search_program_passes_through_unchanged(tmp_path: Path):
    from supporth.adapters import of_config

    config = tmp_path / "config.json"
    config.write_text(json.dumps({}))
    assert of_config.register_threaded_search(config, "exotic_search", 8) == "exotic_search"


def test_search_option_is_replaced_not_duplicated():
    from supporth.adapters import of_config

    options = "-S diamond -M msa -T iqtree".split()
    assert of_config.strip_option(options, "-S") == "-M msa -T iqtree".split()


def test_blast_search_requires_blast_binaries():
    from supporth.adapters.orthofinder import OrthoFinder

    # BLAST as search still uses the default MSA stack, so MAFFT and IQ-TREE
    # remain required unless the caller opts out with -M dendroblast.
    assert set(OrthoFinder().extra_binaries("-S blast".split())) == {
        "blastp",
        "makeblastdb",
        "mafft",
        "iqtree",
    }
    assert set(OrthoFinder().extra_binaries("-S blast -M dendroblast".split())) == {
        "blastp",
        "makeblastdb",
    }


def test_tool_env_does_not_throttle_openmp_by_default():
    from supporth import shell

    assert "OMP_NUM_THREADS" not in shell.tool_env({}, threads=None) or \
        shell.tool_env({}, threads=None).get("OMP_NUM_THREADS") != "1"
    assert shell.tool_env(threads=8)["OMP_NUM_THREADS"] == "8"


def _executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def test_audit_links_a_divergent_bundled_copy_and_can_undo_it(tmp_path: Path, monkeypatch):
    from supporth import deps, paths

    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    monkeypatch.setenv(paths.ENV_HOME, str(home))

    _executable(home / "bin" / "diamond", "echo shared")
    predictor = tmp_path / "sonicparanoid"
    bundled = _executable(predictor / "bin" / "diamond", "echo bundled")

    found = deps.find_bundled({"sonicparanoid": predictor})
    assert len(found) == 1
    assert found[0].kind == "divergent"

    deps.link_bundled(found, log=lambda *_: None)
    assert bundled.is_symlink()
    assert (predictor / "bin" / "diamond.supporth-orig").exists()

    deps.restore_bundled({"sonicparanoid": predictor}, log=lambda *_: None)
    assert not bundled.is_symlink()
    assert "bundled" in bundled.read_text()


def test_audit_reports_a_foreign_platform_binary_as_unrunnable(tmp_path: Path, monkeypatch):
    from supporth import deps, paths

    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    monkeypatch.setenv(paths.ENV_HOME, str(home))

    predictor = tmp_path / "orthofinder"
    (predictor / "bin").mkdir(parents=True)
    # An ELF header on a machine that cannot execute it, as shipped by source
    # distributions built elsewhere.
    foreign = predictor / "bin" / "diamond"
    foreign.write_bytes(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64)
    foreign.chmod(0o755)

    found = deps.find_bundled({"orthofinder": predictor})
    assert found[0].kind == "unrunnable"


def test_already_linked_copies_are_not_relinked(tmp_path: Path, monkeypatch):
    from supporth import deps, paths

    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    monkeypatch.setenv(paths.ENV_HOME, str(home))

    # The shared entry is itself a symlink onto a system install, which is the
    # normal case and the one that broke naive full-path resolution.
    system = _executable(tmp_path / "system" / "diamond", "echo system")
    shared = home / "bin" / "diamond"
    shared.symlink_to(system)

    predictor = tmp_path / "orthofinder"
    (predictor / "bin").mkdir(parents=True)
    bundled = predictor / "bin" / "diamond"
    bundled.symlink_to(shared)

    found = deps.find_bundled({"orthofinder": predictor})
    assert len(found) == 1
    assert found[0].kind == "linked", "a link onto the shared bin is not a duplicate"


def test_orthofinder_install_includes_iqtree_not_fasttree():
    from supporth import deps

    default = deps.binaries_for(["orthofinder"])
    broccoli = deps.binaries_for(["broccoli"])

    assert "iqtree" in default and "mafft" in default and "blastp" in default
    assert "FastTree" not in default
    assert "FastTree" in broccoli
    assert "iqtree" not in broccoli


def test_complementary_stacks_do_not_share_a_search_program():
    from supporth.method_stack import BROCCOLI, OMA, ORTHOFINDER, SONICPARANOID

    searches = {ORTHOFINDER.search, BROCCOLI.search, SONICPARANOID.search, OMA.search}
    assert len(searches) == 4
    assert ORTHOFINDER.tree == "iqtree"
    assert BROCCOLI.tree == "FastTree"
    assert SONICPARANOID.tree == "none"


def _complete_pfam_db(adapter, directory: Path) -> Path:
    """A profile database with every file SonicParanoid insists on."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in adapter.pfam_files():
        (directory / name).write_text(f"contents of {name}")
    return directory


def test_sonicparanoid_is_not_ready_until_pfam_is_indexed(tmp_path, monkeypatch):
    from supporth.adapters.sonicparanoid import SonicParanoid

    adapter = SonicParanoid()
    root = tmp_path / "sonicparanoid"
    profile_db = adapter.profile_db(root)
    profile_db.mkdir(parents=True)

    monkeypatch.setattr(SonicParanoid, "installed_root", lambda self: root)
    monkeypatch.setattr(
        Adapter, "check", lambda self, options=None: (True, "ok (sonicparanoid)")
    )

    # A half-built index must not read as ready: SonicParanoid throws away a
    # partial database and starts again, so the run would stall unexpectedly.
    (profile_db / "pfama.mmseqs.idx.0").write_text("partial")
    ready, detail = adapter.check()
    assert not ready
    assert "not indexed" in detail

    # Nor may the marker alone vouch for a database whose files went missing.
    (profile_db / adapter.PFAM_MARKER).write_text("prepared by suppOrth\n")
    assert not adapter.check()[0]

    _complete_pfam_db(adapter, profile_db)
    ready, detail = adapter.check()
    assert ready and detail == "ok (sonicparanoid)"


def test_oma_is_not_launched_and_reads_pairwise_tables(tmp_path, proteomes):
    from supporth.adapters.oma import OMA

    sp1, sp2 = proteomes
    adapter = OMA()
    run = tmp_path / "oma_run" / "Output" / "PairwiseOrthologs"
    run.mkdir(parents=True)
    (run / "tmuris-contortus.txt").write_text(
        "# Format: Protein 1<tab>Protein 2<tab>ID1<tab>ID2\n"
        "1\t2\ta1\tb1 extra description\t1:1\t12\n"
        "3\t4\ta2\tb2\t1:many\n"
    )
    (run / "tmuris-contortus_cutted.txt").write_text("a3\tb3\n")
    (tmp_path / "oma_run" / "Output" / "OrthologousGroups.txt").write_text(
        "OMA00001\tsp1:a1\tsp2:b1\n"
    )

    tables = adapter.result_tables(tmp_path / "oma_run")
    assert [p.name for p in tables] == ["tmuris-contortus.txt"]

    result = adapter.parse(tmp_path / "oma_run", sp1, sp2)
    assert result.pairs == {("a1", "b1"), ("a2", "b2")}


def test_pipeline_ingests_user_supplied_oma(tmp_path, proteomes, monkeypatch):
    sp1, sp2 = proteomes
    oma_root = tmp_path / "oma" / "Output" / "PairwiseOrthologs"
    oma_root.mkdir(parents=True)
    (oma_root / "A-B.txt").write_text("1\t2\ta1\tb1\t1:1\n")

    monkeypatch.setattr(pipeline, "resolve", lambda _: [StubAdapter()])
    config = pipeline.RunConfig(
        fasta1=sp1.fasta,
        fasta2=sp2.fasta,
        outdir=tmp_path / "out",
        annotate="none",
        oma=tmp_path / "oma",
        support_lv=1,
    )
    outcome = pipeline.run_pipeline(config, log=lambda *_: None)

    names = [r.tool for r in outcome["reports"]]
    assert names == ["stub", "oma"]
    assert "oma" in outcome["table"].columns
    pair = outcome["table"]
    row = pair[(pair["Protein1"] == "a1") & (pair["Protein2"] == "b1")].iloc[0]
    assert row["oma"] == 1
    assert row["stub"] == 1


def test_resolve_rejects_oma_as_a_runnable_tool():
    from supporth import adapters

    with pytest.raises(KeyError, match="--oma"):
        adapters.resolve("oma")


def test_sonicparanoid_prefers_its_pairwise_table_over_derived_groups(tmp_path, proteomes):
    """Regression: the pairwise file is named 'pairwise_orthologs.<run>.tsv'.

    A pattern of '*ortholog*pairs*' does not match that, so the parser fell
    through to the group table, which is derived from the pairs and had lost
    some of them by then -- 130 real orthologues on a worm proteome pair.
    """
    from supporth.adapters.sonicparanoid import SonicParanoid

    sp1, sp2 = proteomes
    adapter = SonicParanoid()
    run = tmp_path / "results" / "runs" / "sp2_123_custom_mmseqs"
    (run / "ortholog_groups").mkdir(parents=True)

    pairwise = run / "pairwise_orthologs.sp2_123_custom_mmseqs.tsv"
    pairwise.write_text("a1\tb1\na2\tb2\n")
    (run / "ortholog_groups" / "ortholog_groups.tsv").write_text(
        "group_id\tgroup_size\tsp1.fa\tsp2.fa\n1\t2\ta1\tb1\n"
    )
    (run / "ortholog_groups" / "not_assigned_genes.ortholog_groups.tsv").write_text("a3\n")
    (run / "ortholog_groups" / "flat.ortholog_groups.tsv").write_text("a1\tb1\n")

    assert adapter._result_tables(tmp_path) == [pairwise]

    result = adapter.parse(tmp_path, sp1, sp2)
    assert result.pairs == {("a1", "b1"), ("a2", "b2")}


def test_adopting_an_incomplete_pfam_database_is_refused(tmp_path):
    from supporth.adapters.base import AdapterError
    from supporth.adapters.sonicparanoid import SonicParanoid

    adapter = SonicParanoid()
    root = tmp_path / "sonicparanoid"
    source = _complete_pfam_db(adapter, tmp_path / "elsewhere" / "profile_db")

    # The shards MMseqs2 leaves behind when interrupted look like an index but
    # are not one, and adopting them would trigger the rebuild we are avoiding.
    (source / "pfama.mmseqs.idx").unlink()
    (source / "pfama.mmseqs.idx.0").write_text("shard")

    with pytest.raises(AdapterError, match="not a complete profile database"):
        adapter.adopt_pfam_profiles(root, source, log=lambda *_: None)
    assert not adapter.profile_db(root).exists(), "a refused source must leave nothing behind"

    assert adapter.missing_pfam_files(source) == ["pfama.mmseqs.idx"]


def test_adopting_a_complete_pfam_database_marks_it_ready(tmp_path):
    from supporth.adapters.sonicparanoid import SonicParanoid

    adapter = SonicParanoid()
    root = tmp_path / "sonicparanoid"
    source = _complete_pfam_db(adapter, tmp_path / "elsewhere" / "profile_db")

    adapter.adopt_pfam_profiles(root, source, log=lambda *_: None)

    assert adapter.pfam_ready(root)
    assert (adapter.profile_db(root) / "pfama.mmseqs.idx").read_text() == (
        "contents of pfama.mmseqs.idx"
    )
    # A directory holding profile_db/ is accepted too, since that is how the
    # database is usually carried around.
    other = tmp_path / "second"
    adapter.adopt_pfam_profiles(other, source.parent, log=lambda *_: None)
    assert adapter.pfam_ready(other)


def test_disk_reclaims_a_predictors_abandoned_artefacts(tmp_path, monkeypatch):
    from supporth import disk, paths
    from supporth.adapters.sonicparanoid import SonicParanoid

    monkeypatch.setenv(paths.ENV_HOME, str(tmp_path / "home"))
    paths.ensure_layout()

    adapter = SonicParanoid()
    root = paths.tool_dir("sonicparanoid", "2.0.9")
    profile_db = adapter.profile_db(root)
    profile_db.mkdir(parents=True)
    (profile_db / "pfama.mmseqs.idx.0").write_text("x" * 900)
    (profile_db / "pfama.mmseqs").write_text("keep me")

    manifest.record_predictor(
        "sonicparanoid",
        ref="2.0.9",
        commit="def456",
        source="https://example.invalid/sp.git",
        root=root,
        entrypoint=str(root / "site-packages" / "bin" / "sonicparanoid"),
    )

    # The disk report cannot recognise a half-built index on its own; it has to
    # ask the adapter, which is the only thing that knows what one looks like.
    items = disk.reclaimable()
    assert [item.path for item in items] == [profile_db / "pfama.mmseqs.idx.0"]

    disk.remove(items)
    assert (profile_db / "pfama.mmseqs").exists(), "the database itself must survive"

    _complete_pfam_db(adapter, profile_db)
    (profile_db / adapter.PFAM_MARKER).write_text("ready\n")
    assert disk.reclaimable() == [], "a finished index is not a leftover"


def test_disk_separates_leftovers_from_the_live_installation(tmp_path, monkeypatch):
    from supporth import disk, paths

    monkeypatch.setenv(paths.ENV_HOME, str(tmp_path / "home"))
    paths.ensure_layout()

    live = paths.tool_dir("orthofinder", "v2.5.5")
    (live / "scripts_of").mkdir(parents=True)
    (live / "scripts_of" / "main.py").write_text("x" * 100)
    (live / "build" / "temp").mkdir(parents=True)
    (live / "build" / "temp" / "object.o").write_text("y" * 500)

    stale = paths.tool_dir("orthofinder", "v2.5.4")
    stale.mkdir(parents=True)
    (stale / "old.py").write_text("z" * 300)

    manifest.record_predictor(
        "orthofinder",
        ref="v2.5.5",
        commit="abc123",
        source="https://example.invalid/of.git",
        root=live,
        entrypoint=str(live / "orthofinder.py"),
    )

    safe = disk.reclaimable()
    assert [item.path for item in safe if item.cost == "nothing"] == [live / "build"]

    stale_items = [item for item in safe if item.cost == "refetch"]
    assert [item.path for item in stale_items] == [stale]

    disk.remove([item for item in safe if item.cost == "nothing"])
    assert not (live / "build").exists()
    assert (live / "scripts_of" / "main.py").exists(), "the installed tree must survive"
    assert stale.exists(), "a checkout costing a refetch is only removed on request"


def test_pipeline_runs_end_to_end_and_supports_resume(tmp_path, proteomes, monkeypatch):
    sp1, sp2 = proteomes
    monkeypatch.setattr(pipeline, "resolve", lambda _: [StubAdapter()])

    config = pipeline.RunConfig(
        fasta1=sp1.fasta,
        fasta2=sp2.fasta,
        outdir=tmp_path / "out",
        annotate="none",
        support_lv=1,
    )
    outcome = pipeline.run_pipeline(config, log=lambda *_: None)

    assert len(outcome["table"]) == 2
    assert outcome["output"].exists()
    assert (tmp_path / "out" / "provenance.json").exists()
    assert outcome["reports"][0].status == "ok"

    config.resume = True
    again = pipeline.run_pipeline(config, log=lambda *_: None)
    assert again["reports"][0].status == "reused"


def test_pipeline_runs_three_proteomes_as_one_job(tmp_path, monkeypatch):

    paths = []
    for name, seqs in (
        ("one.faa", ">a1\nMAAAK\n"),
        ("two.faa", ">b1\nMBBBK\n"),
        ("three.faa", ">c1\nMCCCK\n"),
    ):
        path = tmp_path / name
        path.write_text(seqs)
        paths.append(path)

    class TripleStub(StubAdapter):
        def run(self, context: StageContext) -> Path:
            out = context.work_dir / "results"
            out.mkdir(parents=True, exist_ok=True)
            (out / "pairs.tsv").write_text("a1\tb1\na1\tc1\nb1\tc1\n")
            return out

    monkeypatch.setattr(pipeline, "resolve", lambda _: [TripleStub()])
    config = pipeline.RunConfig(
        fastas=tuple(paths),
        outdir=tmp_path / "out3",
        annotate="none",
        support_lv=1,
    )
    outcome = pipeline.run_pipeline(config, log=lambda *_: None)
    table = outcome["table"]
    assert set(zip(table["Protein1"], table["Protein2"])) == {
        ("a1", "b1"),
        ("a1", "c1"),
        ("b1", "c1"),
    }
    assert set(table["Species1"]) <= {"one", "two"}
    assert {"one", "two", "three"} == set(table["Species1"]) | set(table["Species2"])


def test_pipeline_reports_missing_predictors_before_running(tmp_path, proteomes, monkeypatch):
    sp1, sp2 = proteomes

    class NotReady(StubAdapter):
        def check(self, options=None):
            return False, "missing shared binaries: diamond"

    monkeypatch.setattr(pipeline, "resolve", lambda _: [NotReady()])
    config = pipeline.RunConfig(fasta1=sp1.fasta, fasta2=sp2.fasta, outdir=tmp_path / "out2")

    with pytest.raises(pipeline.PipelineError, match="not ready"):
        pipeline.run_pipeline(config, log=lambda *_: None)
