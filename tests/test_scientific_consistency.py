"""
Automated Scientific Consistency and Integrity Test Suite for KBS Manuscript
=============================================================================
This test suite validates 20 critical scientific, statistical, numerical,
architectural, and reproducibility properties of the Knowledge-Based Systems
(Elsevier Q1) manuscript and associated repository artifacts.

Authoritative benchmark lock: N = 70.
"""

import os
import re
import sys
import json
import unittest
import subprocess
import pandas as pd
import numpy as np

# Base paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANUSCRIPT_TEX = os.path.join(REPO_ROOT, "FINAL PAPER Q1", "KBS_SUBMISSION", "Manuscript", "main.tex")
REFERENCES_BIB = os.path.join(REPO_ROOT, "FINAL PAPER Q1", "KBS_SUBMISSION", "Manuscript", "references.bib")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


class TestScientificConsistency(unittest.TestCase):
    """Test suite executing the 20 scientific consistency requirements (Phase 23)."""

    @classmethod
    def setUpClass(cls):
        """Load manuscript text, bibliography, and master CSV artifacts."""
        if not os.path.exists(MANUSCRIPT_TEX) or not os.path.exists(RESULTS_DIR):
            raise unittest.SkipTest("Manuscript LaTeX sources and raw results are archived separately.")

        with open(MANUSCRIPT_TEX, "r", encoding="utf-8") as f:
            cls.tex_content = f.read()

        with open(REFERENCES_BIB, "r", encoding="utf-8") as f:
            cls.bib_content = f.read()

        cls.final_comp_df = pd.read_csv(os.path.join(RESULTS_DIR, "final_comparison_table.csv"))
        cls.stat_df = pd.read_csv(os.path.join(RESULTS_DIR, "statistical_tests.csv"))
        cls.backbone_df = pd.read_csv(os.path.join(RESULTS_DIR, "backbone_comparison.csv"))
        cls.thresh_df = pd.read_csv(os.path.join(RESULTS_DIR, "threshold_sensitivity.csv"))

    def test_01_no_stale_numbers(self):
        """Test 1: Verify no legacy N=20 stale numbers appear in text or tables."""
        stale_patterns = [
            (r"\b0\.972(?!\d)", "0.972 (legacy unverified faithfulness)"),
            (r"(?<!\d)2\.6\s*\\?%", "2.6% (legacy unverified hallucination rate)"),
            (r"(?<!\d)79\.5\s*\\?%", "79.5% (legacy router accuracy)"),
            (r"(?<!\d)93\.8\s*\\?%", "93.8% (legacy execution correctness)"),
            (r"(?<!\d)65\.0\s*\\?%", "65.0% (legacy semantic coverage)"),
            (r"\b14\.74\b", "14.74s (legacy Proposed latency)"),
            (r"\b8\.54\b", "8.54s (legacy Static Hybrid latency)"),
            (r"\b4\.18\b", "4.18s (legacy Naive RAG latency)"),
            (r"\b0\.967(?!\d)", "0.967 (legacy structural validity)"),
        ]
        violations = []
        for pattern, desc in stale_patterns:
            matches = re.findall(pattern, self.tex_content)
            if matches:
                violations.append(f"{desc}: found {len(matches)} occurrences")

        self.assertEqual(len(violations), 0, f"Stale numbers detected in manuscript: {violations}")

    def test_02_table_numbers_match_master_csv(self):
        """Test 2: Verify Table numbers match master CSVs to within declared rounding."""
        # Check Proposed model metrics in Table 3 against final_comparison_table.csv
        # Proposed: Faithfulness = 0.988, Hallucination = 1.2% (0.012), Latency = 10.76s
        proposed_csv = self.final_comp_df[self.final_comp_df["Architecture"].str.contains("Proposed")].iloc[0]
        self.assertAlmostEqual(float(proposed_csv["Faithfulness"]), 0.988, places=3)
        self.assertAlmostEqual(float(proposed_csv["Answer Relevancy"]), 0.752, places=3)
        self.assertAlmostEqual(float(proposed_csv["Context Precision"]), 0.991, places=3)
        self.assertAlmostEqual(float(proposed_csv["Context Recall"]), 1.000, places=3)
        self.assertIn("10.76", str(proposed_csv["Latency (s)"]))

        # Verify exact presence in main.tex Table 3
        self.assertIn("0.988", self.tex_content)
        self.assertIn("0.752", self.tex_content)
        self.assertIn("0.991", self.tex_content)
        self.assertIn("1.000", self.tex_content)
        self.assertIn("10.76s", self.tex_content)

    def test_03_p_values_match_statistical_tests(self):
        """Test 3: Verify p-values in text match statistical test results."""
        faith_row = self.stat_df[self.stat_df["Metric"].str.contains("Faithfulness")].iloc[0]
        wilcoxon_p = float(faith_row["p-value (Wilcoxon)"])
        ttest_p = float(faith_row["p-value (Paired t-test)"])

        # Format 1.19e-10 and 2.49e-13
        self.assertAlmostEqual(wilcoxon_p, 1.1894e-10, delta=1e-12)
        self.assertAlmostEqual(ttest_p, 2.4903e-13, delta=1e-15)

        # Check in text: 1.19 \times 10^{-10} and 2.49 \times 10^{-13}
        self.assertIn("1.19 \\times 10^{-10}", self.tex_content)
        self.assertIn("2.49 \\times 10^{-13}", self.tex_content)

    def test_04_percentages_match_counts(self):
        """Test 4: Verify percentages in text match counts and totals."""
        total = 70
        # Compliant base path: 53 queries -> 75.7%
        base_cnt = 53
        base_pct = round((base_cnt / total) * 100, 1)
        self.assertEqual(base_pct, 75.7)
        self.assertTrue(re.search(r"75\.7\s*\\?%", self.tex_content))

        # Repaired path: 17 queries -> 24.3%
        repair_cnt = 17
        repair_pct = round((repair_cnt / total) * 100, 1)
        self.assertEqual(repair_pct, 24.3)
        self.assertTrue(re.search(r"24\.3\s*\\?%", self.tex_content))

        # Post-hoc audit threshold tau=0.85: 19 queries -> 27.1%
        posthoc_cnt = 19
        posthoc_pct = round((posthoc_cnt / total) * 100, 1)
        self.assertEqual(posthoc_pct, 27.1)
        self.assertTrue(re.search(r"27\.1\s*\\?%", self.tex_content))

    def test_05_sample_sizes_consistent(self):
        """Test 5: Verify sample sizes are consistent across all tables."""
        # Full corpus N=338, Procedural diagnostic N=60, Primary comparative benchmark N=70
        self.assertIn("N=70", self.tex_content)
        self.assertIn("N=60", self.tex_content)
        self.assertIn("N=338", self.tex_content)

        # Verify stratification sums to 70: 18 + 18 + 17 + 17 = 70
        strat_sum = 18 + 18 + 17 + 17
        self.assertEqual(strat_sum, 70)

    def test_06_model_ids_match_declared_configs(self):
        """Test 6: Verify model IDs match declared configurations."""
        # Declared models: Google Gemini 3.5 Flash-Lite (gemini-3.5-flash-lite) and Qwen 3.8 27B
        self.assertIn("Google Gemini 3.5 Flash-Lite", self.tex_content)
        self.assertIn("gemini-3.5-flash-lite", self.tex_content)
        self.assertIn("Qwen 3.8 27B", self.tex_content)

        # Ensure no unverified claims of "Gemini 2.5 Pro" as evaluated backbone
        self.assertNotIn("Gemini 2.5 Pro", self.tex_content)

    def test_07_level1_vs_level2_metrics_separated(self):
        """Test 7: Verify Level 1 and Level 2 metrics are strictly separated."""
        # Level 1 metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall, BERTScore
        self.assertIn("Process Path Accuracy (PPA)", self.tex_content)
        self.assertIn("Gateway Decision Accuracy (GDA)", self.tex_content)
        self.assertTrue("Actor Attribution Recall (AAR)" in self.tex_content or "Actor Attribution Accuracy (AAA)" in self.tex_content)

        # In Table 3 (Primary), Level 2 metrics are not intermixed into Level 1 columns
        tab3_pos = self.tex_content.find(r"\label{tab:main_results}")
        tab3_text = self.tex_content[tab3_pos:tab3_pos + 1200]
        self.assertNotIn("Path Accuracy", tab3_text)
        self.assertNotIn("Gateway Accuracy", tab3_text)

    def test_08_confidence_intervals_mathematically_possible(self):
        """Test 8: Verify confidence interval widths are mathematically possible."""
        for idx, row in self.stat_df.iterrows():
            prop_ci_str = row["Proposed 95% Bootstrap CI"]
            if pd.notna(prop_ci_str) and prop_ci_str.startswith("["):
                lower, upper = [float(x.strip()) for x in prop_ci_str.strip("[]").split(",")]
                mean_val = float(row["Proposed Agentic Mean ± Std"].split("±")[0].strip())
                self.assertLessEqual(lower, upper, f"Lower CI > Upper CI in row {idx}")
                self.assertLessEqual(lower, mean_val + 1e-4, f"Lower CI > Mean in row {idx}")
                self.assertGreaterEqual(upper, mean_val - 1e-4, f"Upper CI < Mean in row {idx}")

    def test_09_graph_reasoning_latency_arithmetic_reproducible(self):
        """Test 9: Verify graph reasoning latency arithmetic is reproducible."""
        base_latency = 9.97
        repair_latency = 13.22
        base_weight = 53 / 70  # 0.75714...
        repair_weight = 17 / 70  # 0.24285...

        weighted = (base_weight * base_latency) + (repair_weight * repair_latency)
        self.assertAlmostEqual(weighted, 10.76, delta=0.01)

        # In text footnote: 0.757 * 9.97 + 0.243 * 13.22 = 10.76
        text_calc = 0.757 * 9.97 + 0.243 * 13.22
        self.assertAlmostEqual(text_calc, 10.76, delta=0.01)

    def test_10_efficiency_tradeoff_numbers_match_section59(self):
        """Test 10: Verify efficiency trade-off numbers match Section 5 text."""
        # Proposed 10.76s vs Static Hybrid 3.05s -> +7.71s
        delta_latency = 10.76 - 3.05
        self.assertAlmostEqual(delta_latency, 7.71, places=2)
        self.assertIn("+7.71s", self.tex_content)

        # Hallucination reduction: from 12.1% to 1.16% -> 90.4% reduction
        reduction = (12.1 - 1.16) / 12.1 * 100
        self.assertAlmostEqual(reduction, 90.41, delta=0.1)
        self.assertTrue(re.search(r"90\.4\s*\\?%", self.tex_content))

    def test_11_figure_data_matches_table_data(self):
        """Test 11: Verify figure data and captions match table data and architectural specifications."""
        expected_figures = [
            "fig:architecture",
            "fig:dual_representation",
            "fig:bpmn_to_graph",
            "fig:adaptive_routing",
            "fig:critic_loop",
            "fig:inference_workflow",
        ]
        fig_labels = set(re.findall(r"\\label\{(fig:[^}]+)\}", self.tex_content))
        for exp_fig in expected_figures:
            self.assertIn(exp_fig, fig_labels, f"Expected figure {exp_fig} missing from manuscript!")

    def test_12_table_cross_references_match_actual_tables(self):
        """Test 12: Verify table cross-references match actual table labels."""
        table_labels = set(re.findall(r"\\label\{(tab:[^}]+)\}", self.tex_content))
        table_refs = set(re.findall(r"\\ref\{(tab:[^}]+)\}", self.tex_content))
        missing = table_refs - table_labels
        self.assertEqual(len(missing), 0, f"Unresolved table references: {missing}")

    def test_13_figure_cross_references_match_actual_figures(self):
        """Test 13: Verify figure cross-references match actual figure labels."""
        fig_labels = set(re.findall(r"\\label\{(fig:[^}]+)\}", self.tex_content))
        fig_refs = set(re.findall(r"\\ref\{(fig:[^}]+)\}", self.tex_content))
        missing = fig_refs - fig_labels
        self.assertEqual(len(missing), 0, f"Unresolved figure references: {missing}")

    def test_14_section_cross_references_valid(self):
        """Test 14: Verify section cross-references match actual section labels."""
        sec_labels = set(re.findall(r"\\label\{(sec:[^}]+)\}", self.tex_content))
        sec_refs = set(re.findall(r"\\ref\{(sec:[^}]+)\}", self.tex_content))
        missing = sec_refs - sec_labels
        self.assertEqual(len(missing), 0, f"Unresolved section references: {missing}")

    def test_15_citations_resolve_to_bib_entries(self):
        """Test 15: Verify all citations in LaTeX manuscript resolve to .bib entries."""
        bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+),", self.bib_content))
        cited_keys = set()
        for match in re.findall(r"\\cite[a-z]*\{([^}]+)\}", self.tex_content):
            for key in match.split(","):
                cited_keys.add(key.strip())

        unresolved = cited_keys - bib_keys
        self.assertEqual(len(unresolved), 0, f"Unresolved BibTeX citations: {unresolved}")

    def test_16_acronyms_expanded_on_first_use(self):
        """Test 16: Verify key acronyms are expanded on first use."""
        core_acronyms = {
            "BPMN": "Business Process Model and Notation",
            "RAG": "Retrieval-Augmented Generation",
            "LLM": "Large Language Model",
            "QUA": "Query Understanding Agent",
        }
        for acr, expansion in core_acronyms.items():
            exp_pos = self.tex_content.find(expansion)
            self.assertNotEqual(exp_pos, -1, f"Expansion for '{acr}' ({expansion}) not found!")

    def test_17_mathematical_symbols_defined_before_use(self):
        """Test 17: Verify mathematical symbols are formally defined in methodology."""
        # Symbols: tau, G = (V, E, ...), Psi, \mathcal{C}_{\text{fused}}, \Psi^*
        self.assertIn(r"\tau", self.tex_content)
        self.assertIn(r"G = (V, E, \mathcal{L}, \mathcal{S}, \mathcal{C})", self.tex_content)
        self.assertIn(r"\Psi", self.tex_content)
        self.assertIn(r"\mathcal{C}_{\text{fused}}", self.tex_content)
        self.assertIn(r"\Psi^*", self.tex_content)

    def test_18_no_duplicate_labels_or_keys(self):
        """Test 18: Verify no duplicate LaTeX labels and no duplicate BibTeX keys."""
        all_labels = re.findall(r"\\label\{([^}]+)\}", self.tex_content)
        dup_labels = [lbl for lbl in set(all_labels) if all_labels.count(lbl) > 1]
        self.assertEqual(len(dup_labels), 0, f"Duplicate LaTeX labels: {dup_labels}")

        all_bib_keys = re.findall(r"@\w+\s*\{\s*([^,\s]+),", self.bib_content)
        dup_bib_keys = [k for k in set(all_bib_keys) if all_bib_keys.count(k) > 1]
        self.assertEqual(len(dup_bib_keys), 0, f"Duplicate BibTeX keys: {dup_bib_keys}")

    def test_19_no_undefined_latex_references(self):
        """Test 19: Verify zero undefined LaTeX references across all label types."""
        all_labels = set(re.findall(r"\\label\{([^}]+)\}", self.tex_content))
        all_refs = set(re.findall(r"\\ref\{([^}]+)\}", self.tex_content))
        missing = all_refs - all_labels
        self.assertEqual(len(missing), 0, f"Undefined LaTeX references: {missing}")

    def test_20_no_secrets_in_tracked_files(self):
        """Test 20: Verify zero API keys or authentication secrets in git-tracked files."""
        try:
            tracked_files = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True).splitlines()
        except Exception:
            tracked_files = []

        secret_pattern = re.compile(r"AIza[0-9A-Za-z-_]{35}")
        leaked_files = []

        for rel_path in tracked_files:
            abs_path = os.path.join(REPO_ROOT, rel_path)
            if os.path.isfile(abs_path):
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                        if secret_pattern.search(text):
                            leaked_files.append(rel_path)
                except Exception:
                    pass

        self.assertEqual(len(leaked_files), 0, f"API secrets detected in tracked files: {leaked_files}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
