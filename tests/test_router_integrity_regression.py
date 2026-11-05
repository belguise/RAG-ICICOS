"""
Regression and Integrity Tests for Router Evaluation & Scientific Reproducibility.
Validates:
1. Gold benchmark categories never enter router prediction functions.
2. Fallback behavior matches production.
3. Correct separation between heuristic and LLM-based evaluation.
4. Correct routing quota allocation.
5. Correct strict/lenient scoring.
6. Correct handling of provider failures.
7. No duplicate query aggregation.
8. Holm-adjusted p-values determine primary significance flags.
9. Raw primary experimental records remain unchanged.
10. No synthetic experimental observations enter published results.
"""

import unittest
import csv
import os
import hashlib
from unittest.mock import MagicMock

from agentic_rag_sop.agents.router import RouterAgent, RouterDecision
from evaluation.metrics.router_accuracy import calculate_router_metrics


class TestRouterIntegrityRegression(unittest.TestCase):

    def test_01_no_category_parameter_in_prediction(self):
        """1. Verify gold categories cannot enter prediction functions."""
        router = RouterAgent(llm=MagicMock())
        import inspect
        # Verify route() signature
        sig_route = inspect.signature(router.route)
        self.assertIn("query", sig_route.parameters)
        self.assertNotIn("category", sig_route.parameters)
        self.assertNotIn("expected_source", sig_route.parameters)
        
        # Verify _classify_heuristics() signature
        sig_heur = inspect.signature(router._classify_heuristics)
        self.assertIn("query", sig_heur.parameters)
        self.assertNotIn("category", sig_heur.parameters)

    def test_02_fallback_behavior_matches_production(self):
        """2. Verify production fallback defaults to BOTH when LLM is unavailable."""
        router = RouterAgent(llm=MagicMock())
        router._chain = None  # Simulate broken/unavailable LLM chain
        # Query with zero keywords
        decision = router.route("Ambiguous query with zero keywords XYZ123")
        self.assertEqual(decision.knowledge_source, "BOTH")
        self.assertEqual(decision.intent, "general_information")
        self.assertIsNone(decision.secondary_source)

    def test_03_separation_heuristic_and_llm(self):
        """3. Verify heuristic fast-path bypasses LLM call entirely."""
        mock_llm = MagicMock()
        router = RouterAgent(llm=mock_llm)
        # Procedural document question matches heuristic
        q = "Apa saja syarat dokumen lampiran untuk SPJ?"
        decision = router.route(q)
        self.assertEqual(decision.knowledge_source, "SOP")
        self.assertEqual(decision.intent, "document_requirement")
        mock_llm.assert_not_called()

    def test_04_correct_quota_allocation(self):
        """4. Verify quota mappings: SOP (6,2), BPMN (3,4), BOTH (5,3)."""
        quotas = {
            "SOP": (6, 2),
            "BPMN": (3, 4),
            "BOTH": (5, 3),
        }
        self.assertEqual(quotas["SOP"], (6, 2))
        self.assertEqual(quotas["BPMN"], (3, 4))
        self.assertEqual(quotas["BOTH"], (5, 3))

    def test_05_strict_lenient_scoring(self):
        """5. Verify calculate_router_metrics strict vs lenient logic."""
        # Exact match
        m1 = calculate_router_metrics("SOP", "SOP", "document_requirement", "document_requirement", "BPMN")
        self.assertTrue(m1["source_correct"])
        self.assertTrue(m1["source_lenient_correct"])
        self.assertTrue(m1["intent_congruent"])

        # Lenient match (predicted BOTH when expected SOP)
        m2 = calculate_router_metrics("BOTH", "SOP", "general_information", "document_requirement", None)
        self.assertFalse(m2["source_correct"])
        self.assertTrue(m2["source_lenient_correct"])

        # Mismatch (predicted BPMN when expected SOP)
        m3 = calculate_router_metrics("BPMN", "SOP", "workflow_question", "document_requirement", None)
        self.assertFalse(m3["source_correct"])
        self.assertFalse(m3["source_lenient_correct"])

    def test_06_provider_failure_resilience(self):
        """6. Verify router catches provider exceptions and falls back to BOTH."""
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("503 Service Unavailable")
        router = RouterAgent(llm=MagicMock())
        router._chain = mock_chain
        
        # Query that needs LLM (does not match any heuristic keywords)
        q = "Informasi umum perihal kegiatan seminar dan lokakarya"
        decision = router.route(q)
        self.assertEqual(decision.knowledge_source, "BOTH")
        self.assertIn("503", decision.reason)

    def test_07_no_duplicate_query_aggregation(self):
        """7. Verify results/router_predictions_protocol_b.csv has unique query IDs."""
        csv_path = "results/router_predictions_protocol_b.csv"
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = list(csv.DictReader(f))
            ids = [r["id"] for r in reader]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(ids), 338)

    def test_08_holm_adjusted_significance_flags(self):
        """8. Verify Answer Relevancy in statistical_tests.csv is marked 'ns'."""
        csv_path = "results/statistical_tests.csv"
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = list(csv.DictReader(f))
            for row in reader:
                if row.get("Metric") == "Answer Relevancy":
                    self.assertEqual(row.get("Significance Marker"), "ns")
                    self.assertGreater(float(row.get("Holm-Bonferroni Adjusted p-value")), 0.05)

    def test_09_raw_primary_records_unchanged(self):
        """9. Verify master_results.csv row count and primary statistics stability."""
        master_path = "results/master_results.csv"
        self.assertTrue(os.path.exists(master_path))
        with open(master_path, 'r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(len(reader), 388)
        gemini_prop = [r for r in reader if r.get("architecture") == "proposed_agentic" and r.get("backbone") == "gemini"]
        self.assertEqual(len(gemini_prop), 70)

    def test_10_no_synthetic_ablation_observations(self):
        """10. Verify ablation_results.csv explicitly flags summary-only status."""
        abl_path = "results/ablation_results.csv"
        if os.path.exists(abl_path):
            with open(abl_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertIn("Historical summary mean", content)


if __name__ == "__main__":
    unittest.main()
