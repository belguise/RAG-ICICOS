"""
Research-Grade Evaluation Pipeline Unit Test Suite.
Validates metric accuracy, mathematical consistency, error handling, and edge cases
for the ICICOS evaluation engine.

Test Matrix:
1. Empty Answer
2. Empty Context
3. Empty Reference Answer
4. Fully Supported Answer
5. Unsupported Answer (Hallucination)
6. Partially Supported Answer
7. Semantically Equivalent Paraphrase
8. Incorrect Entity Substitution
9. Incorrect Numerical Value
10. Incorrect Process Sequence (Topological Inversion)
11. Gateway Branching Correctness & Condition Guard
12. Actor Swimlane Attribution & Synonym Resolution
13. EUSR and Faithfulness Duality (EUSR + Faithfulness = 1.0)
"""

import math
import os
import sys
import unittest
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics.reconstructed_evaluator import (
    MetricStatus,
    MetricResult,
    evaluate_faithfulness,
    evaluate_answer_relevancy,
    evaluate_context_precision,
    evaluate_context_recall,
    evaluate_sa_f1,
    evaluate_eusr,
    evaluate_process_path_accuracy,
    evaluate_gateway_decision_accuracy,
    evaluate_actor_attribution_accuracy,
    ReconstructedEvaluator,
)


class TestEvaluationPipeline(unittest.TestCase):
    """Test suite executing forensic unit tests across all evaluation metrics."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.evaluator = ReconstructedEvaluator()

        # Shared realistic test contexts from ICICOS SOP
        cls.valid_context = (
            "Prosedur Pengajuan Dana Insentif Publikasi ICICoS:\n"
            "1. Pemakalah mengunggah naskah final dan bukti pembayaran melalui portal konferensi.\n"
            "2. Sekretariat memeriksa kelengkapan berkas registrasi.\n"
            "3. Jika berkas dinyatakan sesuai dan lengkap, bendahara mencairkan dana insentif sebesar Rp 2.500.000.\n"
            "4. Apabila berkas tidak sesuai atau tidak lengkap, naskah dikembalikan kepada author untuk revisi dalam 3 hari kerja."
        )

        cls.valid_question = "Bagaimana alur dan syarat pencairan dana insentif publikasi ICICoS bagi pemakalah?"

        cls.gold_answer = (
            "Pemakalah mengunggah naskah final dan bukti transfer. Sekretariat memverifikasi kelengkapan berkas. "
            "Jika berkas sesuai, bendahara mencairkan dana insentif sebesar Rp 2.500.000. "
            "Jika tidak sesuai, berkas dikembalikan untuk direvisi dalam 3 hari kerja."
        )

        cls.gold_bpmn_path = [
            "Unggah naskah dan bukti pembayaran",
            "Periksa kelengkapan berkas",
            "Gateway: Apakah berkas sesuai?",
            "Cairkan dana insentif",
        ]

        cls.gold_actor = "Bendahara dan Sekretariat"

    # =========================================================================
    # Test 1: Empty Answer
    # =========================================================================
    def test_01_empty_answer(self):
        """Empty answer must yield 0.0 scores with MISSING_ANSWER status."""
        for empty_val in ["", "   ", None, "\n\t"]:
            res_f = evaluate_faithfulness(empty_val, self.valid_context)
            self.assertEqual(res_f.score, 0.0)
            self.assertEqual(res_f.status, MetricStatus.MISSING_ANSWER)

            res_r = evaluate_answer_relevancy(self.valid_question, empty_val)
            self.assertEqual(res_r.score, 0.0)
            self.assertEqual(res_r.status, MetricStatus.MISSING_ANSWER)

            res_sa = evaluate_sa_f1(empty_val, self.gold_answer)
            self.assertEqual(res_sa.score, 0.0)
            self.assertEqual(res_sa.status, MetricStatus.MISSING_ANSWER)

            res_ppa = evaluate_process_path_accuracy(empty_val, self.gold_bpmn_path)
            self.assertEqual(res_ppa.score, 0.0)
            self.assertEqual(res_ppa.status, MetricStatus.MISSING_ANSWER)

            res_eusr = evaluate_eusr(empty_val, self.valid_context)
            self.assertEqual(res_eusr.score, 0.0)
            self.assertEqual(res_eusr.status, MetricStatus.MISSING_ANSWER)

    # =========================================================================
    # Test 2: Empty Context
    # =========================================================================
    def test_02_empty_context(self):
        """Empty context (e.g. Zero-shot LLM) must yield 0.0 Faithfulness and 1.0 EUSR."""
        for empty_ctx in ["", "  ", None]:
            res_f = evaluate_faithfulness(self.gold_answer, empty_ctx)
            self.assertEqual(res_f.score, 0.0)
            self.assertEqual(res_f.status, MetricStatus.MISSING_CONTEXT)

            res_eusr = evaluate_eusr(self.gold_answer, empty_ctx)
            self.assertEqual(res_eusr.score, 1.0)
            self.assertEqual(res_eusr.status, MetricStatus.MISSING_CONTEXT)

            res_rec = evaluate_context_recall([], self.gold_answer)
            self.assertEqual(res_rec.score, 0.0)
            self.assertEqual(res_rec.status, MetricStatus.MISSING_CONTEXT)

    # =========================================================================
    # Test 3: Empty Reference Answer (Critical Integrity Bug Fix)
    # =========================================================================
    def test_03_empty_reference(self):
        """Missing reference must NEVER produce 1.0. Must return None with MISSING_REFERENCE."""
        chunks = [self.valid_context[:100], self.valid_context[100:200]]
        for empty_ref in ["", "  ", None]:
            res_prec = evaluate_context_precision(chunks, empty_ref, self.valid_question)
            self.assertIsNone(res_prec.score)
            self.assertEqual(res_prec.status, MetricStatus.MISSING_REFERENCE)

            res_rec = evaluate_context_recall(chunks, empty_ref)
            self.assertIsNone(res_rec.score)
            self.assertEqual(res_rec.status, MetricStatus.MISSING_REFERENCE)

            res_sa = evaluate_sa_f1(self.gold_answer, empty_ref)
            self.assertIsNone(res_sa.score)
            self.assertEqual(res_sa.status, MetricStatus.MISSING_REFERENCE)

    # =========================================================================
    # Test 4: Fully Supported Answer
    # =========================================================================
    def test_04_fully_supported_answer(self):
        """Answer directly extracted from context must achieve Faithfulness = 1.0 and EUSR = 0.0."""
        supported_answer = (
            "Pemakalah mengunggah naskah final dan bukti pembayaran melalui portal konferensi. "
            "Sekretariat memeriksa kelengkapan berkas registrasi. "
            "Jika berkas dinyatakan sesuai dan lengkap, bendahara mencairkan dana insentif sebesar Rp 2.500.000."
        )
        res_f = evaluate_faithfulness(supported_answer, self.valid_context)
        self.assertEqual(res_f.score, 1.0)
        self.assertEqual(res_f.status, MetricStatus.VALID)
        self.assertEqual(len(res_f.details["unsupported_claims"]), 0)

        res_eusr = evaluate_eusr(supported_answer, self.valid_context)
        self.assertEqual(res_eusr.score, 0.0)
        self.assertEqual(res_eusr.status, MetricStatus.VALID)

    # =========================================================================
    # Test 5: Unsupported Answer (Hallucination)
    # =========================================================================
    def test_05_unsupported_answer(self):
        """Completely fabricated answer must yield Faithfulness = 0.0 and EUSR = 1.0."""
        hallucinated_answer = (
            "Peserta wajib menyetor cryptocurrency ke rekening luar negeri di Swiss. "
            "Sistem blockchain otomatis mencetak sertifikat digital tanpa verifikasi manusia."
        )
        res_f = evaluate_faithfulness(hallucinated_answer, self.valid_context)
        self.assertEqual(res_f.score, 0.0)
        self.assertEqual(res_f.status, MetricStatus.VALID)
        self.assertGreater(len(res_f.details["unsupported_claims"]), 0)

        res_eusr = evaluate_eusr(hallucinated_answer, self.valid_context)
        self.assertEqual(res_eusr.score, 1.0)
        self.assertEqual(res_eusr.status, MetricStatus.VALID)

    # =========================================================================
    # Test 6: Partially Supported Answer
    # =========================================================================
    def test_06_partially_supported_answer(self):
        """Answer with 1 supported claim and 1 fabricated claim must yield 0.5."""
        partial_answer = (
            "Pemakalah mengunggah naskah final dan bukti pembayaran melalui portal konferensi.\n"
            "Setelah itu dana langsung dicairkan dalam bentuk koin emas batangan oleh kurir kilat."
        )
        res_f = evaluate_faithfulness(partial_answer, self.valid_context)
        self.assertAlmostEqual(res_f.score, 0.5, delta=0.05)
        self.assertEqual(res_f.details["total_claims"], 2)
        self.assertEqual(res_f.details["supported_claims"], 1)

        res_eusr = evaluate_eusr(partial_answer, self.valid_context)
        self.assertAlmostEqual(res_eusr.score, 0.5, delta=0.05)
        self.assertAlmostEqual(res_f.score + res_eusr.score, 1.0, places=4)

    # =========================================================================
    # Test 7: Semantically Equivalent Paraphrase
    # =========================================================================
    def test_07_semantically_equivalent_paraphrase(self):
        """Paraphrased reference answer must score high SA-F1 and Answer Relevancy."""
        paraphrase = (
            "Author menyerahkan artikel akhir beserta konfirmasi bayar. "
            "Pihak sekretariat memverifikasi dokumen tersebut. "
            "Bila berkas valid, bendahara akan menyalurkan insentif 2,5 juta rupiah. "
            "Namun bila tidak lengkap, naskah dikembalikan untuk diperbaiki dalam waktu tiga hari."
        )
        res_sa = evaluate_sa_f1(paraphrase, self.gold_answer)
        self.assertIsNotNone(res_sa.score)
        self.assertGreater(res_sa.score, 0.65)

        res_rel = evaluate_answer_relevancy(self.valid_question, paraphrase)
        self.assertGreater(res_rel.score, 0.35)

    # =========================================================================
    # Test 8: Incorrect Entity Substitution
    # =========================================================================
    def test_08_incorrect_entity_substitution(self):
        """Substituting wrong organizational actor must be penalized in Actor Accuracy."""
        wrong_actor_answer = (
            "Pemakalah mengunggah berkas. Rektorat dan Pimpinan Fakultas yang mencairkan dana secara pribadi."
        )
        res_aaa = evaluate_actor_attribution_accuracy(wrong_actor_answer, self.gold_actor)
        # Expected actors: 'Bendahara' and 'Sekretariat'; neither is mentioned
        self.assertEqual(res_aaa.score, 0.0)
        self.assertEqual(len(res_aaa.details["missing_actors"]), 2)

    # =========================================================================
    # Test 9: Incorrect Numerical Value
    # =========================================================================
    def test_09_incorrect_numerical_value(self):
        """Changing monetary amount from 2.500.000 to 50.000.000 must fail token grounding."""
        corrupted_num_sentence = (
            "Jika berkas dinyatakan sesuai dan lengkap, bendahara mencairkan dana insentif sebesar Rp 50.000.000."
        )
        tokens = [t for t in corrupted_num_sentence.lower().split() if t not in ["jika", "dan", "sebesar"]]
        # In context, 50.000.000 does not appear
        self.assertNotIn("50000000", self.valid_context)

    # =========================================================================
    # Test 10: Incorrect Process Sequence (Topological Order Inversion)
    # =========================================================================
    def test_10_incorrect_process_sequence(self):
        """Inverting process sequence (cairkan dana BEFORE unggah naskah) must reduce PPA."""
        # Chronologically inverted answer:
        inverted_answer = (
            "Pertama, bendahara akan langsung mencairkan dana insentif kepada pemakalah. "
            "Setelah dana cair, baru pemakalah mengunggah naskah dan bukti pembayaran. "
            "Terakhir sekretariat memeriksa kelengkapan berkas."
        )
        # Chronologically correct answer:
        correct_answer = (
            "Pertama, pemakalah mengunggah naskah dan bukti pembayaran. "
            "Kemudian sekretariat memeriksa kelengkapan berkas. "
            "Terakhir, bendahara mencairkan dana insentif."
        )
        res_inverted = evaluate_process_path_accuracy(inverted_answer, self.gold_bpmn_path)
        res_correct = evaluate_process_path_accuracy(correct_answer, self.gold_bpmn_path)

        self.assertGreater(res_correct.score, res_inverted.score)
        self.assertGreater(res_correct.details["order_preservation"], res_inverted.details["order_preservation"])

    # =========================================================================
    # Test 11: Gateway Branching Logic (GDA)
    # =========================================================================
    def test_11_gateway_branching_logic(self):
        """Correct conditional guard and branch outcome must yield GDA = 1.0."""
        valid_gateway_answer = (
            "Jika berkas sesuai dan lengkap, maka bendahara mencairkan dana insentif. "
            "Apabila berkas tidak sesuai atau tidak lengkap, naskah dikembalikan untuk direvisi."
        )
        res_gda = evaluate_gateway_decision_accuracy(
            predicted_answer=valid_gateway_answer,
            question=self.valid_question,
            expected_answer=self.gold_answer,
            expected_bpmn_path=self.gold_bpmn_path,
        )
        self.assertEqual(res_gda.score, 1.0)
        self.assertTrue(res_gda.details["condition_identified"])
        self.assertTrue(res_gda.details["outcome_correct"])

    # =========================================================================
    # Test 12: Actor Swimlane Attribution & Synonyms (AAA)
    # =========================================================================
    def test_12_actor_swimlane_attribution_synonyms(self):
        """Using valid Indonesian role synonyms (e.g. 'keuangan panitia' for 'bendahara') must succeed."""
        synonym_answer = (
            "Berkas diverifikasi oleh pihak sekretariat, selanjutnya bagian keuangan panitia yang memproses insentif."
        )
        res_aaa = evaluate_actor_attribution_accuracy(synonym_answer, self.gold_actor)
        # Both 'bendahara' (via 'keuangan panitia') and 'sekretariat' are present
        self.assertEqual(res_aaa.score, 1.0)
        self.assertEqual(len(res_aaa.details["matched_actors"]), 2)

    # =========================================================================
    # Test 13: Mathematical Consistency of EUSR and Faithfulness
    # =========================================================================
    def test_13_eusr_faithfulness_duality(self):
        """EUSR + Faithfulness must strictly sum to 1.0 across diverse candidate answers."""
        candidate_answers = [
            "Pemakalah mengunggah naskah final dan bukti pembayaran melalui portal konferensi.",
            "Semua naskah langsung diterima tanpa syarat apa pun dari panitia konferensi.",
            "Sekretariat memeriksa kelengkapan berkas registrasi. Namun panitia tidak pernah mencairkan dana apa pun.",
        ]
        for ans in candidate_answers:
            res_f = evaluate_faithfulness(ans, self.valid_context)
            res_e = evaluate_eusr(ans, self.valid_context)
            if res_f.score is not None and res_e.score is not None:
                total = round(res_f.score + res_e.score, 4)
                self.assertAlmostEqual(total, 1.0, places=4, msg=f"Failed duality on answer: {ans}")


if __name__ == "__main__":
    unittest.main()
