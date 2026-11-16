"""Selection invariants; no models, numpy or API access needed."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.budget_context import BudgetContextBuilder
from src.context_builder import SimpleContextBuilder


class Records:
    records = {"long": "x" * 12, "a": "a" * 4, "b": "b" * 6}

    def record_text(self, rid):
        return self.records[rid]


class BudgetTests(unittest.TestCase):
    def test_skip_oversized_and_fill_exact_budget(self):
        built = BudgetContextBuilder(10).build("question", ["long", "a", "b"], Records())
        self.assertEqual(built["selected_record_ids"], ["a", "b"])
        self.assertEqual(built["skipped_record_ids"], ["long"])
        self.assertEqual(built["evidence_chars"], 10)
        self.assertEqual(built["prompt"], SimpleContextBuilder().build("question", ["a", "b"], Records())["prompt"])

    def test_no_document_fits(self):
        built = BudgetContextBuilder(1).build("question", ["a"], Records())
        self.assertEqual(built["contexts"], [])
        self.assertEqual(built["evidence_chars"], 0)

    def test_large_budget_preserves_baseline(self):
        ids = ["long", "a", "b"]
        built = BudgetContextBuilder(100).build("question", ids, Records())
        self.assertEqual(built["selected_record_ids"], ids)
        self.assertEqual(built["prompt"], SimpleContextBuilder().build("question", ids, Records())["prompt"])

    def test_invalid_budget(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                BudgetContextBuilder(value)


if __name__ == "__main__":
    unittest.main()
