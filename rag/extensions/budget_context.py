"""Whole-document selection under an evidence-character budget."""
from src.context_builder import SimpleContextBuilder


class BudgetContextBuilder:
    def __init__(self, budget_chars):
        if type(budget_chars) is not int or budget_chars < 1:
            raise ValueError("budget_chars must be a positive integer")
        self.budget_chars = budget_chars

    def build(self, transcript, record_ids, data):
        selected, skipped = [], []
        used = 0
        for rid in record_ids:
            size = len(data.record_text(rid))
            if used + size <= self.budget_chars:
                selected.append(rid)
                used += size
            else:
                skipped.append(rid)
        built = SimpleContextBuilder().build(transcript, selected, data)
        return dict(built, selected_record_ids=selected,
                    skipped_record_ids=skipped, evidence_chars=used,
                    budget_chars=self.budget_chars,
                    prompt_chars=len(built["prompt"]))
