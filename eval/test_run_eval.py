import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval  # noqa: E402
from run_eval import ac  # noqa: E402

BODY = ("We need to formulate the planner as a convex program so it stays tractable online. "
        "This recurring thread is for questions that might not warrant their own thread. "
        "I finally figured out the controller gains after a week of tuning. "
        "The formulation of the problem as a convex program is standard practice. "
        "El comportamiento del robot es errático cuando la batería baja. ") * 3
POST = {"id": "p1", "subreddit": "robotics", "title": "t", "url": "u", "body": BODY}
ITEMS = [
    {"word": "formulate", "cefr": "C1", "definition": "d", "academic_example": "a",
     "reddit_sentence": "We need to formulate the planner as a convex program so it stays tractable online."},
    {"word": "warrant", "cefr": "C1", "definition": "d", "academic_example": "a",
     "reddit_sentence": "This recurring thread is for questions that might not warrant their own thread."},
    {"word": "tractable", "cefr": "C2", "definition": "d", "academic_example": "a",
     "reddit_sentence": "The planner is tractable in every setting we tried."},          # invented sentence
    {"word": "planner", "cefr": "B1", "definition": "d", "academic_example": "a",
     "reddit_sentence": "We need to formulate the planner as a convex program so it stays tractable online."},
    {"word": "Formulate", "cefr": "C1", "definition": "d", "academic_example": "a",
     "reddit_sentence": "We need to formulate the planner as a convex program so it stays tractable online."},
    {"word": "formulation", "cefr": "C1", "definition": "d", "academic_example": "a",
     "reddit_sentence": "The formulation of the problem as a convex program is standard practice."},
    {"word": "figure out", "cefr": "B2", "definition": "d", "academic_example": "a",
     "reddit_sentence": "I finally figured out the controller gains after a week of tuning."},
]
ITEMS.insert(3, {"word": "errático", "cefr": "C1", "definition": "d", "academic_example": "a",
                 "reddit_sentence": "El comportamiento del robot es errático cuando la batería baja."})


def fake_llm(*args, usage=None, **kwargs):
    usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + 1000
    usage["completion_tokens"] = usage.get("completion_tokens", 0) + 200
    return ITEMS


class EvalTests(unittest.TestCase):
    def test_stage_counts_old_vs_new_without_touching_real_state(self):
        real_db, real_summary = ac.DB_PATH, ac.SUMMARY_PATH
        judge = lambda config, key, entries, usage=None: ["idiom" for _ in entries]
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(run_eval, "RESULTS", Path(tmp)), \
                patch.object(ac, "fetch_posts", return_value=[POST, POST]), \
                patch.object(ac, "call_llm", side_effect=fake_llm), \
                patch.object(run_eval.vf, "judge_phrases", side_effect=judge), \
                patch.object(run_eval, "PER_UNIT_CAP", 10), \
                patch.object(ac, "load_env", return_value={"DEEPSEEK_API_KEY": "k"}):
            rows, words, meta, compare = run_eval.main(["--subreddits", "robotics", "--posts", "20", "--papers", "--no-anki"])
            unit = rows[0]
            self.assertEqual(unit["texts"], 1)                  # duplicate post id counted once
            self.assertEqual(unit["candidates"], 8)
            self.assertEqual(unit["rejected_cefr"], 1)          # planner (B1)
            self.assertEqual(unit["rejected_not_verbatim"], 1)  # tractable
            self.assertEqual(unit["rejected_non_english"], 1)   # errático
            self.assertEqual(unit["rejected_basic_expression"], 1)  # figure out
            self.assertEqual(unit["duplicates_removed"], 1)     # Formulate
            self.assertEqual(unit["stem_duplicates_removed"], 1)  # formulation
            self.assertEqual(unit["final_words"], 2)
            self.assertEqual(unit["hallucinated_examples"], 1)
            self.assertEqual(unit["prompt_tokens"], 1000)
            new = sorted(w["word"] for w in words if w["pipeline"] == "new")
            old = sorted(w["word"] for w in words if w["pipeline"] == "old")
            self.assertEqual(new, ["formulate", "warrant"])
            # the old filter let through the Spanish word, the basic phrasal verb and the stem duplicate
            self.assertEqual(old, ["errático", "figure out", "formulate", "formulation", "warrant"])
            totals = {(r["source"], r["metric"]): r for r in compare if r["unit"] == "TOTAL"}
            self.assertEqual(totals[("reddit", "residual_basic_expression")]["old_filter_same_candidates"], 1)
            self.assertEqual(totals[("reddit", "residual_stem_duplicate")]["old_filter_same_candidates"], 1)
            self.assertEqual(totals[("reddit", "residual_non_english")]["old_filter_same_candidates"], 1)
            self.assertEqual(totals[("reddit", "residual_total")]["old_filter_same_candidates"], 3)
            self.assertEqual(totals[("reddit", "residual_total")]["new_filter"], 0)
            files = sorted(p.name.split("_")[0] for p in Path(tmp).iterdir())
            self.assertEqual(files, ["candidates", "compare", "meta", "summary", "words"])
            self.assertEqual(meta["paper_chunks_max"], 10)
            self.assertEqual(meta["anki"], "skipped (--no-anki)")
            self.assertNotEqual(ac.DB_PATH, real_db)           # redirected to the temp dir
            self.assertFalse(ac.DB_PATH.exists())              # and cleaned up
        self.assertNotEqual(real_summary, ac.SUMMARY_PATH)


if __name__ == "__main__":
    unittest.main()
