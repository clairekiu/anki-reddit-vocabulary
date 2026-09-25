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
        "This recurring thread is for questions that might not warrant their own thread. ") * 3
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
]


def fake_llm(*args, usage=None, **kwargs):
    usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + 1000
    usage["completion_tokens"] = usage.get("completion_tokens", 0) + 200
    return ITEMS


class EvalTests(unittest.TestCase):
    def test_stage_counts_and_outputs_without_touching_real_state(self):
        real_db, real_summary = ac.DB_PATH, ac.SUMMARY_PATH
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(run_eval, "RESULTS", Path(tmp)), \
                patch.object(ac, "fetch_posts", return_value=[POST, POST]), \
                patch.object(ac, "call_llm", side_effect=fake_llm), \
                patch.object(ac, "load_env", return_value={"DEEPSEEK_API_KEY": "k"}):
            rows, words, meta = run_eval.main(["--subreddits", "robotics", "--posts", "20", "--papers", "--no-anki"])
            unit = rows[0]
            self.assertEqual(unit["texts"], 1)                  # duplicate post id counted once
            self.assertEqual(unit["candidates"], 5)
            self.assertEqual(unit["rejected_cefr"], 1)          # planner (B1)
            self.assertEqual(unit["rejected_not_verbatim"], 1)  # tractable
            self.assertEqual(unit["passed_filter"], 3)
            self.assertEqual(unit["duplicates_removed"], 1)     # Formulate
            self.assertEqual(unit["final_words"], 2)
            self.assertEqual(unit["hallucinated_examples"], 1)
            self.assertAlmostEqual(unit["hallucination_rate"], 0.2)
            self.assertEqual(unit["prompt_tokens"], 1000)
            self.assertEqual(sorted(w["word"] for w in words), ["formulate", "warrant"])
            files = sorted(p.name.split("_")[0] for p in Path(tmp).iterdir())
            self.assertEqual(files, ["meta", "summary", "words"])
            with open(next(Path(tmp).glob("summary_*.csv")), encoding="utf-8") as handle:
                units = [r["unit"] for r in csv.DictReader(handle)]
            self.assertEqual(units, ["r/robotics", "TOTAL", "TOTAL", "TOTAL"])
            self.assertEqual(meta["anki"], "skipped (--no-anki)")
            self.assertNotEqual(ac.DB_PATH, real_db)           # redirected to the temp dir
            self.assertFalse(ac.DB_PATH.exists())              # and cleaned up
        self.assertNotEqual(real_summary, ac.SUMMARY_PATH)


if __name__ == "__main__":
    unittest.main()
