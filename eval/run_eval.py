#!/usr/bin/env python3
"""Evaluation run for the anki-crawling pipeline: stage-by-stage counts for Reddit and papers.

Never touches the real word DB or real decks: uses a temporary DB and the English::eval-test deck
(or --no-anki), and removes both at the end.
"""
import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
import urllib.error
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import anki_crawler as ac  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
EVAL_DECK = "English::eval-test"
SUBREDDITS = ["robotics", "MachineLearning", "ControlTheory", "ROS", "cpp", "compsci", "math"]
PAPERS = [  # robotics 2, control 1, ML 2
    "https://arxiv.org/abs/2303.04137",  # Diffusion Policy
    "https://arxiv.org/abs/2307.15818",  # RT-2
    "https://arxiv.org/abs/1903.11199",  # Control Barrier Functions: Theory and Applications
    "https://arxiv.org/abs/1706.03762",  # Attention Is All You Need
    "https://arxiv.org/abs/2106.09685",  # LoRA
]
LEVELS = ("B1", "B2", "C1", "C2")
LLM_COUNT = 8        # items requested per LLM call (same cap the daily run uses)
PER_UNIT_CAP = 4     # max words kept per post / paper chunk (same as production)
PAPER_CHUNKS = 10    # chunks visited per paper (same as run_paper)


def level_of(item):
    level = str(item.get("cefr", "")).upper()[:2] if isinstance(item, dict) else ""
    return level if level in LEVELS else "other"


class Unit:
    """Counters for one run unit (a subreddit or a paper)."""

    def __init__(self, source, name):
        self.source, self.name = source, name
        self.texts = self.calls = self.llm_errors = 0
        self.candidates = []
        self.reasons = Counter()
        self.not_verbatim_any = self.schema_ok = 0
        self.duplicates = self.capped = 0
        self.final = []
        self.usage = {}
        self.seconds = 0.0
        self.note = ""

    def judge(self, items, body, known, limit):
        """Run production filters on one LLM reply; return accepted items."""
        kept = []
        for item in items:
            self.candidates.append(item)
            if isinstance(item, dict) and isinstance(item.get("reddit_sentence"), str) and item["reddit_sentence"].strip():
                self.schema_ok += 1
                if not ac.sentence_in_source(item, body):  # hallucination rate, independent of other filters
                    self.not_verbatim_any += 1
            reason = ac.reject_reason(item, body)
            if reason:
                self.reasons[reason] += 1
                continue
            key = ac.word_key(item["word"])
            if not key or key in known:
                self.duplicates += 1
                continue
            if len(kept) >= min(PER_UNIT_CAP, limit):
                self.capped += 1
                continue
            known.add(key)
            kept.append((key, item))
        return kept

    def row(self, run_id, prices):
        before = Counter(level_of(i) for i in self.candidates)
        after = Counter(level_of(i) for i in self.final)
        total = len(self.candidates)
        passed = total - sum(self.reasons.values())
        cost = (self.usage.get("prompt_tokens", 0) * prices[0] + self.usage.get("completion_tokens", 0) * prices[1]) / 1e6
        return {
            "run_id": run_id, "source": self.source, "unit": self.name, "texts": self.texts,
            "llm_calls": self.calls, "llm_errors": self.llm_errors, "candidates": total,
            **{f"before_{k}": before.get(k, 0) for k in LEVELS + ("other",)},
            "rejected_schema": self.reasons["schema"], "rejected_cefr": self.reasons["cefr"],
            "rejected_not_verbatim": self.reasons["not_verbatim"], "rejected_word_missing": self.reasons["word_missing"],
            "passed_filter": passed, "pass_rate": ratio(passed, total),
            "hallucinated_examples": self.not_verbatim_any, "hallucination_rate": ratio(self.not_verbatim_any, self.schema_ok),
            "duplicates_removed": self.duplicates, "cap_removed": self.capped, "final_words": len(self.final),
            **{f"after_{k}": after.get(k, 0) for k in LEVELS + ("other",)},
            "seconds": round(self.seconds, 1), "prompt_tokens": self.usage.get("prompt_tokens", 0),
            "completion_tokens": self.usage.get("completion_tokens", 0), "est_cost_usd": round(cost, 4),
            "note": self.note,
        }


def ratio(part, whole):
    return round(part / whole, 4) if whole else 0.0


def total_row(rows, run_id, source):
    rows = [r for r in rows if source == "all" or r["source"] == source]
    out = {k: sum(r[k] for r in rows) for k in rows[0] if isinstance(rows[0][k], (int, float))} if rows else {}
    out.update(run_id=run_id, source=source, unit="TOTAL", note="")
    out["pass_rate"] = ratio(out.get("passed_filter", 0), out.get("candidates", 0))
    schema_ok = sum(r["candidates"] - r["rejected_schema"] for r in rows)
    out["hallucination_rate"] = ratio(out.get("hallucinated_examples", 0), schema_ok)
    out["est_cost_usd"] = round(out.get("est_cost_usd", 0), 4)
    out["seconds"] = round(out.get("seconds", 0), 1)
    return out


def store(db, unit, kept, post, day, deck):
    for key, item in kept:
        ac.insert_card(db, key, item, post, day, unit.source, deck)
        unit.final.append(item)
    db.commit()


def eval_subreddit(db, config, api_key, subreddit, posts_wanted, known, day):
    unit = Unit("reddit", f"r/{subreddit}")
    started = time.time()
    fetch = {**config, "subreddits": [subreddit], "listing": "new", "fetchLimit": 100,
             "minPostChars": 0, "maxPostChars": 10 ** 9}
    raw = ac.fetch_posts(fetch)  # 429 retry + spacing live in fetch_posts
    seen, posts = set(), []
    for post in raw:  # newest first, distinct, same length filter as the daily run
        if post["id"] in seen or not config["minPostChars"] <= len(post["body"]) <= config["maxPostChars"]:
            continue
        seen.add(post["id"])
        posts.append(post)
    posts = posts[:posts_wanted]
    unit.note = f"fetched {len(raw)} / eligible {len(seen)}"
    for post in posts:
        unit.texts += 1
        try:
            unit.calls += 1
            items = ac.call_llm(config, api_key, post["title"], post["body"], sorted(known)[-300:], LLM_COUNT,
                                usage=unit.usage)
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
            unit.llm_errors += 1
            continue
        store(db, unit, unit.judge(items, post["body"], known, LLM_COUNT), post, day, EVAL_DECK)
    unit.seconds = time.time() - started
    return unit


def eval_paper(db, config, api_key, url, max_words, known, day):
    unit = Unit("paper", url)
    started = time.time()
    try:
        title, link, body = ac.paper_text(url)
    except Exception as error:  # recorded, the run continues with other papers
        unit.note = f"extract failed: {type(error).__name__}"
        unit.seconds = time.time() - started
        return unit
    unit.name = title[:120]
    parts = ac.chunks(body)
    unit.note = f"{url} · {len(body)} chars · {len(parts)} chunks"
    post = {"subreddit": "paper", "title": title[:200], "url": link}
    for index in ac.spread_order(len(parts))[:PAPER_CHUNKS]:
        if len(unit.final) >= max_words:
            break
        unit.texts += 1
        want = min(LLM_COUNT, max_words - len(unit.final) + 3)
        try:
            unit.calls += 1
            items = ac.call_llm(config, api_key, title, parts[index], sorted(known)[-300:], want,
                                kind="research paper excerpt", usage=unit.usage)
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
            unit.llm_errors += 1
            continue
        store(db, unit, unit.judge(items, parts[index], known, max_words - len(unit.final)), post, day, EVAL_DECK)
    unit.seconds = time.time() - started
    return unit


def upload_and_cleanup(db, config):
    """Push to the eval deck only (proves the Anki path), then delete that deck."""
    anki = ac.Anki(config["ankiUrl"])
    if not ac.ensure_anki_running(anki):
        return "anki unavailable"
    if EVAL_DECK in anki.call("deckNames"):
        raise RuntimeError(f"{EVAL_DECK} already exists; remove it before running eval")
    rows = db.execute("SELECT * FROM cards WHERE anki_note_id IS NULL").fetchall()
    anki.call("createDeck", deck=EVAL_DECK)
    added = 0
    try:
        for row in rows:
            note = {"deckName": EVAL_DECK, "modelName": config["modelName"],
                    "fields": ac.note_fields(row, ac.html.escape(row["post_title"])),
                    "tags": ["eval_test"], "options": {"allowDuplicate": True}}
            anki.call("addNote", note=note)
            added += 1
    finally:
        anki.call("deleteDecks", decks=[EVAL_DECK], cardsToo=True)
    left = EVAL_DECK in anki.call("deckNames")
    return f"uploaded {added}/{len(rows)} to {EVAL_DECK}, deck deleted={not left}"


def print_table(rows):
    cols = [("source", 7), ("unit", 34), ("texts", 5), ("candidates", 10), ("pass_rate", 9),
            ("hallucination_rate", 10), ("duplicates_removed", 5), ("final_words", 6), ("seconds", 7), ("est_cost_usd", 8)]
    heads = ["source", "unit", "texts", "cand", "pass", "halluc", "dup", "final", "sec", "cost$"]
    print("  ".join(h.ljust(w) for h, (_, w) in zip(heads, cols)))
    print("  ".join("-" * w for _, w in cols))
    for r in rows:
        cells = []
        for key, width in cols:
            value = r.get(key, "")
            if key in ("pass_rate", "hallucination_rate"):
                value = f"{value * 100:.1f}%"
            cells.append(str(value)[:width].ljust(width))
        print("  ".join(cells))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subreddits", nargs="*", default=SUBREDDITS)
    parser.add_argument("--posts", type=int, default=20, help="recent distinct posts per subreddit")
    parser.add_argument("--papers", nargs="*", default=PAPERS, help="arXiv/PDF links")
    parser.add_argument("--paper-max", type=int, default=15)
    parser.add_argument("--no-anki", action="store_true", help="skip the eval-deck upload step")
    parser.add_argument("--price-in", type=float, default=0.27, help="USD per 1M prompt tokens (estimate)")
    parser.add_argument("--price-out", type=float, default=1.10, help="USD per 1M completion tokens (estimate)")
    args = parser.parse_args(argv)

    config = ac.load_config()
    api_key = ac.load_env().get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY missing in .env")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    workdir = Path(tempfile.mkdtemp(prefix="anki-eval-"))
    # Hard isolation: every module-level path that could reach real state points into the temp dir.
    ac.DB_PATH, ac.SUMMARY_PATH, ac.BRIEFING = workdir / "eval.sqlite3", workdir / "summary.json", workdir / "none"
    db = ac.connect(workdir / "eval.sqlite3")
    day = ac.now_kst().date().isoformat()
    known, units, started = set(), [], time.time()
    try:
        for subreddit in args.subreddits:
            ac.log(f"eval r/{subreddit}")
            units.append(eval_subreddit(db, config, api_key, subreddit, args.posts, known, day))
        for url in args.papers:
            ac.log(f"eval paper {url}")
            units.append(eval_paper(db, config, api_key, url, args.paper_max, known, day))
        anki_status = "skipped (--no-anki)" if args.no_anki else upload_and_cleanup(db, config)
    finally:
        db.close()
        shutil.rmtree(workdir, ignore_errors=True)

    prices = (args.price_in, args.price_out)
    rows = [u.row(run_id, prices) for u in units]
    rows += [total_row(rows, run_id, s) for s in ("reddit", "paper", "all")]
    words = [{"run_id": run_id, "word": i["word"].strip(), "cefr": level_of(i), "source": u.source,
              "origin": u.name, "meaning": i["definition"].strip()} for u in units for i in u.final]
    meta = {
        "run_id": run_id, "date": ac.now_kst().isoformat(timespec="seconds"), "model": config["llmModel"],
        "llm_url": config["llmUrl"], "subreddits": args.subreddits, "posts_per_subreddit": args.posts,
        "reddit_listing": "new", "post_length_filter": [config["minPostChars"], config["maxPostChars"]],
        "papers": args.papers, "paper_max_words": args.paper_max, "paper_chunks_visited": PAPER_CHUNKS,
        "items_requested_per_call": LLM_COUNT, "per_text_cap": PER_UNIT_CAP,
        "filter": "CEFR C1/C2, or B2 for multi-word expressions; example sentence must appear verbatim in source; "
                  "word (or stem) must appear in that sentence; dedup by normalized lemma across the whole run",
        "dedup_scope": "within this eval run (fresh temp DB, empty avoid list)",
        "price_usd_per_1m_tokens": {"prompt": prices[0], "completion": prices[1], "note": "estimate, set via CLI"},
        "anki": anki_status, "total_seconds": round(time.time() - started, 1),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / f"summary_{run_id}.csv", rows)
    if words:
        write_csv(RESULTS / f"words_{run_id}.csv", words)
    (RESULTS / f"meta_{run_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print_table(rows)
    print(f"\nanki: {anki_status}\nresults: {RESULTS}/*_{run_id}.*")
    return rows, words, meta


if __name__ == "__main__":
    main()
