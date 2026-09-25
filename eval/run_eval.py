#!/usr/bin/env python3
"""Evaluation run for the anki-crawling pipeline: stage-by-stage counts for Reddit and papers.

Every LLM reply is screened twice: by the current filter chain (vocab_filters.screen, what production runs)
and by the frozen pre-2026-09-26 filter (legacy_screen), so before/after differences come from the filters
alone, on identical candidates.

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
import vocab_filters as vf  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
EVAL_DECK = "English::eval-test"
BASELINE_RUN = "20260925-213036"  # last run with the old filter, kept for reference
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
PAPER_CHUNKS_MAX = 10  # upper bound on chunks visited per paper (same as run_paper); stops early at --paper-max
NEW_REASONS = ["schema", "non_english", "code_term", "cefr", "basic_expression", "not_verbatim", "word_missing",
               "generic_phrase", "not_idiomatic", "judge_error", "duplicate", "stem_duplicate", "cap"]
QUALITY = {"schema", "non_english", "code_term", "cefr", "basic_expression", "not_verbatim", "word_missing",
           "generic_phrase", "not_idiomatic", "judge_error"}
RESIDUAL = ["non_english", "code_term", "basic_expression", "generic_phrase", "not_idiomatic", "judge_error"]


def legacy_reason(item, body):
    """Frozen copy of the filter used until 2026-09-25 (for before/after comparison only)."""
    if not isinstance(item, dict):
        return "schema"
    fields = ("word", "definition", "reddit_sentence", "academic_example")
    if not all(isinstance(item.get(f), str) and item[f].strip() for f in fields):
        return "schema"
    level = str(item.get("cefr", "C1")).upper()[:2]
    if level not in ("C1", "C2") and not (level == "B2" and " " in item["word"].strip()):
        return "cefr"
    if not vf.sentence_in_source(item, body):
        return "not_verbatim"
    surface = (item.get("surface") or item["word"]).lower()
    stem = item["word"].lower().split()[0][:max(4, len(item["word"].split()[0]) - 2)]
    sentence = vf.normalize(item["reddit_sentence"]).lower()
    return None if surface in sentence or stem in sentence else "word_missing"


def legacy_screen(items, body, known, cap):
    """Old chain: legacy_reason -> exact-form dedup -> cap. Returns (accepted, reasons)."""
    accepted, reasons = [], []
    for item in items:
        reason = legacy_reason(item, body)
        if reason is None:
            key = vf.word_key(item["word"])
            if not key or key in known:
                reason = "duplicate"
            elif len(accepted) >= cap:
                reason = "cap"
            else:
                known.add(key)
                accepted.append(item)
        reasons.append(reason)
    return accepted, reasons


def level_of(item):
    level = str(item.get("cefr", "")).upper()[:2] if isinstance(item, dict) else ""
    return level if level in LEVELS else "other"


def ratio(part, whole):
    return round(part / whole, 4) if whole else 0.0


def stem_duplicates_in(words):
    """How many words in a final list repeat an earlier word's stem (what the old exact-form dedup misses)."""
    seen, count = vf.Seen(), 0
    for word in words:
        count += seen.check(word) == "stem_duplicate"
        seen.add(word)
    return count


class Unit:
    """Counters for one run unit (a subreddit or a paper)."""

    def __init__(self, source, name):
        self.source, self.name = source, name
        self.posts_fetched = self.posts_eligible = self.chunks_total = ""
        self.texts = self.calls = self.llm_errors = 0
        self.candidates = []          # rows for candidates CSV
        self.new_reasons, self.legacy_reasons = Counter(), Counter()
        self.new_final, self.legacy_final = [], []
        self.legacy_final_new_reason = Counter()  # what the new chain says about words the old one accepted
        self.not_verbatim_any = self.schema_ok = 0
        self.usage = {}
        self.seconds = 0.0
        self.note = ""

    def screen_reply(self, items, body, context, config, api_key, new_cap, legacy_cap, text_index):
        new_seen, legacy_known = context
        kept, reasons, verdicts = vf.screen(items, body, new_seen, config, api_key, new_cap, usage=self.usage)
        legacy_kept, legacy_reasons = legacy_screen(items, body, legacy_known, legacy_cap)
        for index, item in enumerate(items):
            has_sentence = isinstance(item, dict) and isinstance(item.get("reddit_sentence"), str) \
                and item["reddit_sentence"].strip()
            verbatim = bool(has_sentence) and vf.sentence_in_source(item, body)
            if has_sentence:  # hallucination rate, independent of the other filters
                self.schema_ok += 1
                self.not_verbatim_any += not verbatim
            self.new_reasons[reasons[index] or "accepted"] += 1
            self.legacy_reasons[legacy_reasons[index] or "accepted"] += 1
            if legacy_reasons[index] is None:
                self.legacy_final_new_reason[reasons[index] or "accepted"] += 1
            word = item.get("word", "") if isinstance(item, dict) else ""
            self.candidates.append({
                "source": self.source, "unit": self.name, "text_index": text_index, "word": str(word).strip(),
                "cefr": level_of(item), "multiword": vf.is_multiword(str(word)) if word else False,
                "sentence": vf.normalize(str(item.get("reddit_sentence", ""))) if isinstance(item, dict) else "",
                "sentence_in_source": verbatim, "judge_category": verdicts[index],
                "legacy_result": legacy_reasons[index] or "accepted", "new_result": reasons[index] or "accepted",
            })
        self.new_final += kept
        self.legacy_final += legacy_kept

    def summary_row(self, run_id, prices):
        before = Counter(level_of(c) for c in self.candidates_items())
        after = Counter(level_of(i) for i in self.new_final)
        total = len(self.candidates)
        quality_rejected = sum(self.new_reasons[r] for r in QUALITY)
        extraction = (self.usage.get("prompt_tokens", 0) * prices[0]
                      + self.usage.get("completion_tokens", 0) * prices[1]) / 1e6
        judge = (self.usage.get("judge_prompt_tokens", 0) * prices[0]
                 + self.usage.get("judge_completion_tokens", 0) * prices[1]) / 1e6
        return {
            "run_id": run_id, "source": self.source, "unit": self.name,
            "posts_fetched": self.posts_fetched, "posts_eligible": self.posts_eligible,
            "chunks_total": self.chunks_total, "texts": self.texts, "llm_calls": self.calls,
            "llm_errors": self.llm_errors, "candidates": total,
            **{f"before_{k}": before.get(k, 0) for k in LEVELS + ("other",)},
            **{f"rejected_{r}": self.new_reasons[r] for r in NEW_REASONS if r not in ("duplicate", "stem_duplicate", "cap")},
            "passed_filter": total - quality_rejected, "pass_rate": ratio(total - quality_rejected, total),
            "hallucinated_examples": self.not_verbatim_any, "hallucination_rate": ratio(self.not_verbatim_any, self.schema_ok),
            "duplicates_removed": self.new_reasons["duplicate"], "stem_duplicates_removed": self.new_reasons["stem_duplicate"],
            "cap_removed": self.new_reasons["cap"], "final_words": len(self.new_final),
            **{f"after_{k}": after.get(k, 0) for k in LEVELS + ("other",)},
            "seconds": round(self.seconds, 1),
            "prompt_tokens": self.usage.get("prompt_tokens", 0), "completion_tokens": self.usage.get("completion_tokens", 0),
            "judge_prompt_tokens": self.usage.get("judge_prompt_tokens", 0),
            "judge_completion_tokens": self.usage.get("judge_completion_tokens", 0),
            "est_cost_usd": round(extraction + judge, 4), "note": self.note,
        }

    def candidates_items(self):
        return [{"cefr": c["cefr"]} for c in self.candidates]

    def compare_metrics(self, prices):
        """Same-candidate before/after metrics for the compare CSV."""
        def final_stats(final):
            n = len(final)
            levels = Counter(level_of(i) for i in final)
            return {
                "final_words": n,
                "final_C1plus_share": ratio(levels["C1"] + levels["C2"], n),
                "final_B2_share": ratio(levels["B2"], n),
                "final_multiword_share": ratio(sum(vf.is_multiword(i["word"]) for i in final), n),
                "residual_stem_duplicate": stem_duplicates_in([i["word"].strip() for i in final]),
            }
        total = len(self.candidates)
        legacy_quality = sum(v for k, v in self.legacy_reasons.items() if k in QUALITY)
        new_quality = sum(self.new_reasons[r] for r in QUALITY)
        extraction = (self.usage.get("prompt_tokens", 0) * prices[0]
                      + self.usage.get("completion_tokens", 0) * prices[1]) / 1e6
        judge = (self.usage.get("judge_prompt_tokens", 0) * prices[0]
                 + self.usage.get("judge_completion_tokens", 0) * prices[1]) / 1e6
        legacy = {"candidates": total, "pass_rate": ratio(total - legacy_quality, total), **final_stats(self.legacy_final),
                  "est_cost_usd": round(extraction, 4)}
        new = {"candidates": total, "pass_rate": ratio(total - new_quality, total), **final_stats(self.new_final),
               "est_cost_usd": round(extraction + judge, 4)}
        for reason in RESIDUAL:  # problem words that reached the final list
            legacy[f"residual_{reason}"] = self.legacy_final_new_reason[reason]
            new[f"residual_{reason}"] = 0  # by construction the new chain rejects these
        legacy["residual_total"] = sum(legacy[f"residual_{r}"] for r in RESIDUAL) + legacy["residual_stem_duplicate"]
        new["residual_total"] = new["residual_stem_duplicate"]
        return legacy, new


def total_row(rows, run_id, source):
    rows = [r for r in rows if source == "all" or r["source"] == source]
    numeric = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool)] if rows else []
    out = {k: sum(r[k] for r in rows if isinstance(r[k], (int, float))) for k in numeric}
    out.update(run_id=run_id, source=source, unit="TOTAL", note="", posts_fetched="", posts_eligible="", chunks_total="")
    out["pass_rate"] = ratio(out.get("passed_filter", 0), out.get("candidates", 0))
    schema_ok = sum(r["candidates"] - r["rejected_schema"] for r in rows)
    out["hallucination_rate"] = ratio(out.get("hallucinated_examples", 0), schema_ok)
    out["est_cost_usd"] = round(out.get("est_cost_usd", 0), 4)
    out["seconds"] = round(out.get("seconds", 0), 1)
    return {k: out.get(k, "") for k in rows[0]} if rows else out


def eval_subreddit(config, api_key, subreddit, posts_wanted, context):
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
    unit.posts_fetched, unit.posts_eligible = len(raw), len(seen)
    for index, post in enumerate(posts[:posts_wanted]):
        unit.texts += 1
        try:
            unit.calls += 1
            items = ac.call_llm(config, api_key, post["title"], post["body"], context[0].words[-300:], LLM_COUNT,
                                usage=unit.usage)
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
            unit.llm_errors += 1
            continue
        unit.screen_reply(items, post["body"], context, config, api_key, PER_UNIT_CAP, PER_UNIT_CAP, index)
    unit.seconds = time.time() - started
    return unit


def eval_paper(config, api_key, url, max_words, context):
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
    unit.chunks_total = len(parts)
    unit.note = f"{url} · {len(body)} chars"
    for index in ac.spread_order(len(parts))[:PAPER_CHUNKS_MAX]:
        if len(unit.new_final) >= max_words:
            break
        unit.texts += 1
        want = min(LLM_COUNT, max_words - len(unit.new_final) + 3)
        try:
            unit.calls += 1
            items = ac.call_llm(config, api_key, title, parts[index], context[0].words[-300:], want,
                                kind="research paper excerpt", usage=unit.usage)
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
            unit.llm_errors += 1
            continue
        unit.screen_reply(items, parts[index], context, config, api_key,
                          min(PER_UNIT_CAP, max_words - len(unit.new_final)),
                          min(PER_UNIT_CAP, max(0, max_words - len(unit.legacy_final))), index)
    unit.seconds = time.time() - started
    return unit


def upload_and_cleanup(units, config):
    """Push the new chain's words to the eval deck only (proves the Anki path), then delete that deck."""
    anki = ac.Anki(config["ankiUrl"])
    if not ac.ensure_anki_running(anki):
        return "anki unavailable"
    if EVAL_DECK in anki.call("deckNames"):
        raise RuntimeError(f"{EVAL_DECK} already exists; remove it before running eval")
    anki.call("createDeck", deck=EVAL_DECK)
    added = total = 0
    try:
        for unit in units:
            for item in unit.new_final:
                total += 1
                row = {"word": item["word"].strip(), "definition": item["definition"].strip(),
                       "reddit_sentence": ac.highlight(vf.normalize(item["reddit_sentence"]), item),
                       "academic_example": item["academic_example"].strip(),
                       "pronunciation": str(item.get("pronunciation") or "")}
                note = {"deckName": EVAL_DECK, "modelName": config["modelName"],
                        "fields": ac.note_fields(row, ac.html.escape(unit.name)),
                        "tags": ["eval_test"], "options": {"allowDuplicate": True}}
                anki.call("addNote", note=note)
                added += 1
    finally:
        anki.call("deleteDecks", decks=[EVAL_DECK], cardsToo=True)
    left = EVAL_DECK in anki.call("deckNames")
    return f"uploaded {added}/{total} to {EVAL_DECK}, deck deleted={not left}"


def baseline_totals():
    """Previous run's totals and residual problems that can be recomputed from its word list."""
    summary, words = RESULTS / f"summary_{BASELINE_RUN}.csv", RESULTS / f"words_{BASELINE_RUN}.csv"
    if not summary.exists() or not words.exists():
        return {}
    with summary.open(encoding="utf-8") as handle:
        rows = {(r["source"], r["unit"]): r for r in csv.DictReader(handle)}
    with words.open(encoding="utf-8") as handle:
        word_rows = list(csv.DictReader(handle))
    out = {}
    for source in ("reddit", "paper", "all"):
        row = rows.get((source, "TOTAL"))
        if not row:
            continue
        final = [w for w in word_rows if source == "all" or w["source"] == source]
        levels = Counter(w["cefr"] for w in final)
        n = len(final)
        seen, stem_dups = vf.Seen(), 0
        for w in final:
            stem_dups += seen.check(w["word"]) == "stem_duplicate"
            seen.add(w["word"])
        problems = Counter(vf.headword_problem(w["word"]) for w in final)
        basic = sum(vf.is_multiword(w["word"]) and vf.lemma_tokens(w["word"]) in vf.BASIC_KEYS for w in final)
        out[source] = {
            "texts": int(row["texts"]), "candidates": int(row["candidates"]), "pass_rate": float(row["pass_rate"]),
            "final_words": n, "final_C1plus_share": ratio(levels["C1"] + levels["C2"], n),
            "final_B2_share": ratio(levels["B2"], n),
            "final_multiword_share": ratio(sum(vf.is_multiword(w["word"]) for w in final), n),
            "residual_non_english": problems["non_english"], "residual_code_term": problems["code_term"],
            "residual_basic_expression": basic, "residual_generic_phrase": "", "residual_not_idiomatic": "",
            "residual_judge_error": "", "residual_stem_duplicate": stem_dups, "residual_total": "",
            "hallucination_rate": float(row["hallucination_rate"]), "est_cost_usd": float(row["est_cost_usd"]),
            "seconds": float(row["seconds"]),
        }
    return out


COMPARE_METRICS = ["texts", "candidates", "pass_rate", "final_words", "final_C1plus_share", "final_B2_share",
                   "final_multiword_share", "residual_non_english", "residual_code_term", "residual_basic_expression",
                   "residual_generic_phrase", "residual_not_idiomatic", "residual_judge_error",
                   "residual_stem_duplicate", "residual_total", "hallucination_rate", "est_cost_usd", "seconds"]


def compare_rows(run_id, units, summary_rows, prices):
    """Long-format before/after table: previous run | old filter on this run's candidates | new filter."""
    baseline = baseline_totals()
    per_unit = {u.name: u.compare_metrics(prices) for u in units}
    rows = []

    def add(scope, unit_name, metric, previous, legacy, new):
        change = round(new - legacy, 4) if isinstance(new, (int, float)) and isinstance(legacy, (int, float)) else ""
        rows.append({"run_id": run_id, "baseline_run": BASELINE_RUN, "source": scope, "unit": unit_name,
                     "metric": metric, "previous_run": previous, "old_filter_same_candidates": legacy,
                     "new_filter": new, "change_new_vs_old": change})

    for source in ("reddit", "paper", "all"):
        chosen = [u for u in units if source == "all" or u.source == source]
        if not chosen:
            continue
        legacy = Counter()
        new = Counter()
        for u in chosen:
            l_metrics, n_metrics = per_unit[u.name]
            for key in l_metrics:
                if key not in ("pass_rate",) and not key.endswith("_share"):
                    legacy[key] += l_metrics[key]
                    new[key] += n_metrics[key]
        total = next(r for r in summary_rows if r["source"] == source and r["unit"] == "TOTAL")
        legacy_final = [i for u in chosen for i in u.legacy_final]
        new_final = [i for u in chosen for i in u.new_final]
        legacy_quality = sum(v for u in chosen for k, v in u.legacy_reasons.items() if k in QUALITY)

        def shares(final):
            n, levels = len(final), Counter(level_of(i) for i in final)
            return {"final_C1plus_share": ratio(levels["C1"] + levels["C2"], n), "final_B2_share": ratio(levels["B2"], n),
                    "final_multiword_share": ratio(sum(vf.is_multiword(i["word"]) for i in final), n)}
        legacy_values = {**legacy, **shares(legacy_final), "texts": total["texts"],
                         "pass_rate": ratio(total["candidates"] - legacy_quality, total["candidates"]),
                         "residual_stem_duplicate": stem_duplicates_in([i["word"].strip() for i in legacy_final]),
                         "hallucination_rate": total["hallucination_rate"], "seconds": ""}
        legacy_values["residual_total"] = sum(legacy_values[f"residual_{r}"] for r in RESIDUAL) + legacy_values["residual_stem_duplicate"]
        new_values = {**new, **shares(new_final), "texts": total["texts"], "pass_rate": total["pass_rate"],
                      "residual_stem_duplicate": stem_duplicates_in([i["word"].strip() for i in new_final]),
                      "hallucination_rate": total["hallucination_rate"], "est_cost_usd": total["est_cost_usd"],
                      "seconds": total["seconds"]}
        new_values["residual_total"] = new_values["residual_stem_duplicate"]
        legacy_values["est_cost_usd"] = round(legacy["est_cost_usd"], 4)
        for metric in COMPARE_METRICS:
            add(source, "TOTAL", metric, baseline.get(source, {}).get(metric, ""), legacy_values.get(metric, ""),
                new_values.get(metric, ""))
    for u in units:
        l_metrics, n_metrics = per_unit[u.name]
        for metric in ("candidates", "pass_rate", "final_words", "final_C1plus_share", "residual_total"):
            add(u.source, u.name, metric, "", l_metrics[metric], n_metrics[metric])
    return rows


def print_table(rows, cols, heads):
    print("  ".join(h.ljust(w) for h, (_, w) in zip(heads, cols)))
    print("  ".join("-" * w for _, w in cols))
    for r in rows:
        cells = []
        for key, width in cols:
            value = r.get(key, "")
            if isinstance(value, float) and r.get("metric", key).endswith(("rate", "share")):
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
    context = (vf.Seen(), set())  # fresh learned-word state for the new and the old chain
    units, started = [], time.time()
    try:
        for subreddit in args.subreddits:
            ac.log(f"eval r/{subreddit}")
            units.append(eval_subreddit(config, api_key, subreddit, args.posts, context))
        for url in args.papers:
            ac.log(f"eval paper {url}")
            units.append(eval_paper(config, api_key, url, args.paper_max, context))
        anki_status = "skipped (--no-anki)" if args.no_anki else upload_and_cleanup(units, config)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    prices = (args.price_in, args.price_out)
    rows = [u.summary_row(run_id, prices) for u in units]
    rows += [total_row(rows, run_id, s) for s in ("reddit", "paper", "all") if any(r["source"] == s for r in rows) or s == "all"]
    words = [{"run_id": run_id, "pipeline": pipeline, "word": i["word"].strip(), "cefr": level_of(i), "source": u.source,
              "origin": u.name, "meaning": i["definition"].strip()}
             for u in units for pipeline, final in (("new", u.new_final), ("old", u.legacy_final)) for i in final]
    candidates = [{"run_id": run_id, **c} for u in units for c in u.candidates]
    compare = compare_rows(run_id, units, rows, prices)
    meta = {
        "run_id": run_id, "date": ac.now_kst().isoformat(timespec="seconds"), "model": config["llmModel"],
        "llm_url": config["llmUrl"], "subreddits": args.subreddits, "posts_per_subreddit": args.posts,
        "reddit_listing": "new", "post_length_filter": [config["minPostChars"], config["maxPostChars"]],
        "papers": args.papers, "paper_max_words": args.paper_max,
        "paper_chunks_max": PAPER_CHUNKS_MAX,
        "paper_chunks_note": "upper bound per paper; a paper stops as soon as paper_max_words are kept, so summary "
                             "'texts' (chunks actually visited) is usually smaller; 'chunks_total' is the paper's size",
        "items_requested_per_call": LLM_COUNT, "per_text_cap": PER_UNIT_CAP,
        "filter_chain": "schema -> non_english (non-ASCII headword, or example sentence not English) -> code_term "
                        "(code shape, acronym, tool name) -> cefr (C1/C2, or B2 multi-word) -> basic_expression "
                        "(stem-matched blocklist) -> not_verbatim -> word_missing -> LLM judge for multi-word "
                        "expressions and non-dictionary single words (generic_phrase, not_idiomatic for B2, "
                        "basic_expression, code_term, non_english) -> duplicate / stem_duplicate (Porter stems, "
                        "phrase containment) -> cap",
        "old_filter": "cefr (C1/C2 or any B2 multi-word) -> not_verbatim -> word_missing -> exact-form dedup -> cap",
        "comparison": "old and new filters applied to the same LLM candidates in this run; previous_run column is "
                      f"run {BASELINE_RUN} (different posts/LLM samples, reference only)",
        "dictionary": str(vf.DICTIONARY), "stemmer": "Porter (1980), checked against nltk ORIGINAL_ALGORITHM",
        "dedup_scope": "within this eval run (fresh state, empty avoid list)",
        "price_usd_per_1m_tokens": {"prompt": prices[0], "completion": prices[1], "note": "estimate, set via CLI"},
        "anki": anki_status, "total_seconds": round(time.time() - started, 1),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / f"summary_{run_id}.csv", rows)
    if words:
        write_csv(RESULTS / f"words_{run_id}.csv", words)
    if candidates:
        write_csv(RESULTS / f"candidates_{run_id}.csv", candidates)
    write_csv(RESULTS / f"compare_{BASELINE_RUN}_vs_{run_id}.csv", compare)
    (RESULTS / f"meta_{run_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print_table(rows, [("source", 7), ("unit", 34), ("texts", 5), ("candidates", 5), ("pass_rate", 7),
                       ("hallucination_rate", 7), ("stem_duplicates_removed", 5), ("final_words", 5), ("seconds", 7),
                       ("est_cost_usd", 7)],
                ["source", "unit", "texts", "cand", "pass", "halluc", "stemd", "final", "sec", "cost$"])
    print()
    print_table([r for r in compare if r["unit"] == "TOTAL"],
                [("source", 7), ("metric", 26), ("previous_run", 9), ("old_filter_same_candidates", 9), ("new_filter", 9)],
                ["source", "metric", "prev run", "old", "new"])
    print(f"\nanki: {anki_status}\nresults: {RESULTS}/*{run_id}*")
    return rows, words, meta, compare


if __name__ == "__main__":
    main()
