#!/usr/bin/env python3
"""Reddit posts or papers -> LLM vocabulary extraction -> Anki cards (no duplicates).

Daily mode: 10-15 new words from subreddits.  Paper mode: `--paper <pdf|url|txt>` on demand.
"""
import hashlib
import tempfile
import argparse
import html
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "state.sqlite3"
CONFIG_PATH = ROOT / "config.json"
ENV_PATH = ROOT / ".env"
SUMMARY_PATH = ROOT / "daily_summary.json"
DISABLED = ROOT / "disabled"
BRIEFING = Path("/Users/minjaeku/hermes-yuna/scripts/evening_briefing.py")
KST = ZoneInfo("Asia/Seoul")
USER_AGENT = "macos:anki-crawling:v1.0 (personal vocabulary builder)"
ATOM = "{http://www.w3.org/2005/Atom}"

DEFAULT_CONFIG = {
    "subreddits": ["robotics", "MachineLearning", "ControlTheory", "ROS", "cpp", "compsci", "math"],
    "listing": "top",
    "timeRange": "week",
    "dailyMin": 10,
    "dailyMax": 15,
    "minPostChars": 300,
    "maxPostChars": 6000,
    "maxPostsPerRun": 12,
    "fetchLimit": 50,
    "deckName": "English::reddit-vocab",
    "paperDeckName": "English::paper-vocab",
    "paperMaxWords": 15,
    "modelName": "vocab (eng)",
    "ankiUrl": "http://127.0.0.1:8765",
    "llmUrl": "https://api.deepseek.com/chat/completions",
    "llmModel": "deepseek-chat",
}

PROMPT = """You are building English vocabulary flashcards for a Korean robotics/mechanical engineering graduate student (English level: upper-intermediate).
From the {kind} below, extract up to {count} vocabulary items that are useful for robotics, CS, math, or academic/technical discussion, at CEFR B2-C2 level.
Include idioms and phrasal verbs if useful. Prefer words that are hard for Korean learners: academic verbs, nuanced adjectives, collocations, idioms, phrasal verbs (e.g. "formulate", "tractable", "rule of thumb", "iron out").
Skip words a Korean engineering graduate student surely knows already (e.g. facility, capacity, robot, algorithm, numerical, humanoid, sparse, linear system),
basic words, proper nouns, product names, code identifiers and acronyms. Fewer good items is better than padding.
Do NOT pick any of these already-learned words: {avoid}

Return JSON only: {{"items": [{{
  "word": "headword in dictionary form (lemma)",
  "surface": "the exact form as it appears in the text",
  "pronunciation": "Korean Hangul transcription of the American pronunciation, stress syllable unmarked, e.g. 포뮬레이트",
  "cefr": "honest CEFR level of the headword: B1, B2, C1 or C2",
  "definition": "concise Korean meaning in this context",
  "reddit_sentence": "ONE sentence copied VERBATIM from the text that contains the word",
  "academic_example": "a new English sentence as it might appear in a robotics/engineering paper or interview"
}}]}}

TITLE: {title}
TEXT:
{body}"""


def now_kst():
    return datetime.now(KST)


def log(message):
    print(f"[{now_kst():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def load_config():
    try:
        stored = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        stored = {}
    return {**DEFAULT_CONFIG, **stored}


def load_env():
    values = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("'\"")
    return {**values, **{k: v for k, v in os.environ.items() if k.endswith("_API_KEY")}}


# ---------- storage ----------

def connect(path=None):
    db = sqlite3.connect(str(path or DB_PATH))
    db.row_factory = sqlite3.Row
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY, subreddit TEXT, title TEXT, url TEXT, processed_at REAL
        );
        CREATE TABLE IF NOT EXISTS cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word_key TEXT UNIQUE NOT NULL,
            word TEXT NOT NULL, definition TEXT, reddit_sentence TEXT, academic_example TEXT,
            subreddit TEXT, post_title TEXT, post_url TEXT,
            created_date TEXT NOT NULL, created_at REAL NOT NULL,
            anki_note_id INTEGER, pushed_at REAL
        );
        """
    )
    columns = [r[1] for r in db.execute("PRAGMA table_info(cards)")]
    for column in ("cefr", "pronunciation", "source", "deck"):
        if column not in columns:
            db.execute(f"ALTER TABLE cards ADD COLUMN {column} TEXT")
    return db


def word_key(word):
    return re.sub(r"[^a-z0-9 ]", "", word.lower().replace("-", " ")).strip()


def added_today(db, day):
    return db.execute("SELECT COUNT(*) FROM cards WHERE created_date = ? AND COALESCE(source, 'reddit') = 'reddit'",
                      (day,)).fetchone()[0]


# ---------- reddit ----------

def http_get(url, timeout=20):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def html_to_text(fragment):
    fragment = re.sub(r"(?is)<(pre|code)[^>]*>.*?</\1>", " ", fragment)
    fragment = re.sub(r"(?i)<br\s*/?>|</p>|</li>", "\n", fragment)
    text = html.unescape(re.sub(r"<[^>]+>", " ", fragment))
    text = re.sub(r"submitted by\s+/u/\S+.*$", "", text, flags=re.S)
    text = re.sub(r"\[link\]|\[comments\]", " ", text)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", text)).strip()


def parse_feed(raw, subreddit):
    posts = []
    for entry in ET.fromstring(raw).findall(f"{ATOM}entry"):
        post_id = (entry.findtext(f"{ATOM}id") or "").strip()
        link = entry.find(f"{ATOM}link")
        body = html_to_text(entry.findtext(f"{ATOM}content") or "")
        if post_id:
            posts.append({
                "id": post_id,
                "subreddit": subreddit,
                "title": html.unescape(entry.findtext(f"{ATOM}title") or "").strip(),
                "url": link.get("href") if link is not None else "",
                "body": body,
            })
    return posts


def fetch_posts(config):
    posts = []
    for subreddit in config["subreddits"]:
        url = f"https://www.reddit.com/r/{subreddit}/{config['listing']}/.rss?t={config['timeRange']}&limit={config.get('fetchLimit', 50)}"
        for attempt in range(3):
            try:
                posts.extend(parse_feed(http_get(url), subreddit))
                break
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == 2:
                    log(f"r/{subreddit} fetch failed: {error}")
                    break
                time.sleep(30 * (attempt + 1))  # rate limited
            except (urllib.error.URLError, ET.ParseError, OSError) as error:
                log(f"r/{subreddit} fetch failed: {error}")
                break
        time.sleep(8)  # unauthenticated RSS is rate limited
    return [p for p in posts if config["minPostChars"] <= len(p["body"]) <= config["maxPostChars"]]


# ---------- llm ----------

def call_llm(config, api_key, title, body, avoid, count, kind="Reddit post", usage=None):
    prompt = PROMPT.format(kind=kind, count=count, avoid=", ".join(avoid) or "(none)", title=title, body=body)
    payload = {
        "model": config["llmModel"],
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
    }
    request = urllib.request.Request(
        config["llmUrl"],
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        reply = json.loads(response.read())
    content = reply["choices"][0]["message"]["content"]
    if usage is not None:  # token accounting for eval
        for key in ("prompt_tokens", "completion_tokens"):
            usage[key] = usage.get(key, 0) + reply.get("usage", {}).get(key, 0)
    items = json.loads(content).get("items", [])
    return items if isinstance(items, list) else []


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def reject_reason(item, body):
    """None if the item passes every quality filter, else the first failing check."""
    if not isinstance(item, dict):
        return "schema"
    fields = ("word", "definition", "reddit_sentence", "academic_example")
    if not all(isinstance(item.get(f), str) and item[f].strip() for f in fields):
        return "schema"
    # Hard single words only; multi-word expressions (idioms, collocations) may be B2.
    level = str(item.get("cefr", "C1")).upper()[:2]
    if level not in ("C1", "C2") and not (level == "B2" and " " in item["word"].strip()):
        return "cefr"
    if not sentence_in_source(item, body):
        return "not_verbatim"
    surface = (item.get("surface") or item["word"]).lower()
    stem = item["word"].lower().split()[0][:max(4, len(item["word"].split()[0]) - 2)]
    sentence = normalize(item["reddit_sentence"]).lower()
    return None if surface in sentence or stem in sentence else "word_missing"


def sentence_in_source(item, body):
    # The example must be a real sentence from the source text (guards against invented examples).
    sentence = normalize(item["reddit_sentence"])
    return len(sentence) >= 20 and sentence.lower() in normalize(body).lower()


def valid_item(item, body):
    return reject_reason(item, body) is None


def highlight(sentence, item):
    target = item.get("surface") or item["word"]
    escaped = html.escape(sentence)
    pattern = re.compile(re.escape(html.escape(target)), re.I)
    if pattern.search(escaped):
        return pattern.sub(lambda m: f"<b>{m.group(0)}</b>", escaped, count=1)
    return escaped


# ---------- anki ----------

class Anki:
    def __init__(self, url):
        self.url = url

    def call(self, action, **params):
        data = json.dumps({"action": action, "version": 6, "params": params}).encode()
        request = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read())
        if result.get("error"):
            raise RuntimeError(f"AnkiConnect {action}: {result['error']}")
        return result.get("result")

    def available(self):
        try:
            return bool(self.call("version"))
        except (urllib.error.URLError, OSError, RuntimeError):
            return False

    def ensure(self, deck, model):
        self.call("createDeck", deck=deck)
        if model not in self.call("modelNames"):
            self.call(
                "createModel",
                modelName=model,
                inOrderFields=["Word", "Meaning", "Sentence", "Academic", "Source"],
                css=".card{font-family:-apple-system,sans-serif;font-size:20px;text-align:center;}"
                ".sent{font-style:italic;margin-top:14px}.src{font-size:12px;color:#888;margin-top:18px}",
                cardTemplates=[{
                    "Name": "Word -> Meaning",
                    "Front": "<div style='font-size:30px'><b>{{Word}}</b></div><div class='sent'>{{Sentence}}</div>",
                    "Back": "{{FrontSide}}<hr id=answer><div><b>뜻</b> {{Meaning}}</div>"
                            "<div class='sent'><b>학술 예문</b> {{Academic}}</div><div class='src'>{{Source}}</div>",
                }],
            )


def note_fields(row, source):
    """Fields of the user's 'vocab (eng)' note type."""
    return {
        "word": html.escape(row["word"]),
        "meaning": html.escape(row["definition"]),
        "example": row["reddit_sentence"],
        "memo": f"<i>{html.escape(row['academic_example'])}</i><br><span style='font-size:14px;color:gray'>{source}</span>",
        "pronunciation": html.escape(row["pronunciation"] or ""),
    }


def ensure_anki_running(anki):
    if anki.available():
        return True
    subprocess.run(["/usr/bin/open", "-g", "-a", "Anki"], capture_output=True, check=False)
    for _ in range(40):
        time.sleep(3)
        if anki.available():
            return True
    return False


def push_pending(db, config):
    rows = db.execute("SELECT * FROM cards WHERE anki_note_id IS NULL ORDER BY id").fetchall()
    if not rows:
        return 0, True
    anki = Anki(config["ankiUrl"])
    if not ensure_anki_running(anki):
        log(f"AnkiConnect unavailable; {len(rows)} card(s) kept pending")
        return 0, False
    for deck in {row["deck"] or config["deckName"] for row in rows}:
        anki.call("createDeck", deck=deck)
    if config["modelName"] not in anki.call("modelNames"):  # user's own note type; never created here
        raise RuntimeError(f"Anki note type not found: {config['modelName']}")
    pushed = 0
    for row in rows:
        if row["source"] == "paper":
            source = f"📄 {html.escape(row['post_title'])}"
            tags = ["paper_auto", row["created_date"]]
        else:
            source = f"r/{html.escape(row['subreddit'])} · <a href='{html.escape(row['post_url'])}'>{html.escape(row['post_title'])}</a>"
            tags = ["reddit_auto", f"r_{row['subreddit']}", row["created_date"]]
        note = {
            "deckName": row["deck"] or config["deckName"],
            "modelName": config["modelName"],
            "fields": note_fields(row, source),
            "tags": tags + ([row["cefr"]] if row["cefr"] else []),
            "options": {"allowDuplicate": False, "duplicateScope": "deck"},
        }
        try:
            note_id = anki.call("addNote", note=note)
        except RuntimeError as error:
            if "duplicate" not in str(error):
                raise
            note_id = -1  # already in Anki; never retry
        db.execute("UPDATE cards SET anki_note_id = ?, pushed_at = ? WHERE id = ?", (note_id, time.time(), row["id"]))
        db.commit()
        pushed += 1
    # Sync is mandatory after adding cards: retry, and report failure to dashboard/briefing.
    for attempt in range(3):
        try:
            anki.call("sync")
            log("AnkiWeb sync done")
            return pushed, True
        except (RuntimeError, urllib.error.URLError, OSError) as error:
            log(f"AnkiWeb sync failed ({attempt + 1}/3): {error}")
            time.sleep(20)
    raise RuntimeError("AnkiWeb 동기화 실패")


# ---------- pipeline ----------

def insert_card(db, key, item, post, day, source, deck):
    db.execute(
        "INSERT INTO cards (word_key, word, definition, reddit_sentence, academic_example, subreddit, post_title,"
        " post_url, created_date, created_at, cefr, pronunciation, source, deck) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (key, item["word"].strip(), item["definition"].strip(),
         highlight(normalize(item["reddit_sentence"]), item), item["academic_example"].strip(),
         post["subreddit"], post["title"], post["url"], day, time.time(),
         str(item.get("cefr", "")).upper()[:2] or None, str(item.get("pronunciation") or "").strip() or None,
         source, deck),
    )


def collect(db, config, api_key, target):
    known = {r[0] for r in db.execute("SELECT word_key FROM cards")}
    seen = {r[0] for r in db.execute("SELECT id FROM posts")}
    posts = [p for p in fetch_posts(config) if p["id"] not in seen]
    random.shuffle(posts)
    log(f"{len(posts)} unseen candidate post(s)")
    added = 0
    day = now_kst().date().isoformat()
    for post in posts[: config["maxPostsPerRun"]]:
        if added >= target:
            break
        avoid = sorted(known)[-300:]
        try:
            items = call_llm(config, api_key, post["title"], post["body"], avoid, min(8, target - added + 3))
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as error:
            log(f"LLM failed for {post['id']}: {error}")
            continue
        per_post = 0
        for item in items:
            if added >= target or per_post >= 4:  # spread words over several posts
                break
            if not valid_item(item, post["body"]):
                continue
            key = word_key(item["word"])
            if not key or key in known:
                continue
            insert_card(db, key, item, post, day, "reddit", config["deckName"])
            known.add(key)
            added += 1
            per_post += 1
        db.execute("INSERT OR IGNORE INTO posts VALUES (?,?,?,?,?)",
                   (post["id"], post["subreddit"], post["title"], post["url"], time.time()))
        db.commit()
        log(f"r/{post['subreddit']} '{post['title'][:50]}': +{per_post}")
    return added


def write_summary(db, day, error=""):
    rows = db.execute("SELECT * FROM cards WHERE created_date = ? AND COALESCE(source, 'reddit') = 'reddit'"
                      " ORDER BY id", (day,)).fetchall()
    summary = {
        "date": day,
        "added_today": len(rows),
        "words": [{"word": r["word"], "definition": r["definition"], "cefr": r["cefr"] or "",
                   "sentence": html.unescape(re.sub(r"<[^>]+>", "", r["reddit_sentence"] or "")),
                   "highlight": html.unescape((re.search(r"<b>(.*?)</b>", r["reddit_sentence"] or "") or [None, ""])[1]),
                   "academic": r["academic_example"], "subreddit": r["subreddit"], "url": r["post_url"]} for r in rows],
        "pending": db.execute("SELECT COUNT(*) FROM cards WHERE anki_note_id IS NULL").fetchone()[0],
        "total_cards": db.execute("SELECT COUNT(*) FROM cards").fetchone()[0],
        "error": error,
        "updated": time.time(),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def update_briefing():
    if BRIEFING.exists():
        subprocess.run(["/usr/bin/python3", str(BRIEFING), "--anki"], check=False, timeout=30)


def run(dry_run=False, force=False):
    config = load_config()
    if DISABLED.exists() and not force:
        log("disabled; skipping")
        return 0
    db = connect()
    day = now_kst().date().isoformat()
    error = ""
    try:
        have = added_today(db, day)
        goal = random.randint(config["dailyMin"], config["dailyMax"])
        if have < config["dailyMin"]:
            api_key = load_env().get("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("DEEPSEEK_API_KEY missing in .env")
            added = collect(db, config, api_key, goal - have)
            log(f"added {added} new word(s) today (goal {goal})")
        else:
            log(f"already {have} word(s) today")
        if not dry_run:
            pushed, ok = push_pending(db, config)
            log(f"pushed {pushed} card(s) to Anki")
            if not ok:
                error = "Anki 연결 실패 (카드 보류)"
    except Exception as exc:  # recorded for dashboard/briefing, then re-raised as exit code
        error = f"{type(exc).__name__}: {exc}"[:200]
        log(error)
    summary = write_summary(db, day, error)
    if not dry_run:
        update_briefing()
    return 1 if error and not summary["added_today"] else 0


# ---------- paper ----------

PDFTOTEXT = "/opt/homebrew/bin/pdftotext"


def paper_text(source):
    """Return (title, url, body) for a PDF path, .txt/.md path, arXiv/PDF/HTML URL."""
    url = source if re.match(r"https?://", source) else ""
    arxiv_title = ""
    if url:
        arxiv = re.search(r"arxiv\.org/(?:abs|pdf|html)/([\w.\-/]+?)(?:v\d+)?(?:\.pdf)?$", url)
        fetch = f"https://arxiv.org/pdf/{arxiv.group(1)}" if arxiv else url
        if arxiv:  # PDF metadata titles are unreliable; the abs page has the official one
            try:
                page = http_get(f"https://arxiv.org/abs/{arxiv.group(1)}").decode("utf-8", "replace")
                found = re.search(r'<meta name="citation_title" content="([^"]+)"', page)
                arxiv_title = html.unescape(found.group(1)).strip() if found else ""
            except (urllib.error.URLError, OSError):
                pass
        raw = http_get(fetch, timeout=60)
        if raw[:4] != b"%PDF":
            text = html_to_text(raw.decode("utf-8", "replace"))
            title = re.search(r"(?is)<title>(.*?)</title>", raw.decode("utf-8", "replace"))
            return (html.unescape(title.group(1)).strip() if title else url), url, clean_paper(text)
        handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        handle.write(raw)
        handle.close()
        path = Path(handle.name)
    else:
        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(source)
    if path.suffix.lower() in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        return path.stem, url or str(path), clean_paper(text)
    text = subprocess.run([PDFTOTEXT, "-enc", "UTF-8", str(path), "-"], capture_output=True, text=True,
                          timeout=120, check=True).stdout
    info = subprocess.run([PDFTOTEXT.replace("pdftotext", "pdfinfo"), str(path)], capture_output=True, text=True,
                          timeout=30).stdout
    if arxiv_title:
        return arxiv_title, url, clean_paper(text)
    title = re.search(r"^Title:\s*(.+)$", info, re.M)
    title = title.group(1).strip() if title and len(title.group(1).strip()) > 8 else ""
    if not title:
        title = next((l.strip() for l in text.splitlines() if 15 <= len(l.strip()) <= 200), path.stem)
    return title, url or path.name, clean_paper(text)


def clean_paper(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # undo end-of-line hyphenation
    text = text.replace("\f", "\n")
    cut = [m.start() for m in re.finditer(r"\n\s*(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n", text)]
    if cut and cut[-1] > len(text) * 0.4:
        text = text[:cut[-1]]
    paragraphs = [normalize(p) for p in re.split(r"\n\s*\n", text)]
    # drop page numbers, captions-only fragments, equations and table rows
    return "\n".join(p for p in paragraphs if len(p) > 60 and len(re.findall(r"[a-zA-Z]{3,}", p)) > 8)


def chunks(body, size=4500):
    sentences = re.split(r"(?<=[.!?])\s+", body)
    out, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) > size and current:
            out.append(current)
            current = ""
        current += sentence + " "
    if current.strip():
        out.append(current)
    return out


def spread_order(count):
    """Chunk visiting order spread across the paper (intro, method, results) instead of front pages only."""
    return sorted(range(count), key=lambda i: (i * 7919) % count) if count > 1 else [0]


def run_paper(source, deck=None, max_words=None):
    config = load_config()
    deck = deck or config["paperDeckName"]
    max_words = max_words or config["paperMaxWords"]
    api_key = load_env().get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY missing in .env")
    title, url, body = paper_text(source)
    if len(body) < 500:
        raise RuntimeError("could not extract enough text from the paper (scanned PDF?)")
    db = connect()
    known = {r[0] for r in db.execute("SELECT word_key FROM cards")}
    day = now_kst().date().isoformat()
    parts = chunks(body)
    # spread picks across the paper (intro, method, results, discussion) rather than the first pages only
    order = spread_order(len(parts))
    post = {"subreddit": "paper", "title": title[:200], "url": url}
    added, skipped_known = [], 0
    for index in order[:10]:
        if len(added) >= max_words:
            break
        try:
            items = call_llm(config, api_key, title, parts[index], sorted(known)[-300:],
                             min(8, max_words - len(added) + 3), kind="research paper excerpt")
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as error:
            log(f"LLM failed for chunk {index}: {error}")
            continue
        per_chunk = 0
        for item in items:
            if len(added) >= max_words or per_chunk >= 4:
                break
            if not valid_item(item, parts[index]):
                continue
            key = word_key(item["word"])
            if not key:
                continue
            if key in known:
                skipped_known += 1
                continue
            insert_card(db, key, item, post, day, "paper", deck)
            known.add(key)
            added.append({"word": item["word"].strip(), "cefr": str(item.get("cefr", "")).upper()[:2],
                          "meaning": item["definition"].strip()})
            per_chunk += 1
        db.commit()
        log(f"chunk {index + 1}/{len(parts)}: +{per_chunk}")
    pushed, ok = push_pending(db, config)
    return {"title": title, "deck": deck, "added": added, "skippedAlreadyKnown": skipped_known,
            "pushedToAnki": pushed, "ankiAvailable": ok, "synced": ok,
            "pending": db.execute("SELECT COUNT(*) FROM cards WHERE anki_note_id IS NULL").fetchone()[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="collect only; do not push to Anki or briefing")
    parser.add_argument("--force", action="store_true", help="run even when disabled")
    parser.add_argument("--paper", metavar="PDF_OR_URL", help="extract words from one paper instead of Reddit")
    parser.add_argument("--deck", help="deck for --paper (default: paperDeckName)")
    parser.add_argument("--max", type=int, help="max new words for --paper")
    args = parser.parse_args()
    if args.paper:
        try:
            result = run_paper(args.paper, args.deck, args.max)
        except Exception as error:
            print(json.dumps({"error": f"{type(error).__name__}: {error}"[:300]}, ensure_ascii=False))
            sys.exit(1)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(0)
    sys.exit(run(dry_run=args.dry_run, force=args.force))
