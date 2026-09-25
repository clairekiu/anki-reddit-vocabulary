"""Dashboard status and toggle for the daily Reddit -> Anki vocabulary crawler."""
import json
import os
import pathlib
import re
import subprocess
import time
from datetime import datetime, timedelta


ROOT = pathlib.Path.home() / "anki-crawling"
LABEL = "com.minjaeku.anki-crawling"
DISABLED = ROOT / "disabled"
CONFIG = ROOT / "config.json"
DEFAULT_SUBREDDITS = ["robotics", "MachineLearning", "ControlTheory", "ROS", "cpp", "compsci", "math"]
SUBREDDIT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]{1,20}")
BRIEFINGS = pathlib.Path.home() / "hermes-yuna/briefings"


def _read_summary():
    try:
        return json.loads((ROOT / "daily_summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _launchd():
    try:
        result = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/{LABEL}"], capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return {"installed": False, "lastExitCode": None}
    code = re.search(r"^\s*last exit code = (.+)$", result.stdout, re.MULTILINE)
    return {"installed": not bool(result.returncode), "lastExitCode": code.group(1) if code else None}


def _tail(name, limit=20):
    path = ROOT / name
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]


def _briefing(today):
    """Today's 21:00 Telegram message exactly as Hermes will send it."""
    try:
        return (BRIEFINGS / f"{today}.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _config():
    try:
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_subreddits(names):
    if not isinstance(names, list) or not 1 <= len(names) <= 20:
        raise ValueError("subreddits must be a list of 1-20 names")
    cleaned = []
    for name in names:
        name = str(name).strip().removeprefix("r/").removeprefix("/r/")
        if not SUBREDDIT.fullmatch(name):
            raise ValueError("invalid subreddit name")
        if name.lower() not in {n.lower() for n in cleaned}:
            cleaned.append(name)
    config = _config()
    config["subreddits"] = cleaned
    temporary = CONFIG.with_suffix(".tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(CONFIG)


def status():
    summary = _read_summary()
    today = datetime.now().date().isoformat()
    is_today = summary.get("date") == today
    now = datetime.now()
    next_run = now.replace(hour=20, minute=30, second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(days=1)
    return {
        "enabled": not DISABLED.exists(),
        "addedToday": summary.get("added_today", 0) if is_today else 0,
        "words": summary.get("words", []) if is_today else [],
        "pending": summary.get("pending", 0),
        "totalCards": summary.get("total_cards", 0),
        "error": summary.get("error", ""),
        "lastRun": summary.get("updated", 0),
        "nextRun": next_run.timestamp(),
        "launchd": _launchd(),
        "log": _tail("anki-crawling.log"),
        "briefing": _briefing(today),
        "subreddits": _config().get("subreddits") or DEFAULT_SUBREDDITS,
        "updated": time.time(),
    }


def update(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("enabled"), bool):
        raise ValueError("enabled must be boolean")
    if "subreddits" in payload:
        _save_subreddits(payload["subreddits"])
    if payload["enabled"]:
        DISABLED.unlink(missing_ok=True)
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        DISABLED.touch()
    if payload.get("action") == "run_now":
        subprocess.run(["launchctl", "kickstart", f"gui/{os.getuid()}/{LABEL}"], capture_output=True, timeout=3, check=False)
    elif payload.get("action", "save") != "save":
        raise ValueError("unsupported action")
    return status()
