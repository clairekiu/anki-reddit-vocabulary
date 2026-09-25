"""Quality filters shared by the daily Reddit run, paper mode and the eval script.

screen() applies, in order:
  schema -> non_english -> code_term -> cefr -> basic_expression -> not_verbatim -> word_missing
  -> LLM judge for multi-word expressions and for single words missing from the dictionary
     (generic_phrase / not_idiomatic / basic_expression / code_term / non_english)
  -> duplicate (same normalized form) / stem_duplicate (same Porter stems or phrase containment)
  -> cap (max words per text)
"""
import json
import re
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

DICTIONARY = Path("/usr/share/dict/words")

# Modern / technical words missing from the 1934 Webster word list.
EXTRA_WORDS = {
    "online", "offline", "email", "software", "hardware", "internet", "website", "webpage", "debug", "debugging",
    "dataset", "datasets", "workflow", "startup", "smartphone", "blog", "app", "apps", "upvote", "subreddit",
    "okay", "ok", "laptop", "desktop", "download", "upload", "login", "logout", "username", "password",
    "wifi", "robotics", "robot", "chatbot", "cyber", "cybersecurity", "firmware", "middleware", "bandwidth",
    "latency", "runtime", "codebase", "pipeline", "benchmark", "benchmarks", "scalable", "scalability",
    "modular", "modularity", "actuator", "actuators", "lidar", "radar", "sensor", "sensors", "gigabyte",
    "megabyte", "pixel", "pixels", "algorithm", "algorithmic", "iterate", "iterative", "heuristic", "heuristics",
    "workaround", "tradeoff", "tradeoffs", "boilerplate", "prototype", "prototyping", "teleoperation",
    "multimodal", "backpropagation", "hyperparameter", "hyperparameters", "tokenizer", "tokenization",
    "embedding", "embeddings", "finetune", "finetuning", "overfit", "overfitting", "underfitting",
    "denoise", "denoising", "discretize", "discretization", "parametrize", "parameterize", "vectorize",
    "quantize", "quantization", "regularization", "normalization", "eigenvalue", "eigenvector", "eigenvalues",
    "convolutional", "differentiable", "stochasticity", "nonlinear", "nonlinearity", "setpoint", "gimbal",
    "cheatsheet", "roadmap", "livestream", "podcast", "newsletter", "fanboy", "meme", "memes", "gameplay",
    "unblinded", "preprint", "preprints", "arxiv", "coauthor", "coauthors", "postdoc", "postdocs",
    "internship", "onboarding", "rebase", "refactor", "refactoring", "deprecate", "deprecated", "sandbox",
}

# Tool, library and product names that look like English (compounds of dictionary words).
TOOL_NAMES = {
    "github", "gitlab", "linux", "ubuntu", "python", "pytorch", "tensorflow", "numpy", "scipy", "pandas",
    "matlab", "simulink", "cmake", "pkgconf", "constexpr", "docker", "kubernetes", "gazebo", "rviz", "moveit",
    "opencv", "cuda", "arduino", "raspberry", "jupyter", "vscode", "chatgpt", "openai", "deepmind", "nvidia",
    "isaac", "mujoco", "ros", "rust", "golang", "javascript", "typescript", "html", "css", "json", "yaml",
}

# Common phrasal verbs and fixed phrases an intermediate learner already knows (matched on stems, so any
# tense or article variant is caught).
BASIC_EXPRESSIONS = [
    "look at", "look for", "look into", "look up", "look after", "look forward to", "look back",
    "figure out", "find out", "work out", "work on", "work together", "try out", "check out", "sort out",
    "take care of", "take part in", "take place", "take time", "take a look", "take into account",
    "make sense", "make sure", "make up", "make use of", "get stuck", "get through", "get into", "get out",
    "get up", "get back", "get rid of", "get used to", "get started", "come up with", "come across", "come up",
    "come back", "come from", "pick up", "set up", "give up", "go on", "go into", "go over", "go through",
    "go back", "carry out", "point out", "deal with", "rely on", "depend on", "based on", "due to",
    "as well as", "such as", "at least", "at this point", "at the same time", "all the time", "for example",
    "for instance", "in short", "in fact", "in general", "in detail", "in order to", "in charge of",
    "in addition", "in parallel", "in particular", "in the end", "in terms of", "sort of", "kind of",
    "a lot of", "a bit", "end up", "turn out", "turn around", "turn on", "turn off", "keep in mind",
    "keep up", "reach out", "catch up", "catch up with", "bring up", "break down", "build up", "put together",
    "hold up", "drop off", "run into", "run out of", "fit into", "dive into", "share with", "spend time on",
    "willing to", "supposed to", "be able to", "be for", "be limited to", "play around with", "back and forth",
    "from time to time", "by the way", "on the other hand", "first of all", "as long as", "as soon as",
    "so far", "right now", "more or less", "think about", "talk about", "care about", "fill in", "sign up",
    "log in", "show up", "move on", "move toward", "wait for", "wait on", "ask for", "pay attention to",
    "focus on", "belong to", "respond to", "refer to", "lead to", "result in", "consist of", "agree with",
    "start with", "work with", "help with", "compared to", "up to", "open source",
]

JUDGE_PROMPT = """You are screening English vocabulary flashcards for a Korean engineering graduate student (upper-intermediate English).
Classify each numbered expression as used in its sentence.

For a SINGLE WORD use one of:
- "english_word": a real English word, including technical terms, derived forms and loanwords normally used in English
- "non_english": a word from another language that is not normally used in English (e.g. "errático")
- "code_or_tool": code identifier, command, file format, library, tool, product or company name (e.g. "constexpr", "pkgconf")

For a MULTI-WORD expression use one of:
- "idiom": figurative or fixed expression whose meaning is not the sum of its words (e.g. "rabbit hole", "iron out", "rule of thumb", "talk shop")
- "nonobvious_collocation": conventional combination or phrasal verb a learner would not guess or produce (e.g. "fall short of", "at the expense of", "bear in mind", "shed light on")
- "technical_term": established domain term with a specific technical meaning (e.g. "receding horizon control", "fault tolerance", "type erasure")
- "basic_expression": common phrasal verb or phrase an intermediate learner already knows (e.g. "look at", "figure out", "make sense", "come up with", "in short", "get stuck")
- "generic_phrase": literal, compositional phrase whose meaning is obvious from its words (e.g. "remote desktop", "share my screen", "production line", "interview prep", "industry demand")
- "code_or_tool": code identifier, library, tool, product or company name
- "non_english": not an English expression

Return JSON only: {{"verdicts": [{{"i": 1, "category": "..."}}]}} with one verdict per numbered expression.

{entries}"""

FUNCTION_WORDS = {
    "the", "a", "an", "of", "to", "in", "is", "are", "was", "were", "and", "or", "for", "with", "that", "this",
    "it", "on", "we", "i", "you", "be", "as", "by", "not", "can", "from", "at", "but", "have", "has", "they",
    "which", "if", "so", "our", "my", "your", "their", "its", "these", "those", "there", "what", "how", "will",
    "would", "could", "should", "than", "then", "when", "where", "who", "why", "all", "some", "any", "more",
    "most", "just", "also", "very", "into", "through", "across", "about", "after", "before", "every", "each",
    "never", "only", "other", "such", "do", "does", "did", "been", "being", "he", "she", "him", "her", "them",
    "us", "me", "no", "yes", "while", "because", "using", "used", "get", "got", "much", "many", "even",
}
# Frequent function words of other Latin-script languages (Spanish, Portuguese, French, Italian, German).
FOREIGN_WORDS = {
    "el", "la", "los", "las", "del", "es", "que", "por", "para", "una", "uno", "con", "como", "pero", "muy",
    "est", "les", "des", "une", "pas", "avec", "dans", "pour", "sur", "il", "che", "di", "della", "sono",
    "der", "die", "das", "und", "ist", "nicht", "ein", "eine", "mit", "auf", "fur", "sie", "wir", "ich",
    "nao", "em", "um", "uma", "mais", "sao", "voce", "de", "se", "lo", "su", "sin", "hay", "esto", "esta", "este",
    "estoy", "tengo", "puede", "cuando", "donde", "tambien",
}
PLACEHOLDERS = {"someone", "somebody", "something", "sb", "sth", "oneself", "yourself", "my", "your",
                "his", "her", "their", "our", "its", "a", "an", "the"}
# Particles and function words do not make two phrases "the same expression" on their own.
LIGHT_WORDS = FUNCTION_WORDS | {"up", "down", "out", "off", "over", "into", "through", "about", "around",
                                "away", "back", "under", "upon", "onto", "toward", "towards", "get", "go"}
IRREGULAR = {
    "went": "go", "gone": "go", "goes": "go", "got": "get", "gotten": "get", "took": "take", "taken": "take",
    "made": "make", "came": "come", "threw": "throw", "thrown": "throw", "drew": "draw", "drawn": "draw",
    "fell": "fall", "fallen": "fall", "caught": "catch", "spun": "spin", "brought": "bring", "thought": "think",
    "sold": "sell", "held": "hold", "ran": "run", "wrote": "write", "written": "write", "broke": "break",
    "broken": "break", "did": "do", "done": "do", "does": "do", "had": "have", "has": "have", "was": "be",
    "were": "be", "been": "be", "is": "be", "are": "be", "am": "be", "found": "find", "gave": "give",
    "given": "give", "kept": "keep", "left": "leave", "led": "lead", "stood": "stand", "struck": "strike",
    "understood": "understand", "wore": "wear", "worn": "wear", "built": "build", "sought": "seek",
    "bore": "bear", "borne": "bear", "dealt": "deal", "felt": "feel", "meant": "mean", "said": "say",
    "saw": "see", "seen": "see", "told": "tell", "knew": "know", "known": "know", "began": "begin",
    "begun": "begin", "rose": "rise", "risen": "rise", "shook": "shake", "shaken": "shake", "wound": "wind",
    "hung": "hang", "stuck": "stick", "dug": "dig", "fed": "feed", "met": "meet", "paid": "pay", "laid": "lay",
    "lay": "lie", "lain": "lie", "bent": "bend", "sent": "send", "spent": "spend", "lost": "lose", "won": "win",
    "slid": "slide", "swung": "swing", "tore": "tear", "torn": "tear", "drove": "drive", "driven": "drive",
    "chose": "choose", "chosen": "choose", "froze": "freeze", "frozen": "freeze", "grew": "grow", "grown": "grow",
    "blew": "blow", "blown": "blow", "flew": "fly", "flown": "fly", "hid": "hide", "hidden": "hide",
}
SUFFIXES = [  # (suffix, stems to try after removing it)
    ("ification", ["", "e", "y"]), ("ization", ["ize", "e", ""]), ("ibility", ["", "e", "ible"]),
    ("ability", ["", "e", "able"]), ("ations", ["e", ""]), ("ation", ["e", ""]), ("ically", ["ic", ""]),
    ("ities", ["", "e", "y"]), ("ments", [""]), ("ment", [""]), ("ness", [""]), ("less", [""]), ("ship", [""]),
    ("hood", [""]), ("wise", [""]), ("like", [""]), ("izing", ["", "e", "ize"]), ("ized", ["", "e", "ize"]),
    ("izer", ["", "e", "ize"]), ("ize", ["", "e"]), ("ise", ["", "e"]), ("ical", ["", "e", "y", "ic"]),
    ("ally", ["", "e", "al"]), ("able", ["", "e"]), ("ible", ["", "e"]), ("ance", ["", "e"]), ("ence", ["", "e"]),
    ("ity", ["", "e"]), ("ied", ["y"]), ("ies", ["y"]), ("ions", ["", "e"]), ("ion", ["", "e"]), ("ily", ["y"]), ("ify", ["", "e", "y"]), ("ing", ["", "e"]),
    ("ers", ["", "e"]), ("ors", ["", "e"]), ("est", ["", "e"]), ("ism", ["", "e"]), ("ist", ["", "e"]),
    ("ous", ["", "e"]), ("ive", ["", "e"]), ("ful", [""]), ("ant", ["", "e"]), ("ent", ["", "e"]),
    ("ery", ["", "e"]), ("es", ["", "e"]), ("ed", ["", "e"]), ("ly", ["", "le"]), ("er", ["", "e"]),
    ("or", ["", "e"]), ("al", ["", "e"]), ("ic", ["", "e", "y"]), ("en", ["", "e"]), ("s", [""]), ("y", ["", "e"]),
]
PREFIXES = [
    "counter", "electro", "thermo", "pseudo", "hyper", "inter", "intra", "micro", "macro", "multi", "super",
    "under", "ultra", "cross", "trans", "neuro", "cyber", "photo", "quasi", "semi", "anti", "auto", "self",
    "over", "post", "tele", "nano", "meta", "mega", "mini", "poly", "mono", "back", "down", "non", "pre", "sub",
    "out", "mis", "dis", "bio", "geo", "tri", "re", "un", "de", "co", "bi", "up",
]


@lru_cache(maxsize=1)
def _dictionary():
    try:
        words = {w.strip().lower() for w in DICTIONARY.read_text(encoding="utf-8", errors="ignore").split()}
    except OSError:
        words = set()
    return frozenset(words | EXTRA_WORDS)


@lru_cache(maxsize=50000)
def english_token(token, depth=0):
    """True if token is an English dictionary word or a regular derivation/compound of one."""
    token = token.lower()
    words = _dictionary()
    if token in words:
        return True
    if depth >= 2 or len(token) < 4:
        return False
    for suffix, stems in SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= (2 if suffix in ("ies", "ied") else 3):
            base = token[: -len(suffix)]
            if any(english_token(base + tail, depth + 1) for tail in stems):
                return True
            if len(base) >= 4 and base[-1] == base[-2] and english_token(base[:-1], depth + 1):  # running -> run
                return True
    for prefix in PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= 3 and english_token(token[len(prefix):], depth + 1):
            return True
    return any(token[:i] in words and token[i:] in words for i in range(3, len(token) - 2))  # workflow


def sentence_words(sentence):
    """Plain words of a sentence: URLs and code-like tokens dropped, surrounding punctuation stripped."""
    words = []
    for raw in re.sub(r"https?://\S+|www\.\S+", " ", sentence).split():
        token = raw.strip(".,;:!?()[]{}\"'\u201c\u201d\u2018\u2019*-")
        if token and not re.search(r"[0-9_:/\\#<>{}\[\]=@+.]", token):
            words.extend(part for part in re.split(r"[-'\u2019]", token) if part)
    return words


def looks_english(sentence):
    """Latin script, not dominated by foreign function words, and mostly English dictionary words.

    Proper nouns (capitalized after the first word, or mixed case like SQLite) and acronyms are not counted,
    so English sentences full of product names are not mistaken for another language.
    """
    words = [w for w in sentence_words(sentence) if w.isalpha()]
    letters = "".join(words)
    if not letters or sum(1 for c in letters if c.isascii()) / len(letters) < 0.9:
        return False
    lower = [w.lower() for w in words]
    english, foreign = sum(t in FUNCTION_WORDS for t in lower), sum(t in FOREIGN_WORDS for t in lower)
    if foreign >= 2 and foreign > english:
        return False
    common = [w for i, w in enumerate(words)
              if not (w.isupper() and len(w) >= 2)                      # acronyms
              and not (i > 0 and w[0].isupper())                         # proper nouns mid-sentence
              and not any(c.isupper() for c in w[1:])]                   # SQLite, iPhone
    content = [w.lower() for w in common if len(w) >= 3 and w.lower() not in FUNCTION_WORDS]
    if not content:
        return english > 0
    ratio = sum(english_token(t) for t in content) / len(content)
    return ratio >= 0.85 or (english > 0 and ratio >= 0.6)


def headword_tokens(word):
    word = re.sub(r"\((?:someone|somebody|something|sb|sth|one's|one)\)", " ", word.lower())
    word = re.sub(r"\bone's\b", " someone ", word.replace("\u2019", "'"))
    return re.findall(r"[a-z]+", word.replace("'s", " "))


def headword_problem(word):
    """'non_english' or 'code_term' when the headword alone is clearly not an English word, else None."""
    stripped = re.sub(r"\((?:someone|somebody|something|sb|sth|one's|one)\)", " ", word.strip())
    if any(c.isalpha() and not c.isascii() for c in stripped):
        return "non_english"
    if re.search(r"[0-9_:/\\#<>(){}\[\]=+*@$%^&|~`.]|[a-z][A-Z]", stripped):
        return "code_term"
    if any(len(t) >= 2 and t.isupper() for t in re.findall(r"[A-Za-z]+", stripped)):
        return "code_term"  # acronyms
    tokens = headword_tokens(stripped)
    if not tokens or stripped.lower().replace("-", "").replace(" ", "") in TOOL_NAMES:
        return "code_term"
    if any(t in TOOL_NAMES for t in tokens):
        return "code_term"
    return None


def unknown_tokens(word):
    """Headword tokens missing from the dictionary; such words go to the LLM judge."""
    return [t for t in headword_tokens(word) if len(t) > 2 and t not in PREFIXES and not english_token(t)]


# ---------- stems (Porter 1980) ----------

def _cons(word, i):
    if word[i] in "aeiou":
        return False
    if word[i] == "y":
        return i == 0 or not _cons(word, i - 1)
    return True


def _measure(stem):
    form = "".join("c" if _cons(stem, i) else "v" for i in range(len(stem)))
    return re.sub(r"(.)\1+", r"\1", form).count("vc")


def _has_vowel(stem):
    return any(not _cons(stem, i) for i in range(len(stem)))


def _double(word):
    return len(word) >= 2 and word[-1] == word[-2] and _cons(word, len(word) - 1)


def _cvc(word):
    return (len(word) >= 3 and _cons(word, len(word) - 3) and not _cons(word, len(word) - 2)
            and _cons(word, len(word) - 1) and word[-1] not in "wxy")


def _replace(word, rules, min_measure):
    for suffix, repl in sorted(rules, key=lambda r: -len(r[0])):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            return stem + repl if _measure(stem) > min_measure else word
    return word


STEP2 = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"),
         ("abli", "able"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"), ("ization", "ize"),
         ("ation", "ate"), ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
         ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")]
STEP3 = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", "")]
STEP4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "ou", "ism",
         "ate", "iti", "ous", "ive", "ize"]


@lru_cache(maxsize=50000)
def porter(word):
    word = word.lower()
    if len(word) <= 2:
        return word
    # step 1a
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    # step 1b
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    else:
        for suffix in ("ed", "ing"):
            if word.endswith(suffix) and _has_vowel(word[: -len(suffix)]):
                word = word[: -len(suffix)]
                if word.endswith(("at", "bl", "iz")):
                    word += "e"
                elif _double(word) and word[-1] not in "lsz":
                    word = word[:-1]
                elif _measure(word) == 1 and _cvc(word):
                    word += "e"
                break
    # step 1c
    if word.endswith("y") and _has_vowel(word[:-1]):
        word = word[:-1] + "i"
    word = _replace(word, STEP2, 0)
    word = _replace(word, STEP3, 0)
    # step 4
    for suffix in sorted(STEP4, key=len, reverse=True):
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > 1 and (suffix != "ion" or stem.endswith(("s", "t"))):
                word = stem
            break
    # step 5
    if word.endswith("e"):
        stem = word[:-1]
        if _measure(stem) > 1 or (_measure(stem) == 1 and not _cvc(stem)):
            word = stem
    if _measure(word) > 1 and _double(word) and word.endswith("l"):
        word = word[:-1]
    return word


def lemma_tokens(word):
    """Stems of the meaningful tokens: articles/pronoun placeholders dropped, irregular verbs regularized.

    A final "e" is ignored because Porter keeps it on short words only (tie -> tie, tied -> ti).
    """
    tokens = [IRREGULAR.get(t, t) for t in headword_tokens(word) if t not in PLACEHOLDERS]
    return tuple(porter(t)[:-1] if porter(t).endswith("e") and len(porter(t)) > 2 else porter(t) for t in tokens)


BASIC_KEYS = frozenset(lemma_tokens(p) for p in BASIC_EXPRESSIONS)
LIGHT_STEMS = frozenset(porter(w) for w in LIGHT_WORDS)


def word_key(word):
    return re.sub(r"[^a-z0-9 ]", "", word.lower().replace("-", " ")).strip()


def content_bigrams(lemma):
    content = [t for t in lemma if t not in LIGHT_STEMS]
    return {(a, b) for a, b in zip(content, content[1:])}


class Seen:
    """Words already learned. check() -> None, 'duplicate' (same form) or 'stem_duplicate'."""

    def __init__(self, words=()):
        self.words, self.exact, self.lemmas, self.phrases, self.bigrams = [], set(), set(), [], set()
        for word in words:
            self.add(word)

    def check(self, word):
        if word_key(word) in self.exact:
            return "duplicate"
        lemma = lemma_tokens(word)
        if not lemma:
            return None
        if lemma in self.lemmas:
            return "stem_duplicate"
        # same expression family: shared consecutive content stems ("go down a rabbit hole" vs
        # "fall into the rabbit hole"), or containment ("rabbit hole" in either) with 2+ content stems
        if content_bigrams(lemma) & self.bigrams:
            return "stem_duplicate"
        tokens = set(lemma)
        for other in self.phrases:
            small, large = (tokens, other) if len(tokens) <= len(other) else (other, tokens)
            if len(small - LIGHT_STEMS) >= 2 and small <= large:
                return "stem_duplicate"
        return None

    def add(self, word):
        self.words.append(word)
        self.exact.add(word_key(word))
        lemma = lemma_tokens(word)
        if lemma:
            self.lemmas.add(lemma)
            if len(set(lemma)) >= 2:
                self.phrases.append(set(lemma))
            self.bigrams |= content_bigrams(lemma)


# ---------- the filter chain ----------

def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def sentence_in_source(item, body):
    # The example must be a real sentence from the source text (guards against invented examples).
    sentence = normalize(item["reddit_sentence"])
    return len(sentence) >= 20 and sentence.lower() in normalize(body).lower()


def is_multiword(word):
    """Space-separated expression; hyphenated compounds (cost-effective) count as single words."""
    word = re.sub(r"\((?:someone|somebody|something|sb|sth|one's|one)\)", " ", word)
    return len(word.split()) >= 2


def reject_reason(item, body):
    """First failing deterministic check, or None."""
    if not isinstance(item, dict):
        return "schema"
    fields = ("word", "definition", "reddit_sentence", "academic_example")
    if not all(isinstance(item.get(f), str) and item[f].strip() for f in fields):
        return "schema"
    word = item["word"].strip()
    problem = headword_problem(word)
    if problem:
        return problem
    if not looks_english(item["reddit_sentence"]):
        return "non_english"
    # Hard single words only; multi-word expressions may be B2 (then the judge must call them idiomatic).
    level = str(item.get("cefr", "C1")).upper()[:2]
    if level not in ("C1", "C2") and not (level == "B2" and is_multiword(word)):
        return "cefr"
    if is_multiword(word) and lemma_tokens(word) in BASIC_KEYS:
        return "basic_expression"
    if not sentence_in_source(item, body):
        return "not_verbatim"
    surface = (item.get("surface") or word).lower()
    stem = word.lower().split()[0][:max(4, len(word.split()[0]) - 2)]
    sentence = normalize(item["reddit_sentence"]).lower()
    return None if surface in sentence or stem in sentence else "word_missing"


def judge_phrases(config, api_key, entries, usage=None):
    """LLM verdicts for multi-word expressions, aligned with entries; None if the call failed."""
    lines = "\n".join(f'{i}. "{word}" - sentence: "{sentence}"' for i, (word, sentence) in enumerate(entries, 1))
    payload = {
        "model": config["llmModel"],
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(entries=lines)}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    request = urllib.request.Request(
        config["llmUrl"], data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            reply = json.loads(response.read())
        verdicts = json.loads(reply["choices"][0]["message"]["content"]).get("verdicts", [])
    except (urllib.error.URLError, OSError, KeyError, ValueError, TypeError, AttributeError):
        return None
    if usage is not None:
        for key in ("prompt_tokens", "completion_tokens"):
            usage["judge_" + key] = usage.get("judge_" + key, 0) + reply.get("usage", {}).get(key, 0)
    if not isinstance(verdicts, list):
        return None
    by_index = {v.get("i"): str(v.get("category", "")) for v in verdicts if isinstance(v, dict)}
    return [by_index.get(i, "") for i in range(1, len(entries) + 1)]


def judge_reason(level, category, multiword):
    if category == "code_or_tool":
        return "code_term"
    if category == "non_english":
        return "non_english"
    if not multiword:  # single words reach the judge only to confirm they are English
        return None if category in ("english_word", "technical_term", "idiom", "nonobvious_collocation") \
            else "judge_error"
    if category == "basic_expression":
        return "basic_expression"
    if category == "generic_phrase":
        return "generic_phrase"
    if category in ("idiom", "nonobvious_collocation"):
        return None
    if category == "technical_term":
        return None if level in ("C1", "C2") else "not_idiomatic"
    return "judge_error"  # missing or unknown verdict: reject rather than guess


def screen(items, body, seen, config, api_key, cap, usage=None, judge=None):
    """Run the full filter chain on one LLM reply (the judge is called once per reply, only if needed).

    Returns (accepted_items, reasons, verdicts): reasons[i] is None for accepted items, verdicts[i] is the
    judge category for judged multi-word items. `seen` is updated with accepted words.
    """
    judge = judge or judge_phrases
    reasons = [reject_reason(item, body) for item in items]
    verdicts = [""] * len(items)
    pending = [i for i, r in enumerate(reasons)
               if r is None and (is_multiword(items[i]["word"]) or unknown_tokens(items[i]["word"]))]
    if pending:
        result = judge(config, api_key, [(items[i]["word"].strip(), normalize(items[i]["reddit_sentence"]))
                                         for i in pending], usage)
        for position, i in enumerate(pending):
            verdicts[i] = result[position] if result else ""
            level = str(items[i].get("cefr", "C1")).upper()[:2]
            reasons[i] = judge_reason(level, verdicts[i], is_multiword(items[i]["word"]))
    accepted = []
    for i, item in enumerate(items):
        if reasons[i] is not None:
            continue
        reasons[i] = seen.check(item["word"].strip())
        if reasons[i]:
            continue
        if len(accepted) >= cap:
            reasons[i] = "cap"
            continue
        seen.add(item["word"].strip())
        accepted.append(item)
    return accepted, reasons, verdicts
