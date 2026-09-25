import unittest

import vocab_filters as vf

CONFIG = {"llmModel": "m", "llmUrl": "u"}
BODY = ("We had to go down a rabbit hole to find the bug. Please share my screen during the call. "
        "I finally figured out the controller gains after a week of tuning. "
        "The planner degrades gracefully and remains tractable under heavy load. "
        "El comportamiento del robot es errático cuando la batería baja.")


def item(word, sentence, cefr="C1"):
    return {"word": word, "cefr": cefr, "definition": "d", "academic_example": "a", "reddit_sentence": sentence}


def fake_judge(verdicts):
    def judge(config, api_key, entries, usage=None):
        return [verdicts.get(word, "") for word, _ in entries]
    return judge


class FilterTests(unittest.TestCase):
    def test_non_english_headword_and_sentence(self):
        self.assertEqual(vf.reject_reason(item("errático", "El comportamiento del robot es errático cuando la batería baja."), BODY), "non_english")
        self.assertFalse(vf.looks_english("Das ist ein sehr schwieriges Problem für uns."))
        self.assertTrue(vf.looks_english("Battery life would last easily more than 1 hour."))
        self.assertTrue(vf.looks_english("The drone flies very erratically:"))
        self.assertTrue(vf.looks_english("Ventuno Q is new Robotics/Edge AI SBC from Arudino with Qualcomm SoC, "
                                         "comparable to Raspberry pi 5 and Nvidia Jetson."))
        for spanish in ("Estoy atascado con mi proyecto de dron simulado y necesito ayuda.",
                        "Mi sospecha principal:", "Sin embargo, no sé cómo aislarlo para confirmarlo.",
                        "Ajustar ganancias del PID y horizontes del MPC (ambos fallan igual)."):
            self.assertFalse(vf.looks_english(spanish), spanish)

    def test_code_terms(self):
        for word in ("pkgconf", "constexpr", "std::vector", "CMake", "GitHub", "IMU"):
            self.assertEqual(vf.headword_problem(word), "code_term", word)
        for word in ("denoising", "teleoperation", "fault-tolerant", "take (something) in the wrong direction"):
            self.assertIsNone(vf.headword_problem(word), word)

    def test_stem_and_phrase_duplicates(self):
        seen = vf.Seen(["tangential", "instantiate", "fault tolerance", "rabbit hole", "catch one's eye", "work out"])
        for word in ("tangentially", "instantiation", "fault-tolerant", "go down a rabbit hole",
                     "fall into the rabbit hole", "caught my eye", "worked out"):
            self.assertEqual(seen.check(word), "stem_duplicate", word)
        self.assertEqual(seen.check("Tangential"), "duplicate")
        self.assertEqual(vf.Seen(["tie to"]).check("tied to"), "stem_duplicate")
        self.assertEqual(vf.Seen(["go down a rabbit hole"]).check("fall into the rabbit hole"), "stem_duplicate")
        different = vf.Seen(["catch one's eye", "drawn out", "overshoot", "imitation learning", "fixed point"])
        for word in ("catch one's attention", "draw from", "oversight", "steep learning curve", "point-set topology"):
            self.assertIsNone(different.check(word), word)
        self.assertEqual(vf.Seen(["use"]).check("used"), "stem_duplicate")
        for word in ("work through", "square one", "tangent"):
            self.assertIsNone(seen.check(word), word)

    def test_basic_expressions_any_tense(self):
        for phrase in ("look at", "figured out", "taking care of", "in short", "makes sense", "got stuck", "came up with"):
            self.assertIn(vf.lemma_tokens(phrase), vf.BASIC_KEYS, phrase)
        self.assertNotIn(vf.lemma_tokens("shed light on"), vf.BASIC_KEYS)

    def test_porter_reference_pairs(self):
        pairs = {"tangential": "tangenti", "tangentially": "tangenti", "instantiation": "instanti",
                 "tolerance": "toler", "tolerant": "toler", "relational": "relat", "hopping": "hop"}
        for word, stem in pairs.items():
            self.assertEqual(vf.porter(word), stem, word)

    def test_screen_chain(self):
        items = [
            item("go down a rabbit hole", "We had to go down a rabbit hole to find the bug.", "B2"),
            item("share my screen", "Please share my screen during the call.", "B2"),
            item("figure out", "I finally figured out the controller gains after a week of tuning.", "B2"),
            item("degrade gracefully", "The planner degrades gracefully and remains tractable under heavy load.", "B2"),
            item("tractable", "The planner degrades gracefully and remains tractable under heavy load."),
            item("rabbit hole", "We had to go down a rabbit hole to find the bug.", "C1"),
        ]
        judge = fake_judge({"go down a rabbit hole": "idiom", "share my screen": "generic_phrase",
                            "degrade gracefully": "technical_term", "rabbit hole": "idiom"})
        kept, reasons, verdicts = vf.screen(items, BODY, vf.Seen(), CONFIG, "k", cap=4, judge=judge)
        self.assertEqual([i["word"] for i in kept], ["go down a rabbit hole", "tractable"])
        self.assertEqual(reasons, [None, "generic_phrase", "basic_expression", "not_idiomatic", None, "stem_duplicate"])
        self.assertEqual(verdicts[0], "idiom")

    def test_judge_failure_rejects_instead_of_guessing(self):
        items = [item("go down a rabbit hole", "We had to go down a rabbit hole to find the bug.", "B2")]
        kept, reasons, _ = vf.screen(items, BODY, vf.Seen(), CONFIG, "k", cap=4, judge=lambda *a, **k: None)
        self.assertEqual((kept, reasons), ([], ["judge_error"]))

    def test_unknown_single_word_goes_to_judge(self):
        body = "The response felt empathetic and clear to every reviewer on the panel."
        items = [item("empathetic", body)]
        kept, _, verdicts = vf.screen(items, body, vf.Seen(), CONFIG, "k", cap=4,
                                      judge=fake_judge({"empathetic": "english_word"}))
        self.assertEqual((len(kept), verdicts), (1, ["english_word"]))
        kept, reasons, _ = vf.screen([item("numerics", body.replace("empathetic", "numerics"))],
                                     body.replace("empathetic", "numerics"), vf.Seen(), CONFIG, "k", cap=4,
                                     judge=fake_judge({"numerics": "code_or_tool"}))
        self.assertEqual(reasons, ["code_term"])

    def test_hyphenated_words_are_single_words(self):
        self.assertFalse(vf.is_multiword("cost-effective"))
        self.assertTrue(vf.is_multiword("take (something) in the wrong direction"))
        body = "It literally happens when a thread gets pre-empted by the OS scheduler in practice."
        kept, reasons, verdicts = vf.screen([item("pre-empt", body)], body, vf.Seen(), CONFIG, "k", cap=4,
                                            judge=fake_judge({"pre-empt": "english_word"}))
        self.assertEqual(reasons, [None])
        body = "We need a simple and cost-effective mower design for this kind of lawn."
        _, reasons, _ = vf.screen([item("cost-effective", body, "B2")], body, vf.Seen(), CONFIG, "k", cap=4,
                                  judge=fake_judge({}))
        self.assertEqual(reasons, ["cefr"])  # B2 single word, same rule as before

    def test_cap(self):
        body = "Formulate it. We mitigate drift in practice. " * 1
        items = [item("formulate", "We formulate the problem as a convex program here."),
                 item("mitigate", "We mitigate drift in practice with care.")]
        text = " ".join(i["reddit_sentence"] for i in items)
        kept, reasons, _ = vf.screen(items, text, vf.Seen(), CONFIG, "k", cap=1)
        self.assertEqual(reasons, [None, "cap"])


if __name__ == "__main__":
    unittest.main()
