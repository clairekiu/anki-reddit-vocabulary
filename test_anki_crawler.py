import tempfile
import unittest
from pathlib import Path

import anki_crawler as ac

FEED = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>t3_abc</id><title>MPC question</title><link href="https://reddit.com/x"/>
<content type="html">&lt;p&gt;We need to formulate this trajectory planning problem as a quadratic program.&lt;/p&gt;&lt;pre&gt;code&lt;/pre&gt; submitted by /u/me</content>
</entry></feed>"""


class CrawlerTests(unittest.TestCase):
    def test_parse_feed_strips_html_code_and_footer(self):
        post = ac.parse_feed(FEED, "robotics")[0]
        self.assertEqual(post["id"], "t3_abc")
        self.assertIn("formulate this trajectory", post["body"])
        self.assertNotIn("code", post["body"])
        self.assertNotIn("submitted", post["body"])

    def test_valid_item_requires_verbatim_sentence_with_word(self):
        body = "We need to formulate this trajectory planning problem as a quadratic program. Other text."
        item = {"word": "formulate", "definition": "정립하다", "academic_example": "x",
                "reddit_sentence": "We need to formulate this trajectory planning problem as a quadratic program."}
        self.assertTrue(ac.valid_item(item, body))
        self.assertFalse(ac.valid_item({**item, "reddit_sentence": "We formulate an invented sentence here."}, body))
        self.assertFalse(ac.valid_item({**item, "word": "leverage"}, body))
        self.assertFalse(ac.valid_item({**item, "cefr": "B2"}, body))
        self.assertTrue(ac.valid_item({**item, "cefr": "C1"}, body))

    def test_inflected_form_is_accepted_and_highlighted(self):
        body = "The cost function was formulated to minimize energy consumption overall."
        item = {"word": "formulate", "surface": "formulated", "definition": "d", "academic_example": "x",
                "reddit_sentence": "The cost function was formulated to minimize energy consumption overall."}
        self.assertTrue(ac.valid_item(item, body))
        self.assertIn("<b>formulated</b>", ac.highlight(item["reddit_sentence"], item))

    def test_word_key_dedupes_case_and_hyphen(self):
        self.assertEqual(ac.word_key("Trade-off"), ac.word_key("trade off"))

    def test_collect_never_duplicates_known_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = ac.connect(Path(tmp) / "s.db")
            post = {"id": "p1", "subreddit": "robotics", "title": "t", "url": "u",
                    "body": "We must mitigate drift. Engineers should mitigate drift carefully. We can leverage priors well here."}
            items = [
                {"word": "mitigate", "definition": "완화", "academic_example": "a", "reddit_sentence": "We must mitigate drift."},
                {"word": "Mitigate", "definition": "완화", "academic_example": "a", "reddit_sentence": "Engineers should mitigate drift carefully."},
                {"word": "leverage", "definition": "활용", "academic_example": "a", "reddit_sentence": "We can leverage priors well here."},
            ]
            orig_fetch, orig_llm = ac.fetch_posts, ac.call_llm
            ac.fetch_posts = lambda config: [post]
            ac.call_llm = lambda *a: items
            try:
                added = ac.collect(db, dict(ac.DEFAULT_CONFIG), "k", 10)
                again = ac.collect(db, dict(ac.DEFAULT_CONFIG), "k", 10)  # post already seen
            finally:
                ac.fetch_posts, ac.call_llm = orig_fetch, orig_llm
            self.assertEqual(added, 2)
            self.assertEqual(again, 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM cards").fetchone()[0], 2)


if __name__ == "__main__":
    unittest.main()
