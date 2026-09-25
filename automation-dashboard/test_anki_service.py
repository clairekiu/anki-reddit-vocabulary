import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import anki_service


class AnkiServiceTests(unittest.TestCase):
    def test_subreddits_saved_cleaned_and_other_keys_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "config.json").write_text('{"deckName": "English::reddit-vocab"}')
            with patch.object(anki_service, "ROOT", root), patch.object(anki_service, "CONFIG", root / "config.json"), \
                    patch.object(anki_service, "DISABLED", root / "disabled"):
                result = anki_service.update({"enabled": True, "subreddits": ["r/robotics", "math", "Math"]})
                saved = json.loads((root / "config.json").read_text())
            self.assertEqual(saved, {"deckName": "English::reddit-vocab", "subreddits": ["robotics", "math"]})
            self.assertEqual(result["subreddits"], ["robotics", "math"])

    def test_invalid_subreddits_rejected(self):
        for bad in ([], ["../etc"], ["a"], ["x"] * 21):
            with self.assertRaises(ValueError):
                anki_service._save_subreddits(bad)


if __name__ == "__main__":
    unittest.main()
