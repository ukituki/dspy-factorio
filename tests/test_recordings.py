import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dspy_factorio.recordings import load_saved_episodes, replay_html, save_episode


class RecordingTests(unittest.TestCase):
    def episode(self, directory):
        return {"id": "run1", "model": "test", "module": "Predict", "goal": "Make plates",
                "status": "running", "elapsed": 1.0, "directory": str(directory),
                "steps": [{"index": 1, "label": "Step 1", "observation": "<script>bad()</script>",
                           "program": "print('hello')", "reward": 2, "done": False}]}

    def test_checkpoint_reloads_and_html_is_self_contained(self):
        with TemporaryDirectory() as root:
            directory = Path(root) / "run1"
            directory.mkdir()
            image = directory / "step.png"
            image.write_bytes(b"test png bytes")
            episode = self.episode(directory)
            episode["steps"][0]["image"] = str(image)
            output = save_episode(episode)
            html = output.read_text()
            self.assertIn("data:image/png;base64,", html)
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;", html)
            self.assertNotIn('src="http', html)
            episodes, issues = load_saved_episodes(Path(root))
            self.assertEqual(issues, [])
            self.assertEqual(episodes[0]["status"], "saved checkpoint")
            self.assertEqual(episodes[0]["steps"][0]["reward"], 2)
            self.assertEqual(json.loads((directory / "episode.json").read_text())["status"], "running")

    def test_copied_recordings_resolve_images_locally(self):
        with TemporaryDirectory() as root:
            directory = Path(root) / "copied"
            episode = self.episode(directory)
            episode["steps"][0]["image"] = "/old/computer/step.png"
            save_episode(episode)
            (directory / "step.png").write_bytes(b"png")
            episodes, issues = load_saved_episodes(Path(root))
            self.assertFalse(issues)
            self.assertEqual(episodes[0]["steps"][0]["image"], str(directory / "step.png"))

    def test_corrupt_recording_does_not_hide_good_runs(self):
        with TemporaryDirectory() as root:
            save_episode(self.episode(Path(root) / "good"))
            bad = Path(root) / "bad"
            bad.mkdir()
            (bad / "episode.json").write_text("{", encoding="utf-8")
            episodes, issues = load_saved_episodes(Path(root))
            self.assertEqual(len(episodes), 1)
            self.assertEqual(len(issues), 1)
            self.assertIn("Unavailable", replay_html(episodes[0]))


if __name__ == "__main__":
    unittest.main()
