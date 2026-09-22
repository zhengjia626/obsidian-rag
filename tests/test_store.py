import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from store import VectorStore
import unittest


class TestVectorStore(unittest.TestCase):

    def _store(self):
        s = VectorStore()
        s.add(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [{"t": "x"}, {"t": "y"}, {"t": "z"}],
        )
        return s

    def test_search_returns_nearest_first(self):
        s = self._store()
        res = s.search([1.0, 0.0, 0.0], top_k=3)
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0][1]["t"], "x")
        self.assertGreaterEqual(res[0][0], res[1][0])
        self.assertGreaterEqual(res[1][0], res[2][0])

    def test_search_top_k_limits(self):
        s = self._store()
        self.assertEqual(len(s.search([1.0, 0.0, 0.0], top_k=2)), 2)

    def test_search_exact_match_score_one(self):
        s = self._store()
        score, _ = s.search([1.0, 0.0, 0.0], top_k=1)[0]
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_search_empty_store(self):
        self.assertEqual(VectorStore().search([1, 2, 3]), [])

    def test_save_load_roundtrip(self):
        s = self._store()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "index.pkl")
            s.save(path)
            s2 = VectorStore.load(path)
            self.assertEqual(s2.search([0.0, 1.0, 0.0], top_k=1)[0][1]["t"], "y")


if __name__ == "__&#8203;main__":
    unittest.main()
