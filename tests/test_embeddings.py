import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings import TfidfBackend
import unittest


class TestTfidfBackend(unittest.TestCase):

    def test_embed_before_fit_raises(self):
        b = TfidfBackend()
        with self.assertRaises(RuntimeError):
            b.embed(["hello world"])

    def test_dim_equals_vocab_size(self):
        b = TfidfBackend()
        corpus = ["hello world", "world foo", "foo bar baz"]
        b.fit(corpus)
        expected = set()
        for doc in corpus:
            expected.update(doc.lower().split())
        self.assertEqual(b.dim, len(expected))
        self.assertGreater(b.dim, 0)

    def test_embed_deterministic(self):
        b = TfidfBackend()
        b.fit(["hello world foo", "world foo bar"])
        self.assertEqual(b.embed(["hello world"]), b.embed(["hello world"]))

    def test_embed_different_texts_differ(self):
        b = TfidfBackend()
        b.fit(["hello world foo", "world foo bar", "completely different tokens"])
        a = b.embed(["hello world"])[0]
        c = b.embed(["completely different"])[0]
        self.assertNotEqual(a, c)

    def test_embed_vector_length_matches_dim(self):
        b = TfidfBackend()
        b.fit(["alpha beta gamma", "beta gamma delta"])
        self.assertEqual(len(b.embed(["alpha beta"])[0]), b.dim)


if __name__ == "__&#8203;main__":
    unittest.main()
