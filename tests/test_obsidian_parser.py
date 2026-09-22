import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obsidian_parser import (
    strip_frontmatter,
    clean_wikilinks,
    split_by_heading,
    sliding_window,
    tokenize,
    parse_vault,
)
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestObsidianParser(unittest.TestCase):

    def test_strip_frontmatter_removes_block(self):
        raw = "---\ntitle: x\n---\n# Hello\nbody"
        self.assertEqual(strip_frontmatter(raw), "# Hello\nbody")

    def test_strip_frontmatter_no_frontmatter_unchanged(self):
        self.assertEqual(strip_frontmatter("plain text"), "plain text")

    def test_clean_wikilinks_alias(self):
        self.assertEqual(clean_wikilinks("see [[Note|别名]] here"), "see 别名 here")

    def test_clean_wikilinks_plain(self):
        self.assertEqual(clean_wikilinks("link [[Note]] end"), "link Note end")

    def test_split_by_heading(self):
        md = "# A\nhello\n# B\nworld"
        self.assertEqual(split_by_heading(md), [("A", "hello"), ("B", "world")])

    def test_tokenize_mixed(self):
        self.assertEqual(tokenize("Hello World 你好"), ["hello", "world", "你", "好"])

    def test_sliding_window_short(self):
        self.assertEqual(sliding_window("短文本", size=500, overlap=100), ["短文本"])

    def test_sliding_window_long_overlap(self):
        text = "x" * 1200
        parts = sliding_window(text, size=500, overlap=100)
        self.assertGreater(len(parts), 1)
        self.assertIn("x" * 50, "".join(parts))

    def test_parse_sample_vault(self):
        docs = parse_vault(os.path.join(ROOT, "sample_vault"))
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertGreater(len(doc.chunks), 0)
        chunk = doc.chunks[0]
        self.assertTrue(chunk.source.endswith("SSH 远程接入.md"))
        self.assertTrue(chunk.title)
        self.assertTrue(chunk.section)


if __name__ == "__&#8203;main__":
    unittest.main()