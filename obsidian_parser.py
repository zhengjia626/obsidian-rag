"""
obsidian_parser.py — 把 Obsidian 仓库的 Markdown 笔记解析成"带元数据的文本片段(Chunk)"

设计目标（这四点就是面试时能讲清楚的设计取舍）：
1. 剥离 YAML frontmatter（笔记顶部的 --- ... --- 元数据，不是正文）
2. 保留 wikilink 显示文本：[[Note|别名]] -> 别名，否则检索会搜到一堆 [[
3. 按 Markdown 标题分段，超长段再滑动窗口切分
4. 每个 Chunk 记住来源文件 / 章节，方便检索结果回跳原文
"""

import os
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """一个文本片段及其溯源信息"""
    text: str
    source: str      # 来源文件路径
    title: str       # 笔记标题（文件名或 H1）
    section: str     # 所属章节（## 标题，没有则为 "前言"）


@dataclass
class Document:
    """一篇解析后的笔记"""
    path: str
    title: str
    text: str
    chunks: list = field(default_factory=list)


def strip_frontmatter(text: str) -> str:
    """去掉 YAML frontmatter（--- 包裹的块，通常在文件最开头）"""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3:].lstrip("\n")
    return text


def clean_wikilinks(text: str) -> str:
    """[[Note|别名]] -> 别名 ; [[Note]] -> Note"""
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)  # 带别名
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)             # 不带别名
    return text


def split_by_heading(text: str):
    """按 Markdown 标题(#)切成 (章节名, 正文) 列表"""
    lines = text.splitlines()
    sections = []
    current_title = "前言"
    current_body = []
    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            if current_body:
                sections.append((current_title, "\n".join(current_body).strip()))
            current_title = m.group(2).strip()
            current_body = []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_title, "\n".join(current_body).strip()))
    return sections


def sliding_window(text: str, size: int = 500, overlap: int = 100):
    """超长正文按字符滑动窗口切分，避免单个片段过大"""
    if len(text) <= size:
        return [text] if text.strip() else []
    windows = []
    start = 0
    while start < len(text):
        end = start + size
        windows.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [w for w in windows if w]


def parse_file(path: str, chunk_size: int = 500, overlap: int = 100) -> Document:
    """解析单个 .md 文件"""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    text = strip_frontmatter(raw)
    text = clean_wikilinks(text)

    title = os.path.splitext(os.path.basename(path))[0]
    h1 = re.search(r"^#\s+(.*)$", text, re.MULTILINE)
    if h1:
        title = h1.group(1).strip()

    doc = Document(path=path, title=title, text=text)
    for section, body in split_by_heading(text):
        for piece in sliding_window(body, chunk_size, overlap):
            doc.chunks.append(Chunk(
                text=piece,
                source=path,
                title=title,
                section=section,
            ))
    return doc


def parse_vault(vault_path: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """递归解析整个仓库，返回 Document 列表"""
    docs = []
    for root, _, files in os.walk(vault_path):
        for fn in files:
            if fn.lower().endswith(".md"):
                full = os.path.join(root, fn)
                docs.append(parse_file(full, chunk_size, overlap))
    return docs


def tokenize(text: str):
    """简单分词（供 embeddings 的 TF-IDF 使用）：中文按字、英文/数字按词"""
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())
