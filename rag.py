"""Obsidian RAG 命令行入口。

用法：
    python rag.py build --config config.yaml
    python rag.py ask "Syncthing 如何同步？" --config config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

from embeddings import build_backend
from obsidian_parser import parse_vault
from store import VectorStore


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_build(cfg: dict) -> None:
    vault = cfg["vault_path"]
    print(f"[build] 解析仓库: {vault}")
    docs = parse_vault(vault)
    chunks = [c for d in docs for c in d.chunks]
    if not chunks:
        print("[build] 警告：未解析到任何 .md 片段，请检查 vault_path。")
        return
    print(f"[build] {len(docs)} 篇文档，{len(chunks)} 个片段")

    emb_cfg = cfg.get("embedding", {})
    backend = build_backend(
        emb_cfg.get("backend", "tfidf"), emb_cfg.get("model"), emb_cfg.get("base_url")
    )
    corpus = [c.text for c in chunks]
    backend.fit(corpus)
    vectors = backend.embed(corpus)
    meta = [
        {"text": c.text, "source": c.source, "title": c.title, "section": c.section}
        for c in chunks
    ]

    store = VectorStore()
    store.add(vectors, meta)
    store.save(cfg["store_path"])
    print(f"[build] 索引已保存 -> {cfg['store_path']}")


def synthesize(llm_cfg: dict, question: str, results: list[tuple[float, dict]]) -> str:
    from openai import OpenAI

    client = OpenAI(
        base_url=llm_cfg.get("base_url"),
        api_key=os.environ.get(llm_cfg.get("api_key_env", "OPENAI_API_KEY")),
    )
    context = "\n\n".join(
        f"[来自 {m['title']} › {m['section']}]\n{m['text']}" for _, m in results
    )
    prompt = (
        "你是笔记助手，只能基于下面提供的笔记内容回答问题，"
        "若内容不足请如实说明。\n\n"
        f"{context}\n\n问题：{question}\n\n回答："
    )
    resp = client.chat.completions.create(
        model=llm_cfg.get("model", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def cmd_ask(cfg: dict, question: str) -> None:
    store = VectorStore.load(cfg["store_path"])
    emb_cfg = cfg.get("embedding", {})
    backend = build_backend(
        emb_cfg.get("backend", "tfidf"), emb_cfg.get("model"), emb_cfg.get("base_url")
    )
    # 重建后端所需状态（TF-IDF 重建词表；其他后端为 no-op）
    backend.fit([m["text"] for m in store.meta])
    qv = backend.embed([question])[0]
    results = store.search(qv, cfg.get("top_k", 5))

    llm = cfg.get("llm", {})
    if llm.get("enabled"):
        print(synthesize(llm, question, results))
        return

    print(f"\n问题：{question}\n")
    print("检索到的相关片段：\n")
    for score, m in results:
        print(f"[{score:.3f}] {m['title']} › {m['section']}  ({m['source']})")
        print(m["text"])
        print("-" * 60)


def main() -> None:
    # 公共参数：让 --config 在子命令前后都能使用
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--config", default="config.yaml")

    ap = argparse.ArgumentParser(description="基于 Obsidian 笔记的 RAG 检索")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("build", parents=[parent], help="构建向量索引")
    p_ask = sub.add_parser("ask", parents=[parent], help="检索并回答问题")
    p_ask.add_argument("question", help="要问的问题")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(f"找不到配置文件 {args.config}，请先 cp config.example.yaml config.yaml")
        sys.exit(1)

    cfg = load_config(args.config)
    if args.cmd == "build":
        cmd_build(cfg)
    elif args.cmd == "ask":
        cmd_ask(cfg, args.question)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
