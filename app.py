"""Obsidian RAG 的 Streamlit Web 界面。

把命令行版的 build / ask 变成浏览器里可交互的问答页，便于作品集在线演示。

运行：
    pip install streamlit
    streamlit run app.py
依赖：本文件直接复用 rag.py 的底层函数（load_config / parse_vault / build_backend /
    VectorStore / synthesize），不重复实现 RAG 逻辑——这是"薄 UI 层"的关键设计。
"""
from __future__ import annotations

import os
import sys

import streamlit as st

# 让本文件能 import 同目录下的 rag / embeddings / obsidian_parser / store
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag  # noqa: E402


@st.cache_resource(show_spinner="正在构建索引…")
def build_index(config_path: str):
    """读取配置并构建向量索引。结果用 cache 缓存，避免每次交互都重建。

    逻辑与 rag.cmd_build 一致，但去掉了 print 副作用，改为把 (cfg, store, 计数) 返回给 UI。
    """
    cfg = rag.load_config(config_path)
    docs = rag.parse_vault(cfg["vault_path"])
    chunks = [c for d in docs for c in d.chunks]
    emb_cfg = cfg.get("embedding", {})
    backend = rag.build_backend(
        emb_cfg.get("backend", "tfidf"),
        emb_cfg.get("model"),
        emb_cfg.get("base_url"),
    )
    corpus = [c.text for c in chunks]
    backend.fit(corpus)
    vectors = backend.embed(corpus)
    meta = [
        {"text": c.text, "source": c.source, "title": c.title, "section": c.section}
        for c in chunks
    ]
    store = rag.VectorStore()
    store.add(vectors, meta)
    return cfg, store, len(docs), len(chunks)


def answer(cfg, store, question: str):
    """检索并（可选）调用 LLM，返回 (results, llm_text|None)。

    与 rag.cmd_ask 同逻辑：TF-IDF 必须用已存片段重建词表，保证查询向量与索引同维度。
    """
    emb_cfg = cfg.get("embedding", {})
    backend = rag.build_backend(
        emb_cfg.get("backend", "tfidf"),
        emb_cfg.get("model"),
        emb_cfg.get("base_url"),
    )
    backend.fit([m["text"] for m in store.meta])
    qv = backend.embed([question])[0]
    results = store.search(qv, cfg.get("top_k", 5))
    llm_cfg = cfg.get("llm", {})
    llm_text = None
    if llm_cfg.get("enabled"):
        llm_text = rag.synthesize(llm_cfg, question, results)
    return results, llm_text


def main() -> None:
    st.set_page_config(page_title="Obsidian RAG", page_icon="📚")
    st.title("📚 Obsidian RAG 知识库")

    config_path = st.sidebar.text_input("配置文件", value="config.yaml")
    if not os.path.exists(config_path):
        st.sidebar.error(f"找不到 {config_path}，请先 cp config.example.yaml config.yaml")
        return

    with st.sidebar:
        if st.button("🔄 重建索引"):
            build_index.clear()
            st.rerun()
        st.caption("修改 vault_path 后点此重建")

    cfg, store, n_docs, n_chunks = build_index(config_path)
    st.sidebar.success(f"已索引 {n_docs} 篇文档 / {n_chunks} 个片段")
    st.sidebar.caption(
        f"vault: {cfg['vault_path']}  |  top_k: {cfg.get('top_k', 5)}  |  "
        f"后端: {cfg.get('embedding', {}).get('backend', 'tfidf')}"
    )

    st.write("基于你的 Obsidian 笔记做检索增强问答。在下方输入问题，查看命中的笔记片段。")

    if "results" not in st.session_state:
        st.session_state.results = None
        st.session_state.llm_text = None

    with st.form("ask_form"):
        question = st.text_input("输入你的问题", placeholder="例如：SSH 公钥 认证")
        submitted = st.form_submit_button("检索", type="primary")

    if submitted and question.strip():
        with st.spinner("检索中…"):
            results, llm_text = answer(cfg, store, question)
        st.session_state.results = results
        st.session_state.llm_text = llm_text

    if st.session_state.results is not None:
        hits = [r for r in st.session_state.results if r[0] > 0]
        if st.session_state.llm_text:
            st.subheader("🤖 AI 综合回答")
            st.info(st.session_state.llm_text)
        if not hits:
            st.warning("没有找到相关片段。换个关键词试试（例如：SSH 公钥 认证）。")
        else:
            st.subheader("📑 检索到的相关片段")
            for i, (score, m) in enumerate(hits, 1):
                with st.expander(f"#{i}  [{score:.3f}] {m['title']} › {m['section']}"):
                    st.caption(f"来源: {m['source']}")
                    st.write(m["text"])



if __name__ == "__main__":
    main()
