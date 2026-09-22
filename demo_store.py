"""存储层端到端演示：解析 -> 嵌入 -> 入库 -> 检索（含中文查询）。

运行：python demo_store.py
"""
import obsidian_parser as p
import embeddings as e
import store as s

# 1. 解析：阶段1产出的 Chunk 列表
docs = p.parse_vault("sample_vault")
chunks = [c for d in docs for c in d.chunks]
texts = [c.text for c in chunks]
meta = [
    {"text": c.text, "source": c.source, "title": c.title, "section": c.section}
    for c in chunks
]

# 2. 嵌入：阶段2的 TF-IDF 后端
backend = e.build_backend("tfidf")
backend.fit(texts)
vectors = backend.embed(texts)

# 3. 入库 + 持久化：阶段3的 VectorStore
vs = s.VectorStore()
vs.add(vectors, meta)
vs.save("store/vector_store.pkl")

# 4. 检索：中文查询 -> 最相似的 top-3 片段
query = "SSH 公钥 认证"
qv = backend.embed([query])[0]
results = vs.search(qv, top_k=3)
for sim, m in results:
    print(round(sim, 3), "|", m["title"], ">", m["section"])
