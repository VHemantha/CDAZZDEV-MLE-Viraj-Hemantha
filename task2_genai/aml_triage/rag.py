"""Bonus: RAG fallback. If the fine-tuned model's answer perplexity exceeds a
threshold (low confidence), retrieve the typology handbook + the most similar
training cases from ChromaDB and re-query the model with that context."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Perplexity-gated RAG fallback with ChromaDB over
# a typology handbook and training cases', Date: 2026-10-07

from __future__ import annotations

import json

import numpy as np

from . import config
from .evaluate import generate
from .prompts import RAG_USER_TEMPLATE, STUDENT_SYSTEM_PROMPT


def build_store(train_rows: list[dict], persist_dir: str | None = None):
    """Two document types in one collection:
    * handbook  -- one markdown file per typology (definitions, indicators, actions)
    * case      -- each TRAINING alert with its gold decision (never val/test: no leakage)"""
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
    try:
        client.delete_collection("aml_kb")
    except Exception:
        pass
    col = client.create_collection("aml_kb", metadata={"hnsw:space": "cosine"})
    ids, docs, metas = [], [], []
    for path in sorted(config.KB_DIR.glob("*.md")):
        ids.append(f"handbook-{path.stem}")
        docs.append(path.read_text(encoding="utf-8"))
        metas.append({"kind": "handbook", "typology": path.stem})
    for r in train_rows:
        gold = json.loads(r["completion"][0]["content"])
        ids.append(f"case-{r['case_id']}")
        docs.append(f"{r['prompt'][1]['content']}\nDECISION: typology={gold['typology']}; "
                    f"action={gold['recommended_action']}; rationale={gold['rationale']}")
        metas.append({"kind": "case", "typology": gold["typology"]})
    col.add(ids=ids, documents=docs, metadatas=metas)
    return col


def retrieve(col, alert_text: str, k: int = config.RAG_TOP_K) -> str:
    """Top-k handbook entries + top-k similar labelled cases (kept short)."""
    out = []
    for kind in ("handbook", "case"):
        res = col.query(query_texts=[alert_text], n_results=k, where={"kind": kind})
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            out.append(f"[{kind}: {meta['typology']}]\n{doc[:1200]}")
    return "\n\n".join(out)


def calibrate_threshold(val_perplexities: list[float], quantile: float = 0.75) -> float:
    """Flag the least-confident quarter of answers, as measured on the VALIDATION set."""
    return float(np.quantile(val_perplexities, quantile))


def answer_with_fallback(model, tok, col, row: dict, threshold: float) -> dict:
    first = generate(model, tok, row["prompt"])
    result = {"case_id": row["case_id"], "first": first, "used_rag": False, "threshold": threshold}
    if first["perplexity"] <= threshold:
        result["final"] = first
        return result
    alert_text = row["prompt"][1]["content"]
    ctx = retrieve(col, alert_text)
    messages = [{"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                {"role": "user", "content": RAG_USER_TEMPLATE.substitute(context=ctx, alert=alert_text)}]
    result.update(used_rag=True, context=ctx, final=generate(model, tok, messages))
    return result
