from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


_semantic_model = None


def _get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n\n', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "strategy": "semantic", "chunk_index": 0})]

    from numpy import dot
    from numpy.linalg import norm

    def cosine_sim(a, b):
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    model = _get_semantic_model()
    embeddings = model.encode(sentences)

    chunks = []
    current_sentences = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            chunks.append(Chunk(
                text=" ".join(current_sentences),
                metadata={**metadata, "strategy": "semantic", "chunk_index": len(chunks)}
            ))
            current_sentences = [sentences[i]]
        else:
            current_sentences.append(sentences[i])

    if current_sentences:
        chunks.append(Chunk(
            text=" ".join(current_sentences),
            metadata={**metadata, "strategy": "semantic", "chunk_index": len(chunks)}
        ))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    if not text.strip():
        return ([], [])

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ([], [])

    parent_texts = []
    current_parent = ""
    for para in paragraphs:
        if len(current_parent) + len(para) + (2 if current_parent else 0) > parent_size and current_parent:
            parent_texts.append(current_parent.strip())
            current_parent = para
        else:
            if current_parent:
                current_parent += "\n\n" + para
            else:
                current_parent = para
    if current_parent.strip():
        parent_texts.append(current_parent.strip())

    parents = []
    children = []

    for p_idx, p_text in enumerate(parent_texts):
        pid = f"parent_{p_idx}"
        p_chunk = Chunk(
            text=p_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": p_idx},
            parent_id=pid
        )
        parents.append(p_chunk)

        # Split parent text into child chunks (up to child_size)
        sub_units = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', p_text) if s.strip()]
        if not sub_units:
            sub_units = [p_text]

        current_child = ""
        c_texts = []
        for unit in sub_units:
            if len(unit) > child_size:
                if current_child:
                    c_texts.append(current_child.strip())
                    current_child = ""
                for i in range(0, len(unit), child_size):
                    part = unit[i:i + child_size].strip()
                    if part:
                        c_texts.append(part)
            elif len(current_child) + len(unit) + (1 if current_child else 0) > child_size and current_child:
                c_texts.append(current_child.strip())
                current_child = unit
            else:
                if current_child:
                    current_child += " " + unit
                else:
                    current_child = unit
        if current_child.strip():
            c_texts.append(current_child.strip())

        if not c_texts:
            c_texts = [p_text[i:i + child_size].strip() for i in range(0, len(p_text), child_size) if p_text[i:i + child_size].strip()]

        for c_idx, c_text in enumerate(c_texts):
            c_chunk = Chunk(
                text=c_text,
                metadata={
                    **metadata,
                    "chunk_type": "child",
                    "parent_id": pid,
                    "child_index": len(children),
                },
                parent_id=pid
            )
            children.append(c_chunk)

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    if not text.strip():
        return []

    sections = re.split(r'(^#{1,6}\s+.+$)', text, flags=re.MULTILINE)
    chunks = []
    current_header = ""
    current_header_title = ""
    current_content = ""

    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')

    for part in sections:
        part_strip = part.strip()
        if not part_strip:
            continue
        m = header_pattern.match(part_strip)
        if m:
            if current_header or current_content:
                text_to_save = f"{current_header}\n\n{current_content}".strip() if current_header else current_content.strip()
                if text_to_save:
                    meta = {**metadata, "strategy": "structure", "chunk_index": len(chunks)}
                    if current_header_title:
                        meta["section"] = current_header_title
                    chunks.append(Chunk(text=text_to_save, metadata=meta))
            current_header = part_strip
            current_header_title = m.group(2).strip()
            current_content = ""
        else:
            if current_content:
                current_content += "\n\n" + part_strip
            else:
                current_content = part_strip

    if current_header or current_content:
        text_to_save = f"{current_header}\n\n{current_content}".strip() if current_header else current_content.strip()
        if text_to_save:
            meta = {**metadata, "strategy": "structure", "chunk_index": len(chunks)}
            if current_header_title:
                meta["section"] = current_header_title
            chunks.append(Chunk(text=text_to_save, metadata=meta))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
