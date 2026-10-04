from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


DIAGNOSTIC_TREE = {
    "faithfulness": (
        "LLM tự bịa câu trả lời ngoài tài liệu",
        "Thắt chặt system prompt, giảm nhiệt độ (temperature) về 0"
    ),
    "context_recall": (
        "Hệ thống tìm kiếm bỏ sót đoạn văn đúng",
        "Cải thiện lại bước cắt đoạn hoặc bổ sung từ khóa BM25"
    ),
    "context_precision": (
        "Đoạn văn không liên quan bị xếp lên đầu",
        "Bổ sung tầng Cross-Encoder reranking hoặc lọc theo metadata"
    ),
    "answer_relevancy": (
        "Câu trả lời bị lệch trọng tâm câu hỏi",
        "Viết lại prompt hướng dẫn mô hình trả lời trực tiếp hơn"
    ),
}


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        df = result.to_pandas()
        per_question = []
        for _, row in df.iterrows():
            per_question.append(EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", [])),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            ))

        return {
            "faithfulness": float(result.get("faithfulness", 0.0) or 0.0),
            "answer_relevancy": float(result.get("answer_relevancy", 0.0) or 0.0),
            "context_precision": float(result.get("context_precision", 0.0) or 0.0),
            "context_recall": float(result.get("context_recall", 0.0) or 0.0),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation fallback (no OpenAI API key): {e}")
        per_question = []
        f_scores, a_scores, cp_scores, cr_scores = [], [], [], []

        for q, ans, ctxs, gt in zip(questions, answers, contexts, ground_truths):
            q_words = set(re.findall(r"\w+", q.lower()))
            gt_words = set(re.findall(r"\w+", gt.lower()))
            ans_words = set(re.findall(r"\w+", ans.lower()))
            ctx_text = " ".join(ctxs).lower()
            ctx_words = set(re.findall(r"\w+", ctx_text))

            # Faithfulness: answer grounded in retrieved context
            if ans_words:
                overlap = len(ans_words & ctx_words) / len(ans_words)
                f = min(1.0, overlap * 0.9 + 0.1)
            else:
                f = 0.5

            # Answer Relevancy: answer relevance to the question
            if q_words and ans_words:
                q_overlap = len(q_words & ans_words) / len(q_words)
                f_q = min(1.0, q_overlap * 0.8 + 0.2)
            else:
                f_q = 0.5

            # Context Recall: coverage of ground truth in retrieved contexts
            if gt_words:
                cr = min(1.0, len(gt_words & ctx_words) / len(gt_words))
            else:
                cr = 0.5

            # Context Precision: rank position of the most relevant context
            best_rank = -1
            max_match = 0
            for rank, c in enumerate(ctxs):
                c_words = set(re.findall(r"\w+", c.lower()))
                match_count = len(gt_words & c_words)
                if match_count > max_match:
                    max_match = match_count
                    best_rank = rank

            if best_rank >= 0 and max_match >= 3:
                cp = 1.0 / (best_rank + 1)
            else:
                cp = 0.2 if ctxs else 0.0

            f_scores.append(f)
            a_scores.append(f_q)
            cp_scores.append(cp)
            cr_scores.append(cr)

            per_question.append(EvalResult(
                question=q,
                answer=ans,
                contexts=ctxs,
                ground_truth=gt,
                faithfulness=round(f, 4),
                answer_relevancy=round(f_q, 4),
                context_precision=round(cp, 4),
                context_recall=round(cr, 4),
            ))

        agg_f = sum(f_scores) / len(f_scores) if f_scores else 0.0
        agg_a = sum(a_scores) / len(a_scores) if a_scores else 0.0
        agg_cp = sum(cp_scores) / len(cp_scores) if cp_scores else 0.0
        agg_cr = sum(cr_scores) / len(cr_scores) if cr_scores else 0.0

        return {
            "faithfulness": round(agg_f, 4),
            "answer_relevancy": round(agg_a, 4),
            "context_precision": round(agg_cp, 4),
            "context_recall": round(agg_cr, 4),
            "per_question": per_question,
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    failures = []
    for r in eval_results:
        metric_scores = {
            "faithfulness": r.faithfulness,
            "context_recall": r.context_recall,
            "context_precision": r.context_precision,
            "answer_relevancy": r.answer_relevancy,
        }
        avg_score = sum(metric_scores.values()) / len(metric_scores)
        worst_metric = min(metric_scores, key=metric_scores.get)
        worst_score = metric_scores[worst_metric]
        diagnosis, suggested_fix = DIAGNOSTIC_TREE.get(
            worst_metric,
            ("Lỗi không xác định", "Kiểm tra lại cấu hình pipeline")
        )
        failures.append({
            "question": r.question,
            "worst_metric": worst_metric,
            "score": float(worst_score),
            "avg_score": float(avg_score),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    failures.sort(key=lambda x: x["avg_score"])
    return failures[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
