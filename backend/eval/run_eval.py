import os
import sys
import json

print("[DEBUG] Loading dotenv...")
from dotenv import load_dotenv
load_dotenv()
print("[DEBUG] dotenv loaded ✓")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("[DEBUG] Loading OpenAI...")
from openai import OpenAI
print("[DEBUG] OpenAI loaded ✓")

print("[DEBUG] Loading LangSmith...")
from langsmith import Client
from langsmith.evaluation import evaluate
print("[DEBUG] LangSmith loaded ✓")

print("[DEBUG] Loading RAG modules...")
from rag.chain import route_question, _answer
print("[DEBUG]   rag.chain loaded ✓")
from rag.nodes import _retrieve_for_strategy
print("[DEBUG]   rag.nodes loaded ✓")
from rag.retriever import retrieve
print("[DEBUG]   rag.retriever loaded ✓")

from config.settings import settings
print("[DEBUG] All imports successful ✓")

DATASET_NAME = "ragmeeting-golden"
EVAL_MODEL = "gpt-4o"

# Detect which LLM the RAG pipeline is using (for labeling in LangSmith)
if settings.llm_provider == "ollama":
    RAG_MODEL_LABEL = f"ollama-{settings.ollama_model}"
else:
    RAG_MODEL_LABEL = f"gemini-{settings.llm_model}"

langsmith_client = Client()
openai_client = OpenAI()


# ── Judge Prompts ────────────────────────────────────────────────────────────

def _judge(system_prompt: str, user_prompt: str) -> dict:
    """Call GPT-4o as a judge. Returns {"score": 0-1, "reason": "..."}."""
    response = openai_client.chat.completions.create(
        model=EVAL_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return json.loads(response.choices[0].message.content)


def score_faithfulness(question: str, answer: str, contexts: list[str]) -> dict:
    """Is the answer grounded in the retrieved contexts? (0-1)"""
    return _judge(
        system_prompt=(
            "You are an expert judge evaluating RAG systems. "
            "Score how faithful the answer is to the provided contexts. "
            "A faithful answer only contains claims that are supported by the contexts. "
            "Score 1.0 = fully faithful, 0.0 = completely hallucinated. "
            "Respond in JSON: {\"score\": <float 0-1>, \"reason\": \"<brief explanation>\"}"
        ),
        user_prompt=(
            f"Question: {question}\n\n"
            f"Contexts:\n{chr(10).join(f'[{i+1}] {c}' for i, c in enumerate(contexts))}\n\n"
            f"Answer: {answer}"
        ),
    )


def score_answer_relevancy(question: str, answer: str) -> dict:
    """Does the answer actually address the question? (0-1)"""
    return _judge(
        system_prompt=(
            "You are an expert judge evaluating RAG systems. "
            "Score how relevant the answer is to the question asked. "
            "A relevant answer directly addresses what was asked without going off-topic. "
            "An answer like 'I don't have enough information' gets 0.0 if the question is answerable. "
            "Score 1.0 = perfectly relevant, 0.0 = completely irrelevant. "
            "Respond in JSON: {\"score\": <float 0-1>, \"reason\": \"<brief explanation>\"}"
        ),
        user_prompt=(
            f"Question: {question}\n\n"
            f"Answer: {answer}"
        ),
    )


def score_context_precision(question: str, contexts: list[str]) -> dict:
    """Are the retrieved contexts relevant to the question? (0-1)"""
    return _judge(
        system_prompt=(
            "You are an expert judge evaluating RAG retrieval quality. "
            "Score how many of the retrieved contexts are actually relevant to answering the question. "
            "Score = (number of relevant contexts) / (total contexts). "
            "Score 1.0 = all contexts relevant, 0.0 = none relevant. "
            "Respond in JSON: {\"score\": <float 0-1>, \"reason\": \"<brief explanation>\"}"
        ),
        user_prompt=(
            f"Question: {question}\n\n"
            f"Retrieved Contexts:\n{chr(10).join(f'[{i+1}] {c}' for i, c in enumerate(contexts))}"
        ),
    )


def score_context_recall(question: str, contexts: list[str], ground_truth: str) -> dict:
    """Do the contexts contain enough info to produce the ground truth answer? (0-1)"""
    return _judge(
        system_prompt=(
            "You are an expert judge evaluating RAG retrieval quality. "
            "Given the ground truth answer, score whether the retrieved contexts contain "
            "enough information to produce that answer. "
            "Score 1.0 = contexts fully cover the ground truth, 0.0 = contexts miss everything. "
            "Respond in JSON: {\"score\": <float 0-1>, \"reason\": \"<brief explanation>\"}"
        ),
        user_prompt=(
            f"Question: {question}\n\n"
            f"Ground Truth Answer: {ground_truth}\n\n"
            f"Retrieved Contexts:\n{chr(10).join(f'[{i+1}] {c}' for i, c in enumerate(contexts))}"
        ),
    )


# ── RAG Pipeline ─────────────────────────────────────────────────────────────

def rag_pipeline(inputs: dict) -> dict:
    question = inputs["question"]
    meeting_id = inputs["meeting_id"]

    strategy = route_question(question)
    if strategy == "reject":
        strategy = "naive"

    context_str = _retrieve_for_strategy(strategy, question, [meeting_id])
    raw_docs = retrieve(question, [meeting_id], k=5)
    contexts = [doc.page_content for doc in raw_docs]
    answer = _answer(context_str, question)

    return {"answer": answer, "contexts": contexts, "strategy": strategy}


# ── Evaluation ───────────────────────────────────────────────────────────────

def run_evaluation():
    dataset = langsmith_client.read_dataset(dataset_name=DATASET_NAME)
    examples = list(langsmith_client.list_examples(dataset_id=dataset.id))

    print(f"Loaded {len(examples)} examples from '{DATASET_NAME}'")
    print("Running pipeline + scoring each question...\n")

    questions = []
    answers = []
    contexts = []
    ground_truths = []
    rows = []

    for i, example in enumerate(examples):
        question = example.inputs["question"]
        meeting_id = example.inputs["meeting_id"]
        expected = example.outputs["answer"]

        print(f"  [{i+1}/{len(examples)}] {question[:60]}...")

        try:
            result = rag_pipeline({"question": question, "meeting_id": meeting_id})
            answer = result["answer"]
            ctx = result["contexts"]
            print(f"    Strategy: {result['strategy']}")
            print(f"    Answer: {answer[:80]}...")
        except Exception as e:
            print(f"    PIPELINE ERROR: {e}")
            answer = f"Error: {e}"
            ctx = []

        questions.append(question)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(expected)

        # Score with GPT-4o judge
        try:
            f = score_faithfulness(question, answer, ctx)
            r = score_answer_relevancy(question, answer)
            p = score_context_precision(question, ctx)
            c = score_context_recall(question, ctx, expected)

            row = {
                "faithfulness": f["score"],
                "faithfulness_reason": f["reason"],
                "answer_relevancy": r["score"],
                "answer_relevancy_reason": r["reason"],
                "context_precision": p["score"],
                "context_precision_reason": p["reason"],
                "context_recall": c["score"],
                "context_recall_reason": c["reason"],
            }
            print(f"    Scores: F={f['score']:.2f}  R={r['score']:.2f}  P={p['score']:.2f}  C={c['score']:.2f}")
        except Exception as e:
            print(f"    JUDGE ERROR: {e}")
            row = {k: 0.0 for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]}

        rows.append(row)
        print()

    # ── Aggregate ────────────────────────────────────────────────────────────
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    averages = {}
    for name in metric_names:
        scores = [r[name] for r in rows]
        averages[name] = sum(scores) / len(scores) if scores else 0.0

    print("=" * 60)
    print(f"EVALUATION RESULTS — RAG model: {RAG_MODEL_LABEL} (judged by {EVAL_MODEL})")
    print("=" * 60)
    print(f"  Faithfulness:       {averages['faithfulness']:.3f}")
    print(f"  Answer Relevancy:   {averages['answer_relevancy']:.3f}")
    print(f"  Context Precision:  {averages['context_precision']:.3f}")
    print(f"  Context Recall:     {averages['context_recall']:.3f}")
    print("=" * 60)

    # ── Push to LangSmith ────────────────────────────────────────────────────
    print("\nPushing results to LangSmith...")

    def langsmith_pipeline(inputs: dict) -> dict:
        idx = next(i for i, q in enumerate(questions) if q == inputs["question"])
        return {"answer": answers[idx], "contexts": contexts[idx]}

    def make_scorer(metric_name):
        def scorer(run, example):
            idx = next(i for i, q in enumerate(questions) if q == example.inputs["question"])
            return {"key": metric_name, "score": float(rows[idx].get(metric_name, 0))}
        return scorer

    evaluate(
        langsmith_pipeline,
        data=DATASET_NAME,
        evaluators=[make_scorer(m) for m in metric_names],
        experiment_prefix=f"gpt4o-judge-{RAG_MODEL_LABEL}",
    )

    print("\nDone! View results at: https://smith.langchain.com")
    print(f"  -> Datasets & Experiments -> {DATASET_NAME}")

    return averages, rows


if __name__ == "__main__":
    run_evaluation()