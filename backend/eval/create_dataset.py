"""
One-time script to upload the golden evaluation dataset to LangSmith.

Run from backend directory:
    python eval/create_dataset.py

This creates a dataset called "ragmeeting-golden" in your LangSmith project
with 24 human-verified Q&A pairs from real meetings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langsmith import Client

client = Client()

DATASET_NAME = "ragmeeting-golden"
DATASET_DESCRIPTION = "Golden Q&A pairs from real meetings for RAG evaluation"

examples = [
    # ── Review App Build for Next Stage (8326f6bb) ───────────────────────────
    {
        "question": "Who suggested adding an onboarding step?",
        "meeting_id": "8326f6bb-4430-4cc5-831b-68136aad8aad",
        "expected": "Rhea (Speaker 2) suggested adding a short description or onboarding step to help users understand the value.",
        "type": "factual",
    },
    {
        "question": "How many speakers were in this meeting?",
        "meeting_id": "8326f6bb-4430-4cc5-831b-68136aad8aad",
        "expected": "Three speakers: Speaker 1 (the presenter), Rhea (Speaker 2, focused on UX), and David (Speaker 3, focused on technical aspects).",
        "type": "factual",
    },
    {
        "question": "What was the main conclusion of the meeting?",
        "meeting_id": "8326f6bb-4430-4cc5-831b-68136aad8aad",
        "expected": "The team agreed that they need to balance consistent functionality and good user experience before moving to the next stage. The core idea is working and the flow is logical, but it needs polish.",
        "type": "summary",
    },
    {
        "question": "What was David's concern?",
        "meeting_id": "8326f6bb-4430-4cc5-831b-68136aad8aad",
        "expected": "David noted that the backend works for most cases but still has some edge cases. He said technically it's a good foundation to build on.",
        "type": "factual",
    },
    {
        "question": "What specific edge cases did David mention?",
        "meeting_id": "8326f6bb-4430-4cc5-831b-68136aad8aad",
        "expected": "This was not discussed in the meeting. David mentioned edge cases exist but did not specify what they are.",
        "type": "unanswerable",
    },

    # ── Demo Build User Experience Feedback (e21b8dd9) ───────────────────────
    {
        "question": "Who led the meeting?",
        "meeting_id": "e21b8dd9-dd2a-4480-a273-4a245118826b",
        "expected": "Alex led the meeting and then passed it to Jordan. Sam was also present to check for breaks or optimization needs.",
        "type": "factual",
    },
    {
        "question": "What were the action items from the demo feedback meeting?",
        "meeting_id": "e21b8dd9-dd2a-4480-a273-4a245118826b",
        "expected": "Improve user guidance, add loading indicators, handle error properties, and define UI for output.",
        "type": "summary",
    },
    {
        "question": "What happens when someone enters invalid input?",
        "meeting_id": "e21b8dd9-dd2a-4480-a273-4a245118826b",
        "expected": "The demo fails silently. It doesn't handle invalid input well and shows no error message.",
        "type": "factual",
    },
    {
        "question": "What did Jordan say about the user experience?",
        "meeting_id": "e21b8dd9-dd2a-4480-a273-4a245118826b",
        "expected": "Jordan said he was curious to see how it feels from a user perspective, and noted that as a first-time user he wasn't 100% sure what to do next on the core functionality page.",
        "type": "factual",
    },

    # ── Quick Sync: Updates and Tasks (66da7305) ─────────────────────────────
    {
        "question": "What is Priya blocked on?",
        "meeting_id": "66da7305-d053-4532-86d7-768c5c706943",
        "expected": "Priya is waiting on updated brand assets from the design team to move into high-fidelity mockups for the dashboard redesign.",
        "type": "factual",
    },
    {
        "question": "When is Marcus's error handling deadline?",
        "meeting_id": "66da7305-d053-4532-86d7-768c5c706943",
        "expected": "Wednesday.",
        "type": "factual",
    },
    {
        "question": "What is Priya supposed to do after getting the assets?",
        "meeting_id": "66da7305-d053-4532-86d7-768c5c706943",
        "expected": "Prioritize the landing page mockup, aim to have it ready by Thursday.",
        "type": "factual",
    },
    {
        "question": "What task was Marcus assigned after error handling?",
        "meeting_id": "66da7305-d053-4532-86d7-768c5c706943",
        "expected": "Start the initial setup for the notification service by Friday.",
        "type": "factual",
    },
    {
        "question": "What progress has Marcus made on API integration?",
        "meeting_id": "66da7305-d053-4532-86d7-768c5c706943",
        "expected": "Marcus got the authentication flow working. He still needs to test the error handling edge cases.",
        "type": "summary",
    },

    # ── Demo Build Review and Next Steps (d8657ced) ──────────────────────────
    {
        "question": "What were the next steps from this meeting?",
        "meeting_id": "d8657ced-ce61-42cd-9ea3-d4fb12dd4220",
        "expected": "Fix major bugs, improve the user flow, and add remaining features.",
        "type": "summary",
    },
    {
        "question": "Who stayed up Saturday night for the demo?",
        "meeting_id": "d8657ced-ce61-42cd-9ea3-d4fb12dd4220",
        "expected": "An unnamed person (referred to as 'she') stayed up the whole Saturday night to help with debugging, and the demo was ready as a result.",
        "type": "factual",
    },
    {
        "question": "What observations came out of the walkthrough?",
        "meeting_id": "d8657ced-ce61-42cd-9ea3-d4fb12dd4220",
        "expected": "Minor bugs, areas where the interface could be clearer, parts where the flow felt slow or confusing. Some features were missing or needed improvement. They also highlighted aspects that worked well.",
        "type": "summary",
    },

    # ── Leakage and Fridge Issues Follow-up (6228e16c) ──────────────────────
    {
        "question": "Who was assigned to handle the leakage issue?",
        "meeting_id": "6228e16c-1c8b-48e1-a6cf-a20794eb5cb3",
        "expected": "Rhea, because she is talkative and can handle things well.",
        "type": "factual",
    },
    {
        "question": "What was the fridge issue?",
        "meeting_id": "6228e16c-1c8b-48e1-a6cf-a20794eb5cb3",
        "expected": "Groceries are not stacked up properly in the fridge.",
        "type": "factual",
    },
    {
        "question": "When is the deadline to fix the leakage?",
        "meeting_id": "6228e16c-1c8b-48e1-a6cf-a20794eb5cb3",
        "expected": "No deadline was mentioned in the meeting.",
        "type": "unanswerable",
    },

    # ── TARs, Tablets, and Task Assignments (21a90ee1) ──────────────────────
    {
        "question": "Who was appointed to do Shirley's tar?",
        "meeting_id": "21a90ee1-d733-4139-b54e-234e333d3304",
        "expected": "Kavin was appointed, but there's no update because Erica has not signed it yet.",
        "type": "factual",
    },
    {
        "question": "How were the mini tablet tasks split?",
        "meeting_id": "21a90ee1-d733-4139-b54e-234e333d3304",
        "expected": "Kavin and the speaker would work together, and Pavi and Princi would work together.",
        "type": "factual",
    },
    {
        "question": "What is the deadline for all tasks?",
        "meeting_id": "21a90ee1-d733-4139-b54e-234e333d3304",
        "expected": "Next week.",
        "type": "factual",
    },
    {
        "question": "What did they eat at Porto's Bakery?",
        "meeting_id": "21a90ee1-d733-4139-b54e-234e333d3304",
        "expected": "The speaker only said the food was amazing and authentic, but didn't mention specific dishes.",
        "type": "unanswerable",
    },
]


def main():
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=DATASET_DESCRIPTION,
    )
    print(f"Created dataset: {DATASET_NAME} (id: {dataset.id})")

    for i, ex in enumerate(examples):
        client.create_example(
            inputs={
                "question": ex["question"],
                "meeting_id": ex["meeting_id"],
            },
            outputs={
                "answer": ex["expected"],
            },
            metadata={
                "type": ex["type"],
            },
            dataset_id=dataset.id,
        )
        print(f"  Added example {i+1}/{len(examples)}: {ex['question'][:50]}...")

    print(f"\nDone! {len(examples)} examples uploaded to LangSmith.")
    print(f"View at: https://smith.langchain.com → Datasets → {DATASET_NAME}")


if __name__ == "__main__":
    main()
