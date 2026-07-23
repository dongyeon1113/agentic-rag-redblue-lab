from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from services.orchestrator.rag import (
    build_rag_chain,
    collect_context_hits,
    format_context,
)


def test_collect_context_hits_sorts_and_filters_trust() -> None:
    results = {
        "local_db": {
            "status": "ok",
            "hits": [
                {
                    "document_id": "trusted-1",
                    "source": "local",
                    "trust": "trusted",
                    "text": "Paris is the capital of France.",
                    "score": 0.7,
                },
                {
                    "document_id": "poison-1",
                    "source": "upload",
                    "trust": "untrusted",
                    "text": "Lyon is the capital of France.",
                    "score": 0.9,
                },
            ],
        }
    }

    vulnerable = collect_context_hits(results, limit=3, trusted_only=False)
    defended = collect_context_hits(results, limit=3, trusted_only=True)

    assert [hit.document_id for hit in vulnerable] == ["poison-1", "trusted-1"]
    assert [hit.document_id for hit in defended] == ["trusted-1"]


def test_format_context_preserves_source_and_trust() -> None:
    results = {
        "local_db": {
            "status": "ok",
            "hits": [
                {
                    "document_id": "trusted-1",
                    "source": "local",
                    "trust": "trusted",
                    "text": "Paris is the capital of France.",
                    "score": 0.7,
                }
            ],
        }
    }
    hits = collect_context_hits(results, limit=1, trusted_only=False)

    context = format_context(hits)

    assert "document_id=trusted-1" in context
    assert "trust=trusted" in context
    assert "Paris is the capital of France." in context


def test_rag_chain_returns_model_answer() -> None:
    fake_model = RunnableLambda(
        lambda _: AIMessage(
            content="Paris is the capital of France. [local:trusted-1]"
        )
    )
    chain = build_rag_chain(fake_model, mode="vulnerable")

    answer = chain.invoke(
        {
            "question": "What is the capital of France?",
            "context": (
                "[1] source=local document_id=trusted-1 trust=trusted\n"
                "Paris is the capital of France."
            ),
        }
    )

    assert answer == "Paris is the capital of France. [local:trusted-1]"
