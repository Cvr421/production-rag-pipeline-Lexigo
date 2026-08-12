import argparse
import json
from pathlib import Path
from time import sleep

from .config import get_settings
from .documents import load_markdown_documents
from .pinecone_store import PineconeStore
from .graph import _terms


def _sample_cases_path() -> Path:
    return Path(__file__).resolve().parents[1] / "sample_test_cases.json"


def _wait_for_sample_readiness(store: PineconeStore, timeout_seconds: int = 300) -> None:
    cases_path = _sample_cases_path()
    if not cases_path.exists():
        return

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    in_corpus_cases = [case for case in cases if case.get("type") == "in_corpus"]
    if not in_corpus_cases:
        return

    settings = store.settings
    elapsed = 0
    while elapsed < timeout_seconds:
        all_ready = True
        for case in in_corpus_cases:
            chunks = store.search(case["question"], top_k=settings.top_k)
            actual_files = {chunk["source_file"] for chunk in chunks}
            expected_files = set(case.get("expected_source_files", []))
            question_terms = _terms(case["question"])
            evidence_terms = _terms(" ".join(chunk.get("text", "") for chunk in chunks[:3]))
            overlap = len(question_terms & evidence_terms) / max(len(question_terms), 1)
            top_score = float(chunks[0].get("score", 0.0)) if chunks else 0.0
            ready = bool(
                chunks
                and expected_files.issubset(actual_files)
                and top_score >= settings.min_relevance_score
                and overlap >= settings.min_query_term_overlap
            )
            if not ready:
                all_ready = False
                break
        if all_ready:
            return
        sleep(2)
        elapsed += 2


def ingest_corpus(corpus_dir: str, reset_namespace: bool = False) -> int:
    settings = get_settings()
    chunks = load_markdown_documents(corpus_dir)
    store = PineconeStore(settings)
    store.ensure_index()
    if reset_namespace:
        store.reset_namespace()
    count = store.upsert(chunks)
    _wait_for_sample_readiness(store)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed Markdown documents into Pinecone.")
    parser.add_argument(
        "--corpus",
        default=None,
        help="Corpus directory; defaults to CORPUS_DIR from .env.",
    )
    parser.add_argument(
        "--reset-namespace",
        action="store_true",
        help="Delete all vectors in the configured namespace before upserting.",
    )
    args = parser.parse_args()
    settings = get_settings()
    corpus_dir = args.corpus or settings.corpus_dir
    count = ingest_corpus(str(Path(corpus_dir)), reset_namespace=args.reset_namespace)
    print(f"Upserted {count} chunks into Pinecone namespace '{settings.pinecone_namespace}'.")


if __name__ == "__main__":
    main()

