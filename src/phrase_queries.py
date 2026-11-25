"""Phrase search and cosine-similarity ranking over a positional index.

Run from the repository root:
    python src/phrase_queries.py
    python src/phrase_queries.py --index output/positional_index.json --output output/phrase_results.json
    python src/phrase_queries.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

PostingIndex = Mapping[str, Mapping[str, Sequence[int]]]


def tokenize(text: str) -> List[str]:
    """Use the same normalization/token pattern as src/positional_index.py."""
    return re.findall(r"\b\w+\b", text.lower())


def phrase_matches(index: PostingIndex, phrase: str) -> List[str]:
    """Return document IDs containing all phrase terms consecutively/in order."""
    terms = tokenize(phrase)
    if not terms:
        return []

    # A phrase needs a posting list for every term.
    if any(term not in index for term in terms):
        return []

    common_docs = set(index[terms[0]])
    for term in terms[1:]:
        common_docs.intersection_update(index[term])

    matches: List[str] = []
    for doc_id in sorted(common_docs):
        # Start positions for the first term; every next term must be at +1, +2...
        starts = set(index[terms[0]][doc_id])
        for offset, term in enumerate(terms[1:], start=1):
            starts.intersection_update(
                position - offset for position in index[term][doc_id]
            )
            if not starts:
                break
        if starts:
            matches.append(doc_id)
    return matches


def build_document_vectors(index: PostingIndex):
    """Build TF-IDF vectors using the project-wide Task 2 IDF formula."""
    documents = sorted({doc for postings in index.values() for doc in postings})
    n_docs = len(documents)
    vectors: Dict[str, Dict[str, float]] = {doc: {} for doc in documents}

    if not n_docs:
        return vectors, {}

    idf: Dict[str, float] = {}
    for term, postings in index.items():
        df = len(postings)
        # Task 2 formula: IDF = log10(N / DF), rounded to four decimals.
        idf[term] = round(math.log10(n_docs / df), 4) if df > 0 else 0.0
        for doc_id, positions in postings.items():
            tf = len(positions)
            vectors[doc_id][term] = tf * idf[term]

    return vectors, idf


def cosine_similarity(left: Mapping[str, float],
                      right: Mapping[str, float]) -> float:
    """Cosine similarity between sparse term-weight dictionaries."""
    if not left or not right:
        return 0.0
    dot = sum(weight * right.get(term, 0.0)
              for term, weight in left.items())
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def rank_phrase_matches(index: PostingIndex, phrase: str,
                        matching_docs: Iterable[str] | None = None):
    """Rank only phrase-matching docs using cosine(query TF-IDF, document TF-IDF)."""
    docs = list(matching_docs if matching_docs is not None
                else phrase_matches(index, phrase))
    doc_vectors, idf = build_document_vectors(index)
    query_terms = tokenize(phrase)
    query_tf = Counter(query_terms)
    query_vector = {
        term: frequency * idf[term]
        for term, frequency in query_tf.items() if term in idf
    }

    ranked = [
        {
            "document": doc_id,
            "similarity": cosine_similarity(query_vector, doc_vectors.get(doc_id, {})),
        }
        for doc_id in docs
    ]
    return sorted(ranked, key=lambda item: (-item["similarity"], item["document"]))


def run_queries(index: PostingIndex, phrases: Sequence[str]):
    results = {}
    for phrase in phrases:
        matches = phrase_matches(index, phrase)
        results[phrase] = {
            "matching_documents": matches,
            "ranking": rank_phrase_matches(index, phrase, matches),
        }
    return results


def _self_test():
    sample = {
        "patient": {"d1": [1], "d2": [1], "d3": [1]},
        "care": {"d1": [2], "d2": [3], "d3": [2]},
        "medical": {"d1": [4], "d2": [2]},
        "research": {"d1": [5], "d2": [5]},
    }
    assert phrase_matches(sample, "patient care") == ["d1", "d3"]
    assert phrase_matches(sample, "medical research") == ["d1", "d2"]
    assert phrase_matches(sample, "care patient") == []
    assert phrase_matches(sample, "patient missing") == []
    ranked = rank_phrase_matches(sample, "patient care")
    assert [row["document"] for row in ranked] == ["d1", "d3"]
    assert all(0.0 <= row["similarity"] <= 1.0 for row in ranked)
    print("All phrase-query tests passed.")


def main():
    parser = argparse.ArgumentParser(description="Phrase search and TF-IDF cosine ranking")
    parser.add_argument("--index", default="output/positional_index.json",
                        help="Path to positional_index.json")
    parser.add_argument("--output", default="output/phrase_results.json",
                        help="Path for the JSON results")
    parser.add_argument("--self-test", action="store_true",
                        help="Run built-in test cases and exit")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    index_path = Path(args.index)
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Positional index not found: {index_path}. Run src/positional_index.py first."
        )
    with index_path.open(encoding="utf-8") as file:
        index = json.load(file)

    phrases = ["patient care", "medical research"]
    results = run_queries(index, phrases)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=4), encoding="utf-8")

    for phrase, data in results.items():
        print(f"\nPhrase: {phrase}")
        print("Matching documents:", ", ".join(data["matching_documents"]) or "None")
        if data["ranking"]:
            print("Ranking (document: cosine similarity):")
            for row in data["ranking"]:
                print(f"  {row['document']}: {row['similarity']:.6f}")
        else:
            print("Ranking: no matching documents")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
