"""Boolean phrase queries and cosine-similarity ranking (Person 5).

Run from repository root:
    python src/boolean_queries.py
    python src/boolean_queries.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

PostingIndex = Mapping[str, Mapping[str, Sequence[int]]]


def phrase_sets_from_results(results: Mapping[str, object]):
    """Read matching-document lists produced by Person 4."""
    required = ("patient care", "medical research")
    sets = {}
    for phrase in required:
        if phrase not in results:
            raise KeyError(f"Person 4 phrase results missing: {phrase!r}")
        data = results[phrase]
        if not isinstance(data, dict) or "matching_documents" not in data:
            raise ValueError(f"Invalid phrase result entry for {phrase!r}")
        sets[phrase] = set(data["matching_documents"])
    return sets


def evaluate_boolean(phrase_sets):
    """Evaluate the two project Boolean queries from Person 4's phrase sets."""
    patient_care = set(phrase_sets["patient care"])
    medical_research = set(phrase_sets["medical research"])
    return {
        "patient care AND medical research": sorted(patient_care & medical_research),
        "patient care AND NOT medical research": sorted(patient_care - medical_research),
    }


def build_tfidf_vectors(index: PostingIndex):
    docs = sorted({doc for postings in index.values() for doc in postings})
    n_docs = len(docs)
    vectors = {doc: {} for doc in docs}
    idf = {}
    for term, postings in index.items():
        df = len(postings)
        idf[term] = round(math.log10(n_docs / df), 4) if df else 0.0
        for doc, positions in postings.items():
            vectors[doc][term] = len(positions) * idf[term]
    return vectors, idf


def cosine_similarity(query_vector, doc_vector):
    dot = sum(weight * doc_vector.get(term, 0.0)
               for term, weight in query_vector.items())
    qnorm = math.sqrt(sum(weight * weight for weight in query_vector.values()))
    dnorm = math.sqrt(sum(weight * weight for weight in doc_vector.values()))
    if qnorm == 0.0 or dnorm == 0.0:
        return 0.0
    return dot / (qnorm * dnorm)


def rank_documents(index: PostingIndex, documents, query_terms):
    vectors, idf = build_tfidf_vectors(index)
    query_tf = Counter(query_terms)
    query_vector = {
        term: tf * idf[term]
        for term, tf in query_tf.items() if term in idf
    }
    ranked = [
        {"document": doc, "similarity": round(
            cosine_similarity(query_vector, vectors.get(doc, {})), 6)}
        for doc in documents
    ]
    return sorted(ranked, key=lambda item: (-item["similarity"], item["document"]))


def run_boolean_queries(index, phrase_results):
    phrase_sets = phrase_sets_from_results(phrase_results)
    matches = evaluate_boolean(phrase_sets)
    return {
        "patient care AND medical research": {
            "matching_documents": matches["patient care AND medical research"],
            "ranking": rank_documents(
                index, matches["patient care AND medical research",
                ["patient", "care", "medical", "research"]),
            "logic": "intersection of the two phrase-match sets",
        },
        "patient care AND NOT medical research": {
            "matching_documents": matches["patient care AND NOT medical research"],
            "ranking": rank_documents(
                index, matches["patient care AND NOT medical research",
                ["patient", "care"]),
            "logic": "patient care phrase matches excluding medical research phrase matches",
        },
    }


def _self_test():
    sample_sets = {
        "patient care": {"d1", "d2", "d4"},
        "medical research": {"d2", "d3"},
    }
    assert evaluate_boolean(sample_sets) == {
        "patient care AND medical research": ["d2"],
        "patient care AND NOT medical research": ["d1", "d4"],
    }
    sample_index = {
        "patient": {"d1": [1], "d2": [1], "d4": [1]},
        "care": {"d1": [2], "d2": [2], "d4": [2]},
        "medical": {"d2": [3], "d3": [1]},
        "research": {"d2": [4], "d3": [2]},
    }
    ranked = rank_documents(sample_index, ["d1", "d4"], ["patient", "care"])
    assert [row["document"] for row in ranked] == ["d1", "d4"]
    assert all(0 <= row["similarity"] <= 1 for row in ranked)
    assert evaluate_boolean({"patient care": set(), "medical research": set()}) == {
        "patient care AND medical research": [],
        "patient care AND NOT medical research": [],
    }
    print("All Boolean-query tests passed.")


def main():
    parser = argparse.ArgumentParser(description="Boolean phrase queries and ranking")
    parser.add_argument("--index", default="output/positional_index.json")
    parser.add_argument("--phrase-results", default="output/phrase_results.json")
    parser.add_argument("--output", default="output/boolean_results.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return

    index_path, phrase_path = Path(args.index), Path(args.phrase_results)
    if not index_path.is_file():
        raise FileNotFoundError(f"Positional index not found: {index_path}")
    if not phrase_path.is_file():
        raise FileNotFoundError(f"Person 4 phrase results not found: {phrase_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    phrase_results = json.loads(phrase_path.read_text(encoding="utf-8"))
    results = run_boolean_queries(index, phrase_results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=4), encoding="utf-8")
    for query, data in results.items():
        print(f"\nQuery: {query}")
        print("Matching documents:", ", ".join(data["matching_documents"]) or "None")
        print("Ranking:")
        for row in data["ranking"]:
            print(f"  {row['document']}: {row['similarity']:.6f}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
