import re
import json
from pathlib import Path
from pyspark.sql import SparkSession


def get_doc_id(path):
    # file:///.../dataset/d1.txt  ->  d1
    return path.split("/")[-1].rsplit(".", 1)[0]


def tokenize_document(document):
    doc_id, content = document

    # Convert text to lowercase and extract words
    tokens = re.findall(r"\b\w+\b", content.lower())

    # Positions start from 1
    return [
        (term, (doc_id, position))
        for position, term in enumerate(tokens, start=1)
    ]


def main():
    # Project directories
    project_dir = Path(__file__).resolve().parent.parent
    dataset_dir = project_dir / "dataset"
    output_dir = project_dir / "output"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not any(dataset_dir.glob("*.txt")):
        raise FileNotFoundError(
            f"No text files found in: {dataset_dir}"
        )

    # Start Spark
    spark = (
        SparkSession.builder
        .appName("Healthcare_Positional_Index")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        sc = spark.sparkContext

        # Read the documents using Spark: (file_path, file_content)
        documents_rdd = (
            sc.wholeTextFiles(dataset_dir.as_uri() + "/*.txt")
            .map(lambda pair: (get_doc_id(pair[0]), pair[1]))
        )

        # Generate (term, (document, position))
        term_positions_rdd = documents_rdd.flatMap(
            tokenize_document
        )

        # Group positions by term
        grouped_terms = term_positions_rdd.groupByKey().collect()

        # Build positional index
        positional_index = {}

        for term, occurrences in grouped_terms:
            positional_index[term] = {}

            for doc_id, position in occurrences:
                if doc_id not in positional_index[term]:
                    positional_index[term][doc_id] = []

                positional_index[term][doc_id].append(position)

        # Sort terms, documents, and positions
        sorted_index = {}

        for term in sorted(positional_index):
            sorted_index[term] = {}

            for doc_id in sorted(positional_index[term]):
                sorted_index[term][doc_id] = sorted(
                    positional_index[term][doc_id]
                )

        # Save JSON using Python
        output_file = output_dir / "positional_index.json"

        with output_file.open("w", encoding="utf-8") as file:
            json.dump(
                sorted_index,
                file,
                indent=4,
                ensure_ascii=False
            )

        print(f"Documents processed: {documents_rdd.count()}")
        print(f"Unique terms: {len(sorted_index)}")
        print(f"Positional Index saved to: {output_file}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()