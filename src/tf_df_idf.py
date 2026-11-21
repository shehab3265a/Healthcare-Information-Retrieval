import json
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
POS_INDEX_FILE = BASE_DIR / "output" / "positional_index.json"
OUTPUT_FILE = BASE_DIR / "output" / "tf_df_idf.json"
NUM_DOCS = 10
DOCS = [f"d{i}" for i in range(1, NUM_DOCS + 1)]


def main():
    with open(POS_INDEX_FILE, "r", encoding="utf-8") as f:
        pos_index = json.load(f)

    tf_result = {}
    df_result = {}
    idf_result = {}

    for term in sorted(pos_index.keys()):
        postings = pos_index[term]

        # TF: عدد مرات ظهور الكلمة في كل مستند
        tf_result[term] = {doc: len(postings.get(doc, [])) for doc in DOCS}

        # DF: عدد المستندات اللي الكلمة ظهرت فيها
        df = len(postings)
        df_result[term] = df

        # IDF: log10(N / DF)
        idf = math.log10(NUM_DOCS / df) if df > 0 else 0
        idf_result[term] = round(idf, 4)

    print("Term | DF | IDF")
    print("-" * 25)
    for term in sorted(pos_index.keys()):
        print(f"{term} | {df_result[term]} | {idf_result[term]}")

    print("\nTF (non-zero):")
    for term in sorted(pos_index.keys()):
        counts = {doc: val for doc, val in tf_result[term].items() if val > 0}
        print(f"{term}: {counts}")

    final_output = {
        "tf": tf_result,
        "df": df_result,
        "idf": idf_result,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nDone! Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()