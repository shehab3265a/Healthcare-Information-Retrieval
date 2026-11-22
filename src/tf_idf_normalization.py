import json
import math
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
output_folder = project_dir / "output"
files = [p for p in output_folder.iterdir() if p.is_file() and "tf_df_idf" in p.name]
input_path = files[0]
output_path = output_folder / "tfidf_normalized.json"
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

tf = data["tf"]
idf = data["idf"]

terms = sorted(tf.keys())
docs = list(tf[terms[0]].keys())

# حساب مصفوفة TF-IDF
tfidf = {}
for term in terms:
    tfidf[term] = {}
    for doc in docs:
        tfidf[term][doc] = round(tf[term][doc] * idf[term], 4)

# حساب طول متجه كل مستند
doc_lengths = {}
for doc in docs:
    s = 0
    for term in terms:
        s += tfidf[term][doc] ** 2
    if s > 0:
        doc_lengths[doc] = round(math.sqrt(s), 4)
    else:
        doc_lengths[doc] = 0.0

# حساب Normalized TF-IDF
norm_tfidf = {}
for term in terms:
    norm_tfidf[term] = {}
    for doc in docs:
        if doc_lengths[doc] > 0:
            norm_tfidf[term][doc] = round(tfidf[term][doc] / doc_lengths[doc], 4)
        else:
            norm_tfidf[term][doc] = 0.0

print("Document Vector Lengths:")
for doc in docs:
    print(doc, doc_lengths[doc])

print("\nTF-IDF Matrix:")
for term in terms:
    print(term, tfidf[term])

print("\nNormalized TF-IDF Matrix:")
for term in terms:
    print(term, norm_tfidf[term])

final_data = {
    "tfidf": tfidf,
    "doc_lengths": doc_lengths,
    "normalized_tfidf": norm_tfidf,
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2)

print("\nDone! Saved to", output_path)
