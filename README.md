# Healthcare Information Retrieval Project

A team project (6 members) implementing an Information Retrieval pipeline over the healthcare dataset (`d1.txt`–`d10.txt`) using Apache Spark.

## Project parts

- **Part A — Positional Index:** Build an index mapping each term to documents and 1-based positions.
- **Part B — Retrieval and ranking:** Calculate TF, DF, IDF, TF-IDF, document vector lengths, normalized TF-IDF, phrase queries, Boolean queries, and similarity-based ranking.
- **Part C — Submission:** Source code, result tables, screenshots/outputs, and explanations of the implemented concepts.

## Team division

| Member | Responsibility | Main deliverables |
|---|---|---|
| Person 1 | Spark Positional Index (Part A) | Tokenization, positional index, source code, index output |
| Person 2 | TF, DF & IDF | Term-frequency data, document frequencies, IDF values |
| Person 3 | TF-IDF & Normalization | TF-IDF matrix, vector lengths, normalized matrix |
| Person 4 | Phrase Queries & Ranking | Search for `patient care` and `medical research`, matching documents and similarity ranking |
| Person 5 | Boolean Queries & Ranking | `patient care AND medical research` and `patient care AND NOT medical research`, results and ranking |
| Person 6 | Integration, Testing & Documentation | Project structure, integration, testing, screenshots, final report and submission |

## Dataset

Place the ten dataset files in the `dataset/` directory:

```text
dataset/
  d1.txt
  d2.txt
  ...
  d10.txt
```

## Shared implementation rules

Agree on these with the whole team before implementing separate modules:

1. Use the same tokenization and normalization rules in every module.
2. Positions start at **1** and restart from 1 for each document.
3. A phrase matches only when its terms appear consecutively and in the same order.
4. Use one agreed TF, IDF, vector-normalization, and similarity formula throughout the project. Record the exact formulas in the report.
5. Keep intermediate results and final outputs reproducible.

## Suggested workflow

1. Confirm the implementation language and Spark setup.
2. Agree on tokenization, data structures, and formulas.
3. Develop modules in parallel using shared input/output contracts.
4. Integrate through one main entry point.
5. Run all required queries against the complete dataset.
6. Save result tables, screenshots, and explanations for Part C.

## Collaboration

Use one branch per task or feature. Open a Pull Request into `main` and request a teammate review before merging. Do not commit generated temporary files or personal IDE settings.
