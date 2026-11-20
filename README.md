# Healthcare Information Retrieval Project

## Project Overview

This project implements a Healthcare Information Retrieval system using Apache Spark. The team works with its assigned healthcare dataset, which contains ten text documents: `d1.txt` to `d10.txt`.

The project is divided into three parts: building a Positional Index with a Spark application, implementing retrieval and ranking functions, and preparing the documentation and final submission.

## Assigned Domain and Dataset

- **Assigned domain:** Healthcare
- **Dataset:** Ten text documents, `d1.txt` through `d10.txt`
- **Dataset directory:** `dataset/`

Expected dataset structure:

```text
dataset/
  d1.txt
  d2.txt
  d3.txt
  ...
  d10.txt
```

## Part A — Spark Application: Positional Index

Build a Positional Index from the assigned healthcare dataset using the Spark application.

The Positional Index must display:

- Each term in the dataset.
- The document IDs in which each term appears.
- All token positions of each term within each document.

**Position numbering:** Positions start at **1** for the first token in each document.

## Part B — Retrieval and Ranking

Use the output of the Positional Index from Part A to implement the following retrieval and ranking functions:

1. Compute and display **Term Frequency (TF)** for every term in every document.
2. Compute and display **Document Frequency (DF)** and **Inverse Document Frequency (IDF)** for every term.
3. Compute and display the **TF-IDF matrix**.
4. Compute document vector lengths and display the **Normalized TF-IDF matrix**.
5. Support phrase queries using the Positional Index. The assigned phrase queries are:
   - `patient care`
   - `medical research`
6. For each phrase query:
   - Identify the matching documents.
   - Compute query-document similarity.
   - Display similarity scores.
   - Rank the matching documents.
7. Support the Boolean query: **`patient care AND medical research`**.
8. Support the Boolean query: **`patient care AND NOT medical research`**.
9. For each Boolean query, display the matching documents and rank the returned documents using the similarity score.

## Part C — Documentation and Submission

The final submission must include:

### Required source code and outputs

- Source code for the Spark Positional Index.
- Source code for the retrieval and ranking part.
- The Positional Index.
- TF, DF, and IDF results.
- The TF-IDF matrix.
- Document vector lengths.
- The Normalized TF-IDF matrix.
- Phrase-query results.
- Boolean-query results.
- Similarity scores and document rankings.
- Screenshots or program outputs demonstrating the required functions.

### Required explanations

Explain the following concepts and results:

1. How positional information supports phrase queries.
2. The relationship between DF and IDF, and why repeated terms affect TF.
3. The purpose of TF-IDF and vector normalization.
4. How similarity scores are used to rank documents.
5. The difference between **AND** and **AND NOT**, using examples from the assigned dataset.

## Team Members and Responsibilities

| Member | Assigned responsibility |
|---|---|
| **Shehab Mohsen** | **Spark Positional Index (Part A):** Read the ten dataset documents using Spark, tokenize the text, build the Positional Index with 1-based token positions, and provide the source code and index output. |
| **Youssef Sedky** | **TF, DF & IDF:** Calculate and display Term Frequency for every term in every document, Document Frequency for every term, and the corresponding IDF values. |
| **Mounir Mohamed** | **TF-IDF & Normalization:** Build the TF-IDF matrix, calculate document vector lengths, and produce the Normalized TF-IDF matrix. |
| **Ahmed Reda** | **Phrase Queries & Ranking:** Implement the assigned phrase queries `patient care` and `medical research`, identify matching documents using positional information, calculate similarity scores, and rank the matching documents. |
| **Gerges Tamer** | **Boolean Queries & Ranking:** Implement `patient care AND medical research` and `patient care AND NOT medical research`, display matching documents, and rank the returned documents using similarity scores. |
| **Ziad Mamdouh** | **Integration, Testing & Documentation:** Integrate the project parts, test the required functions and queries, collect outputs and screenshots, and prepare the explanations and final submission. |

## Shared Project Requirements

All teams implement the same project functions. Only the **assigned domain**, **dataset**, and **phrase queries** differ between teams.

All parts of this project must use the same assigned dataset and a consistent Positional Index output so that the retrieval and ranking components can use the results produced in Part A.
