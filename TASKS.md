# Team Tasks and Acceptance Criteria

Dataset: `d1.txt` through `d10.txt`.

## Person 1 — Shehab Mohsen — Spark Positional Index (Part A)
- [ ] Read all ten documents using the agreed Spark approach.
- [ ] Apply the agreed tokenization and normalization rules.
- [ ] Build term → document → positions mappings.
- [ ] Ensure positions are 1-based and restart for each document.
- [ ] Export/display the complete positional index.
- [ ] Provide source code and a sample output.

## Person 2 — Youssef Sedky — TF, DF and IDF
- [ ] Calculate TF for each term in each document using the agreed definition.
- [ ] Calculate DF for every term.
- [ ] Calculate IDF using the formula agreed by the team/instructor.
- [ ] Provide readable TF, DF and IDF tables.
- [ ] Verify calculations against the positional index.

## Person 3 — Mounir Mohamed — TF-IDF and Normalization
- [ ] Build the TF-IDF matrix.
- [ ] Calculate each document vector length.
- [ ] Build the normalized TF-IDF matrix.
- [ ] Handle terms/documents with zero values safely.
- [ ] Provide tables and explain the formulas used.

## Person 4 — Ahmed Reda — Phrase Queries and Ranking
- [ ] Search the phrase `patient care`.
- [ ] Search the phrase `medical research`.
- [ ] Use positional information to require consecutive terms in the correct order.
- [ ] List matching documents for each phrase.
- [ ] Calculate query-document similarity and rank matching documents.
- [ ] Provide outputs and test cases.

## Person 5 — Gerges Tamer — Boolean Queries and Ranking
- [ ] Evaluate `patient care AND medical research`.
- [ ] Evaluate `patient care AND NOT medical research`.
- [ ] Use the phrase-match document sets from Person 4 (or the shared phrase-search module).
- [ ] Rank the resulting documents using the agreed similarity method.
- [ ] Explain the difference between AND and AND NOT.
- [ ] Provide outputs and test cases.

## Person 6 — Ziad Mamdouh — Integration, Testing and Documentation
- [ ] Create and maintain the shared project structure and integration plan.
- [ ] Define module interfaces with the team.
- [ ] Integrate all modules behind one main entry point.
- [ ] Test all calculations and all four required queries.
- [ ] Check consistency of tokenization, formulas and document IDs.
- [ ] Collect complete outputs, tables and screenshots.
- [ ] Prepare the Part C explanation and final submission checklist.

## Shared checklist
- [ ] Confirm programming language, Spark version and execution environment.
- [ ] Agree on tokenization and punctuation handling.
- [ ] Agree on TF, IDF, normalization and similarity formulas.
- [ ] Keep code, outputs and documentation organized.
- [ ] Review changes through Pull Requests before merging to `main`.
