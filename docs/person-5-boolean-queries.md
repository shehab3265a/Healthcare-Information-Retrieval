# Person 5 — Boolean Queries and Ranking

## Phrase sets reused from Person 4

The exact phrase-match sets are read from `output/phrase_results.json`, so this task does not independently reimplement phrase matching.

- `patient care`: **d1, d5, d6**
- `medical research`: **d3, d5, d8**

## Boolean evaluation

### 1. `patient care AND medical research`

AND returns documents that occur in both phrase-match sets (set intersection):

`{d1, d5, d6} ∩ {d3, d5, d8} = {d5}`

Result: **d5**

### 2. `patient care AND NOT medical research`

AND NOT returns documents in the first set but not the second (set difference):

`{d1, d5, d6} − {d3, d5, d8} = {d1, d6}`

Result: **d1, d6**

## Ranking method

Use the same TF-IDF weights and cosine-similarity method used by the phrase-search module. IDF is `log10(N / DF)), rounded to four decimal places; TF is the count from the positional index. Scores are sorted descending, with document ID used to break ties.

- For the AND query, the positive query terms are `patient care medical research`.
- For AND NOT, the ranking query uses the positive phrase `patient care`; the negative phrase is a filter, not a positive relevance signal.

## Output (current dataset)

### `patient care AND medical research`

| Rank | Document | Cosine similarity |
|---:|---|---:|
| 1 | d5 | 0.835320 |

### `patient care AND NOT medical research`

| Rank | Document | Cosine similarity |
|---:|---|---:|
| 1 | d1 | 1.000000 |
| 2 | d6 | 0.225299 |

## AND vs AND NOT

- **AND:** requires both phrase conditions to be true for the document.
- **AND NOT:** requires the first phrase condition to be true while excluding documents that also match the second phrase.

## Test cases

| Test | First set | Second set | AND | AND NOT |
|---|---|---|---|---|
| Dataset result | d1,d5,d6 | d3,d5,d8 | d5 | d1,d6 |
| Overlap check | d1,d2,d4 | d2,d3 | d2 | d1,d4 |
| No matches | empty | empty | empty | empty |

## Run and verify

From the project root:

```bash
python src/boolean_queries.py
python src/boolean_queries.py --self-test
```

The script reads the positional index and Person 4's phrase output, prints results, and writes `output/boolean_results.json`.
