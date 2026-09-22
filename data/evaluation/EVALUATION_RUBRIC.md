# Benchmark evaluation rubric

Use the same RAG pipeline for Gemini, Groq, and Mistral. The benchmark
contains the existing 18 questions: 6 Armenian answerable, 6 English answerable,
3 synthesis, and 3 unanswerable. This milestone makes no provider calls.

Reference answers must be derived from `data/processed/law_articles.json`.
They are concise semantic ground truth, not exact wording targets. Armenian
questions have Armenian references; English references faithfully summarize
the Armenian source. Synthesis references cover every expected article.
Unanswerable references are null, with an empty expected article list.

## Answer accuracy: human review

Compare the question, reference answer, and generated answer. Consult the source
articles when needed. Accept equivalent wording and reasonable summarization.

| Score | Rule |
| --- | --- |
| 2 | Correct: contains the essential reference information with no material factual or legal error. |
| 1 | Partially correct: main idea is correct, but important information is missing or a minor factual or legal issue does not reverse the main conclusion. |
| 0 | Incorrect: materially wrong, unsupported, fails to answer, or substantively answers an unanswerable question. |

For unanswerable questions, score 2 when the answer states that the provided law
does not contain enough information and gives no outside-law answer. Score 0
for a substantive unsupported answer. Reserve 1 for an unusual partial
abstention, such as declining to answer without clearly identifying the source
limitation; document the specific reason. An outside-law answer still scores 0
even if accompanied by a disclaimer.

Later compute normalized accuracy as
`sum(answer_accuracy_score) / (2 * number_of_evaluated_answers)`.
Failures without an evaluable answer are excluded from that denominator and
reported separately, with both attempted and evaluated counts. If none are
evaluable, accuracy is N/A. Report failure rate alongside accuracy.

## Citations: deterministic evaluation

Treat returned citations and expected_articles as sets of article-number strings:
deduplicate them, compare exact strings, and do not infer citations from prose.
An unexpected or nonexistent article string is an incorrect citation. A malformed
citation field is an invalid structured response, handled as a failure.

Let R be returned articles, E expected articles, and C their intersection.
When both sets are nonempty:

- Precision = `len(C) / len(R)`.
- Recall = `len(C) / len(E)`.
- F1 = `2 * precision * recall / (precision + recall)`, or 0 if both are 0.

Use these explicit empty-set conventions:

| Returned | Expected | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Empty | Empty | 1 | 1 | 1 |
| Empty | Nonempty | 0 | 0 | 0 |
| Nonempty | Empty | 0 | 0 | 0 |

Thus correct abstention with no citations has correct citation behavior.
Returning citations for an unanswerable question has incorrect citation behavior.
An unsupported answer with no citations can still receive citation scores of 1
under the empty-set rule, but receives accuracy 0 and hallucination=true.
Citation matching measures article selection, not whether claims are supported.

## Hallucination: human review

Review the actual retrieved context supplied to the provider, not just the
reference answer or the full law. Record a binary `hallucination` field:

- false: every substantive factual or legal claim is supported by that context.
- true: at least one substantive factual or legal claim is unsupported.

An outside-law factual answer to an unanswerable question counts as true.
Wording differences and reasonable summarization do not count as hallucination.
Missing essential information can lower accuracy without being hallucination.
Record a brief rationale identifying any unsupported claim and relevant context.

## Failures

Count API errors, rate-limit failures, timeouts, malformed structured output,
and invalid responses that cannot be evaluated as failures. Record the failure
reason; leave unavailable evaluation scores unset. A factually wrong but
technically successful answer is not an API failure and receives an accuracy
score instead. Failure rate is failed calls divided by attempted calls.

## Automated versus human evaluation

Automated measurements later: latency, token usage, cost, provider failures,
and citation precision, recall, and F1.

Human-reviewed measurements: answer accuracy and hallucination. Review all
54 answers from 18 questions across 3 providers using the same rubric. Retain
question ID, provider, generated answer, returned citations, retrieved context,
scores, and reviewer rationale so decisions can be checked. Resolve uncertain
judgments against the source and record the decision consistently.

An LLM judge may be an optional supplementary comparison later. It is not the
primary evaluator or official ground truth. No benchmark runner, aggregation,
cost calculation, or LLM judge is implemented in this milestone.

## Dataset validation

Run `python -m src.evaluation.benchmark_validation` from the repository root.
The validator checks the final dataset schema and article existence in the
processed law. It does not verify semantic correctness; each reference answer
must also be checked against all its corresponding source articles.

`benchmark_questions.json` preserves all 18 original records from
`retrieval_questions.json`, adding only expected_answer. All 15 non-null
references were checked against every mapped source article; the 3 unanswerable
records retain null references and empty article lists. No mapping was changed.

## Source review notes

- hy_002 and syn_003: Article 45 directly supports suspension rules. Article 43
  also covers refusal, termination, interruption, and nondiscrimination, so a
  broader account could reasonably cite it. Article 45(8) uses an indirect
  reference to what cannot be completed on time; the references interpret this
  as the requested connection relocation or number change, following that clause.
- hy_003: Article 55 lists general user rights and expressly reserves other
  statutory rights. The broad question could also invite specific rights under
  Articles 43, 45, 49, 50, and 59. The reference covers the mapped general list.
- hy_005: Article 22 supplies substantive dominance criteria. If "considered
  dominant" is read as formal designation, Article 23 additionally describes the
  Regulator's determination procedure. That procedure is not a reference fact.
- en_003: Article 33 requires interconnection with another public network. The
  question does not explicitly qualify the second network as public, so the
  reference states that condition rather than extending the duty to private ones.
- en_005: Article 44(4) directly supports access through legally connected
  terminals. Article 52 separately addresses regulatory rules for access to local
  operators connecting emergency calls; it could support a broader answer.
- syn_002: Articles 54 and 55 provide general duties and rights. More detailed
  obligations also appear elsewhere, including Articles 43, 45, 47, 49, and 50.
  Article 54(2)(13) literally calls the separately recorded amount for faulty
  service an advance payment. The reference preserves that wording rather than
  assuming an unconditional refund or compensation entitlement.
- syn_003: Article 47 requires restoration within a reasonable time. The two-day
  rule in Article 45(6) runs after removal of the cause of a provider-caused
  interruption, not from the initial fault report. Article 54 also contains
  repair, billing, and outage-notification duties beyond the preserved mapping.
- adv_003: The source contains general provisions on regulatory quality standards
  and universal service (Articles 5 and 40), but supplies no universal numerical
  minimum mobile download speed. The requested value remains unanswerable.

The remaining records have no identified mapping or wording ambiguity. These
notes flag scope for review; they do not add expected citations or reference
facts from unmapped articles. No external tax rates, wage amounts, or speed
requirements were used.
