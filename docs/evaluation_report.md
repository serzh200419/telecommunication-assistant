# Evaluation Report

## 1. Evaluation objective

This evaluation compares generation providers for an internal Armenian telecommunications legal assistant grounded in the Law of the Republic of Armenia on Electronic Communications. The purpose is to choose a practical generation baseline while separating retrieval coverage, answer correctness, unsupported claims, response time, and cost.

The finalized benchmark contains **18 questions, four providers, 72 provider-question outputs, and 72 human reviews**. Gemini, Groq, and Mistral are the required free-tier comparison; OpenAI is an additional paid comparison. This is a small engineering evaluation, not a statistically significant ranking or production certification.

Questions, conditional preprocessing, BGE-M3 embeddings, the FAISS index, three-chunk retrieval, context assembly, and the shared grounding instruction were held constant. Inspection of the final files confirms identical saved retrieved results across providers for each question. The generation provider and model changed. Provider-specific API capabilities and generation settings were not identical: in particular, Groq uses JSON-object streaming while the other adapters request schema-constrained output.

Final metrics come from `data/evaluation/final_benchmark_summary.json`, with its CSV export checked for consistency. Supporting evidence is in `manual_review.json`, the four `results/<provider>_benchmark.json` files, and `retrieval_results.json`. The methodology follows `src/evaluation/evaluation_summary.py`, `benchmark.py`, and `retrieval_eval.py`. Values below are formatted for readability; the saved JSON retains full precision.

## 2. Evaluation dataset

The dataset retains the project's original questions and adds semantic reference answers grounded in the processed Armenian law. References specify essential legal facts, not wording that generated answers must reproduce exactly. Expected article sets support deterministic citation evaluation.

| Category | Questions | Purpose |
| --- | ---: | --- |
| Armenian answerable | 6 | Native-language legal QA, including operator definitions, suspension grounds, and end-user rights. |
| English answerable | 6 | Cross-lingual retrieval and English generation over Armenian source text, including interconnection, emergency access, and fines. |
| Synthesis | 3 | Combining obligations or protections across multiple articles, including confidentiality and service interruption. |
| Unanswerable/adversarial | 3 | Abstaining from unsupported tax rates and a universal numerical mobile download-speed guarantee. |

The categories are mutually exclusive. The language-specific rows cover ordinary answerable questions, not every Armenian or English question: synthesis and unanswerable items are reported separately. Some broad legal questions admit additional relevant articles; mapping ambiguities are recorded in `data/evaluation/EVALUATION_RUBRIC.md` rather than silently changing the expected sets.

## 3. Retrieval evaluation

Retrieval was evaluated independently of generated answers. The evaluator uses the same conditional query preprocessing and retrieves three chunks. It deduplicates article numbers in rank order before computing expected-article recall and reciprocal rank. The averages cover **15 answerable and synthesis questions**; the three unanswerable questions have no expected articles and are excluded.

| Retrieval metric | Saved result |
| --- | ---: |
| Recall@1 | 0.7667 |
| Recall@3 | 1.0000 |
| Mean reciprocal rank | 0.9222 |
| Synthesis expected-article coverage | 1.0000 |

Recall@3 means all mapped expected articles were found within the three retrieved chunks for this benchmark. It does not demonstrate universal retrieval quality. Article coverage also does not prove that every required clause of a split article is present in the retrieved passage. The saved synthesis coverage field has a legacy `at_5` suffix, but the saved results contain three chunks per question; this report describes the observed coverage without implying five-chunk retrieval.

This separation matters when diagnosing errors. If the expected article is present, a wrong or incomplete answer cannot automatically be attributed to failure to locate the article. Passage completeness and generation behavior still need inspection. The results support the current compact retrieval budget for these questions, not an unrestricted claim about legal retrieval.

## 4. Human review methodology

The official answer accuracy and hallucination fields were completed by human review, not an LLM judge. Each review is linked to one provider and question ID. Aggregation validates all 72 pairs, the 18 records per provider, score and boolean types, question metadata, models, statuses, and numerical measurements before producing a summary.

| Accuracy score | Meaning |
| --- | --- |
| 2 | Essential information is correct, without a material factual or legal error. |
| 1 | Main idea is correct, but an important detail is missing or a minor issue remains. |
| 0 | Materially wrong, unsupported, fails to answer, or substantively answers an unanswerable question. |

Normalized answer accuracy is `sum(scores) / (2 * reviewed successful answers)`. It is a partial-credit score, not the percentage of completely correct responses. Technical failures, such as API errors or malformed output, are reported separately and excluded from semantic averages. All final records succeeded, so every provider's denominator is 18 answers, or 36 possible score points.

Hallucination is true if at least one substantive factual or legal claim lacks support in the **actual retrieved context**. It is not judged against the entire law or only the reference answer. An omission may reduce accuracy without being a hallucination; a plausible fact from an unseen passage can still be unsupported in this RAG response.

Citation precision, recall, and F1 are deterministic set comparisons with expected articles. Duplicate citations are deduplicated. Both sets empty receive 1; only one set empty receives 0. Provider summaries are macro averages of the stored per-question metrics, not an F1 recalculated from aggregate precision and recall. An unsupported answer with no citations can therefore score well on empty-set citation matching while failing human evaluation.

A correct citation does not establish a correct interpretation. Conversely, an extra relevant article can lower citation precision without making the answer wrong. Abstention accuracy counts unanswerable questions with human score 2; each provider correctly abstained on all three such questions.

## 5. Final benchmark results

The comparison below is transposed to keep all requested metrics readable in a document. Times are provider-generation milliseconds, and token counts and costs cover the 18 final records for each provider.

| Metric | Gemini | Groq | Mistral | OpenAI |
| --- | --- | --- | --- | --- |
| Model | gemini-3.5-flash-lite | openai/gpt-oss-120b | voxtral-small-2507 | gpt-5.6-sol |
| Answer accuracy | 86.11% | 91.67% | 83.33% | 97.22% |
| Citation precision | 98.15% | 100.00% | 96.30% | 87.96% |
| Citation recall | 100.00% | 100.00% | 100.00% | 100.00% |
| Citation F1 | 98.89% | 100.00% | 97.78% | 92.22% |
| Hallucination rate | 5.56% | 11.11% | 11.11% | 0.00% |
| Mean TTFT (ms) | 1,479.89 | 1,302.94 | 420.70 | 1,644.80 |
| Mean latency (ms) | 2,318.29 | 1,304.22 | 1,160.23 | 7,293.66 |
| Median latency (ms) | 2,061.34 | 1,119.83 | 909.38 | 3,704.02 |
| P95 latency (ms) | 3,783.91 | 2,529.84 | 2,376.53 | 27,869.72 |
| Prompt tokens | 49,984 | 32,446 | 34,292 | 31,594 |
| Completion tokens | 3,074 | 6,630 | 2,072 | 2,800 |
| Actual benchmark cost (USD) | $0 | $0 | $0 | $0.1823760 |
| Estimated paid cost (USD) | $0.0226802 | $0.0088449 | $0.0042580 | $0.1823760 |
| Failure rate | 0.00% | 0.00% | 0.00% | 0.00% |
| Abstention accuracy | 100.00% | 100.00% | 100.00% | 100.00% |

The score distributions make the partial-credit accuracy metric more concrete:

| Provider | Score 2 | Score 1 | Score 0 | Hallucination count |
| --- | ---: | ---: | ---: | ---: |
| Gemini | 14 | 3 | 1 | 1 |
| Groq | 15 | 3 | 0 | 2 |
| Mistral | 12 | 6 | 0 | 2 |
| OpenAI | 17 | 1 | 0 | 0 |

All providers recovered the expected citations in their successful outputs, but additional citations lowered precision for some. OpenAI's higher semantic accuracy and lower citation precision are therefore not contradictory. Zero reviewed hallucinations for OpenAI describes these 18 outputs only.

## 6. Accuracy by question type

| Question type | Gemini | Groq | Mistral | OpenAI |
| --- | ---: | ---: | ---: | ---: |
| Armenian answerable | 100.00% | 100.00% | 91.67% | 100.00% |
| English answerable | 75.00% | 91.67% | 83.33% | 100.00% |
| Synthesis | 66.67% | 66.67% | 50.00% | 83.33% |
| Unanswerable | 100.00% | 100.00% | 100.00% | 100.00% |

Armenian answerable performance is stronger than English answerable performance for Gemini, Groq, and Mistral; OpenAI scores fully on both six-question groups. These are different questions, not matched translations, so the difference cannot be isolated as a language effect.

Synthesis is the lowest-scoring category for every provider. Combining multiple provisions while preserving exceptions and procedural conditions is more demanding than retrieving the right article numbers. All providers handle the three unanswerable questions correctly, but this narrow sample is not a comprehensive adversarial evaluation.

## 7. Provider analysis

### Gemini

Gemini combines 86.11% normalized accuracy with strong citation matching and one reviewed hallucination. It uses free-tier benchmark access and has intermediate latency, but the largest reported prompt-token total. Different tokenizers and API accounting mean that this total should not be read as evidence that Gemini received more retrieved legal text.

Review notes identify omitted exceptions and obligations: `en_002` omits the low-power device exception, and `syn_002` omits several Article 54 duties. More seriously, `en_004` uses "unless" in a way the reviewer judged to reverse the refusal rule, despite citing the expected Article 42. The English synthesis question `syn_003` receives an Armenian answer and omits suspension protections. These cases show why citation matching and language instructions are insufficient quality checks.

### Groq

Groq has the highest answer accuracy among the required free-tier providers, perfect final citation metrics, and low total latency. Its estimated paid cost is also low. Fifteen answers score 2 and three score 1, with no score-0 answers.

The two reviewed hallucinations occur in synthesis. For `syn_002`, the reviewer flags unsupported "high quality" wording alongside omitted duties; for `syn_003`, "higher-tier operators" is unsupported and several rules are missing. Its 11.11% hallucination rate therefore remains material for legal use despite strong overall accuracy. An archived development result records a rate-limit error, discussed separately from final reliability below.

### Mistral

Mistral has the fastest mean TTFT, the lowest mean total latency, and the lowest estimated paid cost. It retains full citation recall and used free-tier access. The tradeoff is the lowest normalized answer accuracy, with six partially correct answers.

The reviews show incomplete legal conditions rather than merely different wording. `hy_002` lists only three of seven principal suspension grounds; `en_005` omits the technical-feasibility condition. In synthesis, the reviewer flags conflation of confidentiality rules and an overbroad statement of the two-day restoration rule. These omissions show that the latency advantage comes with a measurable quality tradeoff for this legal QA task.

### OpenAI

OpenAI is the additional paid comparison. It has the highest normalized accuracy, 17 fully correct answers, and no hallucinations recorded in final human review. Its remaining partial answer, `syn_003`, omits several notice and suspension protections.

Its lower citation precision comes from extra citations relative to the expected sets, not missing expected articles. For example, `hy_005` cites Articles 22 and 23 where only 22 is expected; the reviewer explicitly accepts the added procedure as supported. This illustrates a limitation of exact article-set scoring, without establishing that every extra citation is necessary.

The tradeoffs are the highest paid cost, substantially higher mean latency, and a much larger latency tail. These measurements support a quality-first baseline on this task, not a claim of universal model superiority.

## 8. Armenian and English handling

The source law is Armenian. BGE-M3 embeds Armenian and English queries directly into the same multilingual space; there is no translation stage or separate English source representation. The benchmark's perfect Recall@3 includes the English answerable questions, so the evidence does not support a claim that English article retrieval failed.

For the lower English answer scores, the review notes point to missing exceptions, omitted conditions, and one reversed rule after retrieval. This is consistent with generation or completeness issues, although article-level retrieval metrics cannot exclude passage-level limitations. The Armenian response to Gemini's English synthesis question is a separate observed language-following problem. No broad claim about language superiority follows from six nonparallel questions per answerable group.

## 9. Latency tradeoffs

Mistral reaches first output at a mean of 420.70 ms and finishes at 1,160.23 ms. Groq is also fast overall at 1,304.22 ms. Gemini takes 2,318.29 ms on average, while OpenAI takes 7,293.66 ms. OpenAI's median of 3,704.02 ms and P95 of 27,869.72 ms show that its slower tail matters beyond the average.

These timings start immediately before the provider request. TTFT ends on the first nonempty generated text fragment, which can be JSON syntax rather than an answer word. Total latency ends when the stream finishes. Neither includes retrieval, embedding model initialization, client setup, or UI rendering. The current UI displays completed responses rather than streaming answer tokens to the user.

Groq's mean TTFT is close to its mean total latency. That is an observed property of received stream events, not a measurement of the provider's internal token-generation schedule. Network delivery and API buffering can influence these timings. P95 uses inclusive linear interpolation over only 18 measurements, so it is descriptive rather than a reliable production tail estimate. Small timing differences should not outweigh substantive quality differences without repeated testing.

## 10. Cost tradeoffs

Gemini, Groq, and Mistral incurred zero API charges for these free-tier runs. Estimated paid cost applies configured standard rates to provider-reported prompt and completion tokens. OpenAI used paid access; the current implementation records actual cost as that same token-based paid calculation, not as an independently reconciled invoice.

| Provider | Input USD / million tokens | Output USD / million tokens | Pricing date |
| --- | ---: | ---: | --- |
| Gemini | 0.30 | 2.50 | 2026-09-23 |
| Groq | 0.15 | 0.60 | 2026-09-23 |
| Mistral | 0.10 | 0.40 | 2026-09-23 |
| OpenAI | 4.00 | 20.00 | 2026-09-24 |

The saved estimated totals are $0.0226802 for Gemini, $0.0088449 for Groq, $0.0042580 for Mistral, and $0.1823760 for OpenAI. The calculation is input tokens times input rate plus output tokens times output rate, with each rate divided by one million. OpenAI uses the configured short-context rates, without long-context multipliers or cached-input discounts.

These are API token costs for the final saved records. They exclude infrastructure, local embedding compute, human review, and any historical attempts no longer represented in those records. Eighteen questions are too few to project a production budget: workload mix, answer length, traffic, and future prices can change costs substantially.

## 11. Reliability and rate limits

Each final provider file contains 18 successful records and no failed records, yielding **0% final failure rate** for all four. This is the completed saved state, not the success rate of every attempt made during development.

The archived `data/evaluation/archive/groq_benchmark_k_5.json` contains a Groq `RateLimitError` with HTTP status 429 and a 15-second retry-after value. It documents historical benchmarking friction under an earlier configuration; it is not included in the final failure metric and does not establish a historical failure rate.

The runner persists each attempt, skips saved successes on resume, retries saved failures on a later invocation, and stops after recording a rate-limit failure. Optional CLI pacing occurs outside generation timing. This makes experimentation recoverable, but 18 completed requests per provider do not establish long-duration reliability, concurrency capacity, or an availability SLA.

Benchmark execution remains an offline CLI workflow. The current Streamlit Benchmark tab displays only the finalized saved comparison and question-type table. It cannot initiate benchmark calls. New raw outputs require human review and summary regeneration before becoming finalized evaluation results.

## 12. Recommendation

Among the three required free-tier providers, I would select **Groq serving `openai/gpt-oss-120b` as the strongest candidate for further validation**. It achieved the highest answer accuracy among the free-tier models at 91.67%, with perfect citation precision, recall, and F1, while also maintaining low latency and a low estimated paid cost. The main concern is its 11.11% hallucination rate, driven by two unsupported synthesis answers, so additional validation and monitoring would be important before using it for consequential legal questions.

Gemini remains a strong balanced free-tier alternative, with lower hallucination than Groq but lower overall answer accuracy. Mistral provides the best latency and estimated cost, but its lower answer accuracy and more frequent incomplete answers make it less suitable when legal completeness is the main priority.

The additional paid comparison, **OpenAI `gpt-5.6-sol`**, produced the strongest semantic quality in this benchmark, with 97.22% normalized accuracy and zero reviewed hallucinations. Based on these results, it would be the preferred quality-first option if the company accepts the higher latency and paid API cost. Its lower citation precision mainly reflects additional citations beyond the narrow expected article sets rather than missing expected evidence.

I would therefore treat Groq as the leading free-tier candidate and OpenAI as the stronger paid quality baseline. The provider abstraction should be retained so this choice remains reversible. Because the benchmark contains only 18 questions, both candidates should be validated on a larger held-out set and under representative operational conditions before long-term standardization.

## 13. Key tradeoffs summary

| Provider | Strength | Weakness | Best fit within measured tradeoffs |
| --- | --- | --- | --- |
| Gemini | Strong citations and fewer reviewed hallucinations than Groq or Mistral | Completeness omissions and one reversed rule | Balanced free-tier baseline |
| Groq | Strong free-tier accuracy, exact citation matching, low latency | Unsupported claims in two synthesis answers | Cost-sensitive alternative with added quality checks |
| Mistral | Fastest TTFT and lowest estimated paid cost | Lowest accuracy and frequent incomplete answers | Latency-sensitive workloads where omissions can be reviewed |
| OpenAI | Highest semantic accuracy and no reviewed hallucinations | Highest paid cost and longest latency tail | Quality-first legal-assistant baseline |

## 14. Known limitations

The evaluation covers one law and only 18 questions used during retrieval development, not a fully held-out generalization study. Human review is small-scale and judgment-dependent; no inter-reviewer agreement measurement is available. Some expected article mappings are deliberately narrow, so citation precision may penalize relevant extra evidence. The runtime validates citation membership but has no automated semantic citation verifier.

Article-level recall can conceal missing clauses. Retrieval is dense-only, without hybrid lexical search or reranking, and English queries rely on multilingual alignment without translated source text. The language groups are not parallel translations. No production concurrency test or long-duration reliability experiment supports claims beyond these final records.

Costs are benchmark-specific, and final-state reliability omits overwritten retry history. Provider and model changes can alter quality, latency, token accounting, and price. The results establish a useful comparison baseline rather than production readiness or a legal-advice guarantee.

## 15. What I would do with more time

First, build a larger held-out set with Armenian phrasing variations, more English questions, harder multi-article synthesis, and realistic unanswerable or adversarial requests. Review borderline examples independently and make essential conditions explicit in the reference criteria. That would better distinguish incomplete retrieval from incomplete generation.

Next, test hybrid lexical and dense retrieval, reranking, and claim-level citation verification against observed errors. Evaluate provider routing or fallback only with transparent provenance and measured quality gates. Add source-version handling and an update process for legal changes. Finally, run concurrency and longer-duration tests, including rate-limit behavior, and monitor quality and latency under representative usage. These are proposed extensions, not existing features.

## 16. Conclusion

Retrieval recovered the mapped articles on this benchmark, while generation quality differed meaningfully in completeness and support. OpenAI provided the strongest semantic baseline; Groq offered the strongest required free-tier balance. Mistral led on speed and estimated cost, and Gemini remained a competitive balanced option. The company should prioritize legal answer quality while validating the chosen model against its latency, budget, and operational requirements on a larger held-out evaluation.
