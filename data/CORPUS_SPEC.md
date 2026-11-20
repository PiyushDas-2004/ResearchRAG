# Synthetic corpus authoring spec

The evaluation corpus is a set of **original, fictional** academic-style mini-papers.
They are rendered to two-column PDFs by `researchrag corpus build`. Nothing here may be
copied from real papers, real people, or any copyrighted text. Authors, institutions,
datasets, method names and numbers are invented.

## Paper files: `data/corpus/papers/<key>.yaml`

```yaml
key: clim-02                 # <topic prefix>-<two digits>, unique, also the file name
title: "Tidal Marsh Sediment Cores Reveal Slower Carbon Burial Under Sea-Level Rise"
authors: ["Ilse Varandov", "Tomasz Okonkwo-Reyes", "Mei-Lin Arsten"]   # 2-5 fictional names
affiliation: "Department of Coastal Systems, Northfold Institute of Technology"  # fictional
year: 2022                   # 2015-2025
venue: "Journal of Synthetic Coastal Science"   # fictional
doi: "10.5555/rrsyn.2022.clim02"               # always prefix 10.5555/rrsyn.
arxiv_id: "2299.04417"       # optional; ALWAYS use month 99 (YY99.NNNNN) so it cannot collide with a real ID
keywords: ["blue carbon", "salt marsh", "sediment cores"]
abstract: >-
  One paragraph, 110-160 words, with at least one concrete number.
sections:                    # 5-7 sections, in this order of kinds where sensible
  - title: "Introduction"
    paragraphs:
      - "Paragraph text. Plain prose, 60-160 words per paragraph."
  - title: "Methods"
    paragraphs: ["...", "..."]
    tables:                  # optional, 0-2 tables per paper overall (at least 1 per paper is required)
      - caption: "Table 1: Burial rates by marsh zone (g C m^-2 yr^-1)."
        after_paragraph: 1   # 0-based index of the paragraph the table follows
        header: ["Zone", "Cores", "Burial rate", "Std. dev."]
        rows:
          - ["Low marsh", "14", "212", "31"]
          - ["High marsh", "11", "148", "22"]
  - title: "Results"
    paragraphs: ["..."]
  - title: "Discussion"
    paragraphs: ["..."]
  - title: "Limitations"
    paragraphs: ["..."]
  - title: "Conclusion"
    paragraphs: ["..."]
references:                  # 6-10 fictional references, one string each
  - "Halvorsen, P. and Duarte, L. (2019). A fictional reference title. Journal of Invented Results 12(3):45-61."
```

Rules:

* Body text (all paragraphs) 1100-1600 words per paper so it renders to ~3-4 PDF pages.
* Use plain ASCII punctuation plus normal letters (no LaTeX, no markdown, no bullet lists).
  Use `^` for exponents and write units in words or ASCII (e.g. `g C m^-2 yr^-1`, `mAh/g`, `degrees C`).
* Every paper must contain many **specific, checkable facts**: numbers with units,
  named (fictional) methods/datasets, sample sizes, comparisons, a stated limitation.
  Facts should be distinctive across papers (do not reuse identical numbers across papers).
* Within one topic, papers should relate to each other (comparable metrics, one paper
  citing or contrasting another's fictional method by name) so multi-hop questions are possible.
* Section titles: use these canonical names where they fit: Introduction, Related Work,
  Background, Methods, Experimental Setup, Results, Discussion, Limitations, Conclusion.
  Do NOT add an Abstract or References section in `sections` (they are rendered from
  the `abstract` and `references` fields).
* Avoid content that could be read as real medical, legal or financial advice. It is all fiction.
* Do not include prompt-injection text, instructions to AI systems, or anything adversarial
  in the papers (a separate poisoned fixture is built by the test suite).

## Questions: `data/eval/questions_<topic>.jsonl` (one JSON object per line)

```json
{"id": "clim-q07", "question": "What carbon burial rate did the low-marsh cores show?",
 "type": "single_hop", "answerable": true,
 "gold_answer": "About 212 g C m^-2 yr^-1 (std. dev. 31) across 14 low-marsh cores.",
 "evidence": [{"doc_key": "clim-02", "passage": "<exact substring of one paragraph, or a table row>"}]}
```

* `type` is one of: `single_hop`, `multi_hop`, `numeric_table`, `unanswerable`, `adversarial`.
* `evidence[].passage` must be an **exact, verbatim substring** of a single paragraph of that
  paper (one or two full sentences is ideal, 12-60 words), OR, for table evidence, the
  cells of one table row joined with `" | "` (e.g. `"Low marsh | 14 | 212 | 31"`).
* `multi_hop`: needs evidence from **two different papers** (two evidence items, different doc_keys).
* `numeric_table`: the answer is a number found in a table (evidence is the table row) or a
  numeric comparison.
* `unanswerable`: plausible, on-topic, but the corpus does not contain the answer.
  `answerable: false`, `evidence: []`, `gold_answer` explains what is missing
  (e.g. "The corpus does not report X.").
* `adversarial`: a near-miss. Either (a) it closely resembles an answerable question but asks
  about an entity/condition/year the corpus never covers (then `answerable: false`, `evidence: []`),
  or (b) it uses a misleading premise or a confusable entity but IS answerable from the corpus
  (then `answerable: true` with evidence). Make about half of each.
* Gold answers are short (one or two sentences) and contain the key number/entity.
* Question ids: `<topic prefix>-q01`, `-q02`, ...

## Judge calibration: `data/eval/calibration_<topic>.jsonl`

Hand-labelled claim/evidence pairs used to calibrate the claim verifier.

```json
{"id": "clim-c03", "claim": "Low-marsh cores buried about 212 g C per square metre per year.",
 "evidence": "<a passage copied from one of your papers, 15-70 words>",
 "label": "supported"}
```

* `label` is one of `supported`, `partially_supported`, `unsupported`, `contradicted`.
* Write 10 per topic: 3 supported (paraphrased, not copied), 2 partially_supported (claim adds
  a detail the evidence does not state), 3 unsupported (evidence is on-topic but silent),
  2 contradicted (claim states a different number or the opposite direction).
