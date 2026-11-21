# Evaluation report: eval-offline
**Components:** offline mock components, illustrative (fake embedder + mock extractive LLM)  
**Dataset:** `eval` (88 questions, hash `feae585ee401`)  
**Config hash:** `855f5037a2c296d5`  **Backend:** memory  **Created:** 2026-11-21T11:18:20+00:00

Means with 95% bootstrap confidence intervals (questions resampled, 2,000 resamples).

| metric | mean | 95% CI | n | direction |
|---|---|---|---|---|
| recall@5 | 0.803 | [0.711, 0.880] | 71 | higher is better |
| mrr | 0.618 | [0.535, 0.708] | 71 | higher is better |
| ndcg@5 | 0.646 | [0.560, 0.729] | 71 | higher is better |
| context_recall | 0.922 | [0.866, 0.965] | 71 | higher is better |
| citation_precision | 0.527 | [0.403, 0.643] | 50 | higher is better |
| faithfulness | 0.918 | [0.867, 0.964] | 55 | higher is better |
| hallucination_rate | 0.286 | [0.175, 0.397] | 63 | lower is better |
| answer_correctness | 0.476 | [0.393, 0.560] | 88 | higher is better |
| abstention_accuracy | 0.705 | [0.602, 0.796] | 88 | higher is better |
| over_abstention | 0.296 | [0.197, 0.408] | 71 | lower is better |
| answer_relevance | 0.752 | [0.708, 0.795] | 55 | higher is better |
| citation_recall | 0.630 | [0.490, 0.760] | 50 | higher is better |
| context_precision | 0.302 | [0.268, 0.338] | 71 | higher is better |
| hit@1 | 0.451 | [0.338, 0.563] | 71 | higher is better |
| hit@3 | 0.761 | [0.662, 0.859] | 71 | higher is better |
| hit@5 | 0.859 | [0.775, 0.930] | 71 | higher is better |
| ndcg@1 | 0.451 | [0.338, 0.563] | 71 | higher is better |
| ndcg@3 | 0.603 | [0.513, 0.695] | 71 | higher is better |
| precision@1 | 0.451 | [0.338, 0.563] | 71 | higher is better |
| precision@3 | 0.352 | [0.296, 0.413] | 71 | higher is better |
| precision@5 | 0.270 | [0.231, 0.310] | 71 | higher is better |
| recall@1 | 0.408 | [0.296, 0.521] | 71 | higher is better |
| recall@3 | 0.711 | [0.613, 0.803] | 71 | higher is better |
| removed_claim_rate | 0.198 | [0.119, 0.291] | 63 | lower is better |
| under_abstention | 0.294 | [0.118, 0.529] | 17 | lower is better |

Latency per query (ms): p50 85.34, p95 124.39. Cost: 0.000000 USD total, 0.00000000 USD/query.

## By question type
| type | n | answer_correctness | abstention_accuracy | recall@5 |
|---|---|---|---|---|
| adversarial | 12 | 0.493 | 0.583 | 0.786 |
| multi_hop | 16 | 0.262 | 0.500 | 0.594 |
| numeric_table | 16 | 0.468 | 0.812 | 0.938 |
| single_hop | 32 | 0.477 | 0.781 | 0.844 |
| unanswerable | 12 | 0.750 | 0.750 | n/a |

## Worst-scoring questions
- **bio-q11** (numeric_table): How many people does SEWERWATCH-12 site H3 serve, and what was its median blaCTX-M relative abundance?
  - gold: 48,300 people, with a median blaCTX-M relative abundance of -2.14 log10 (138 valid samples).
  - answer: (abstained) Not enough evidence in the indexed papers to answer this question. Verification found no claim supported by the retrieved passages.
  - retrieved `bio-03` p.2: 4 Results

Of 1,680 scheduled samples, 1,604 were collected, with 76 missed because of autosampler faults, courier delays or flooding. Of the collected samples,...
  - retrieved `bio-03` p.1: 3 Methods

SEWERWATCH-12 comprises seven Harrowmere sub-sewershed sites, labelled H1 to H7, and five treatment-plant influent sites in the towns of Bellsthorpe,...
  - retrieved `bio-03` p.1: Five targets (sul1, tetW, ermB, blaCTX-M and intI1) were quantified by qPCR and normalised to 16S rRNA gene copies. The sub-sewershed downstream of a fictional ...
- **bio-q14** (multi_hop): The wastewater network took its Harrowmere population estimates from a city model. What basic reproduction number did that model estimate for influenza-like illness?
  - gold: The populations came from the HARBOR-ABM synthetic population, which estimated a posterior median R0 of 1.46.
  - answer: (abstained) Not enough evidence in the indexed papers to answer this question. The passages do not contain enough information to answer the question.
  - retrieved `bio-02` p.1: Abstract

School closures are a recurring option for slowing seasonal influenza-like illness (ILI), but their benefit depends on timing and duration. We built H...
  - retrieved `bio-03` p.1: This paper reports the design and first 70 weeks of SEWERWATCH-12, a fictional network operated in Harrowmere and five nearby towns. We had three aims: to estab...
  - retrieved `bio-02` p.1: School Closure Timing in an Agent-Based Model of Influenza-Like

Illness for the Fictional City of Harrowmere

Tamsin Okafor-Lindqvist, Raoul Deverajan, Ines Ha...
- **bio-q20** (adversarial): What slow-walk MAPE did the Pacewell Band 3 show when worn on the dominant wrist?
  - gold: The corpus does not report this; devices were worn only on the non-dominant wrist and dominant-wrist placement was not tested.
  - answer: A speed-dependent correction, CadenceFloor, cut slow-walk MAPE for the Pacewell Band 3 from 23.7 to 11.2 percent. [1] The Pacewell Band 3 was previously validated for step counting against video by Brenninger and colleagues in the HALO cohort, where it was the most accurate of three wrist devices but undercounted slow walking. [3] Wrist-worn step counters are widely used as exposure measures in population health studies, yet their accuracy at slow walking speeds is rarely checked against a direct reference. [1]
  - retrieved `bio-01` p.1: Abstract

Wrist-worn step counters are widely used as exposure measures in population health studies, yet their accuracy at slow walking speeds is rarely checke...
  - retrieved `bio-01` p.2: Participants aged 65 or older had higher error than younger participants under every condition, but most of this difference disappeared after adjusting for meas...
  - retrieved `bio-04` p.1: 2 Related Work

The Pacewell Band 3 was previously validated for step counting against video by Brenninger and colleagues in the HALO cohort, where it was the m...
- **clim-q17** (unanswerable): What is the annual methane emission from the Vellmar Estuary tidal marsh?
  - gold: The corpus does not report methane emissions; the marsh study states that methane was not measured.
  - answer: The regional inventory had assumed a uniform burial rate of 210 g C m^-2 yr^-1 for the whole Vellmar marsh, which overstates the measured area-weighted rate by 27 percent and implies an annual sink of about 8,000 tonnes of carbon rather than 6,300 tonnes. [4] We did not measure methane emissions or lateral export of dissolved carbon, so burial rates should not be read as a net greenhouse gas balance for the marsh. [6]
  - retrieved `clim-01` p.1: Declining Carbon Burial in Vellmar Estuary Tidal Marsh Sediment

Cores, 1960-2019

Rosalind Achterberg-Nwosu, Teodor Quillfeather, Amara Lindqvast

Coastal Carb...
  - retrieved `clim-01` p.1: Abstract

Tidal marshes store carbon by burying organic matter as sediment accretes, but it is unclear whether burial keeps pace with accelerating sea-level ris...
  - retrieved `clim-04` p.1: Downstream communities and ecosystems depend on the Tarn in different ways. Cold-water fish habitat in the upper river depends on summer meltwater, farms in the...
- **clim-q18** (unanswerable): How many managed honeybee hives are kept in the Brennick Lowlands?
  - gold: The corpus does not give a number of honeybee hives; managed honeybees were excluded from the BPTS counts.
  - answer: Field beans and orchard fruit in the Brennick Lowlands depend heavily on wild bees, because few farms keep managed hives and those insects were excluded from our counts. [1]
  - retrieved `clim-03` p.2: The scale of decline has practical implications for crop pollination in the valley. Field beans and orchard fruit in the Brennick Lowlands depend heavily on wil...
  - retrieved `clim-03` p.1: 1 Introduction

Insect pollinators visit the flowers of most crops grown in the Tarn valley, including field beans, oilseed rape and orchard fruit, and they sus...
  - retrieved `clim-04` p.1: Downstream communities and ecosystems depend on the Tarn in different ways. Cold-water fish habitat in the upper river depends on summer meltwater, farms in the...
- **clim-q20** (adversarial): What afternoon cooling did the CoolCrown roofs deliver in the summer of 2024?
  - gold: The corpus does not cover 2024; the CoolCrown evaluation only reports summers 2019-2022 (0.90 degrees C in 2021 and 0.76 degrees C in 2022).
  - answer: CoolCrown delivered a measurable but modest reduction in street-level afternoon air temperature, roughly a fifth of the 3.9 degrees C baseline heat island. [1]
  - retrieved `clim-02` p.2: 4 Discussion

CoolCrown delivered a measurable but modest reduction in street-level afternoon air temperature, roughly a fifth of the 3.9 degrees C baseline hea...
  - retrieved `clim-02` p.2: Table 1: CoolCrown retrofits and afternoon (14:00-17:00) cooling by treated district, summers 2021-2022.
District | Roofs retrofitted | Roof area fraction | Aft...
  - retrieved `clim-02` p.1: Reflective Roof Retrofits and Afternoon Cooling in Port Calderan:

A Four-Summer Evaluation

Kasimir Oyelaran-Dahl, Juniper Mbeki-Torvald, Haruto Vasquez-Ellery...
- **clim-q22** (adversarial): How much did the CoolCrown reflective roofs raise night-time air temperatures in Port Calderan?
  - gold: They did not raise them; night-time temperatures fell slightly, by a mean of 0.21 degrees C.
  - answer: (abstained) Not enough evidence in the indexed papers to answer this question. The passages do not contain enough information to answer the question.
  - retrieved `clim-02` p.1: Raising the reflectance of roofs is one of the cheapest heat mitigation options, but evidence on how much it cools street-level air, as opposed to roof surfaces...
  - retrieved `clim-02` p.1: 1 Introduction

Urban heat islands raise air temperatures in cities above those of the surrounding land, increasing heat stress and the demand for cooling energ...
  - retrieved `clim-02` p.1: 2 Methods

The Calder Microclimate Network (CMN) consists of 212 ventilated temperature and humidity sensors mounted at 2 m height on lamp posts across Port Cal...
- **mat-q17** (unanswerable): What ionic conductivity did BrAG-6 show at minus 20 degrees C?
  - gold: The corpus does not report it; conductivity was measured from -20 to 80 degrees C to extract activation energies, but only the 25 degrees C value (7.4 mS/cm) is given.
  - answer: Pellets of 12 mm diameter were cold pressed at 375 MPa, and ionic conductivity was measured by impedance spectroscopy between minus 20 and 80 degrees C to extract activation energies. [1] In the DPSE test, BrAG-6 released 0.11 cm^3/g of hydrogen sulfide by the end of the minus 20 degrees C step, against 0.34 cm^3/g for CLAG-0. [2]
  - retrieved `mat-01` p.1: 3 Methods

BrAG-6 was synthesised from lithium sulfide, phosphorus pentasulfide, lithium chloride and lithium bromide weighed to the nominal composition Li5.6PS...
  - retrieved `mat-01` p.2: Three of the 24 cells failed by internal short circuit before completing the protocol, two with CLAG-0 and one with BrAG-6, and they are excluded from the avera...
  - retrieved `mat-01` p.2: 4 Results

BrAG-6 pellets reached an ionic conductivity of 7.4 mS/cm at 25 degrees C, nearly double the 3.9 mS/cm of CLAG-0. The activation energy fell from 0.2...
