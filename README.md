# EVS-LLM Alignment

[![CI](https://github.com/Telefonica-Scientific-Research/evs-llm-alignment/actions/workflows/main.yml/badge.svg)](https://github.com/Telefonica-Scientific-Research/evs-llm-alignment/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/Telefonica-Scientific-Research/evs-llm-alignment/branch/main/graph/badge.svg)](https://codecov.io/gh/Telefonica-Scientific-Research/evs-llm-alignment)

Population-grounded evaluation of multilingual Large Language Model (LLM) alignment using the European Values Survey (EVS).

This repository contains the code associated with the paper **“Population-Grounded Evaluation of Multilingual LLM Alignment with European Societal Values”**, published at the **PANDORA Workshop @ EMNLP 2026**.

## Overview

Most value-alignment evaluations rely on synthetic benchmarks, manually designed scenarios, or predefined normative assumptions. This project instead adopts a **descriptive, population-grounded approach**: multilingual LLMs are treated as survey respondents and their answer distributions are compared with responses observed in European populations through the European Values Survey.

The repository supports the main stages of the paper's experimental pipeline:

1. Parse EVS questionnaires into a structured, machine-readable representation.
2. Administer selected EVS items to multilingual instruction-tuned LLMs under standardised prompting conditions.
3. Validate and aggregate model responses across languages and models.
4. Characterise human value profiles at country and language-family levels.
5. Construct a common value space from human EVS responses.
6. Measure descriptive alignment between human and LLM response distributions.
7. Generate the analyses and figures reported in the paper.

## Main Features

- **Multilingual survey administration:** run equivalent EVS questionnaires across several European languages.
- **Multi-model evaluation:** compare multilingual LLMs from different model families, scales, and development regions.
- **Structured prompt templates:** use language-specific and model-specific Jinja2 templates.
- **Response validation:** enforce constrained answer formats and validate generated responses.
- **Human value-space analysis:** aggregate EVS responses and project population profiles into a PCA-derived value space.
- **Descriptive alignment metrics:** compare human and model response distributions using distributional, agreement, and value-space measures.
- **Reproducible analysis:** regenerate plots, tables, and intermediate artefacts through a unified analysis pipeline.

## Repository Structure

```text
.
├── prompts/                       # Language- and model-specific prompt templates
├── Surveys/                       # EVS questionnaires and survey metadata
├── Surveys_parsed/                # Structured and standardised questionnaires
├── Surveys_responses/             # Human and LLM response data
├── src/
│   ├── data_preprocessing/        # Human-data preparation and analysis
│   ├── llm_responses_analysis/    # LLM analysis and alignment metrics
│   ├── extract_survey_csv_from_pdf.py
│   └── query_survey_llm.py
├── tests/                         # Automated tests
├── run_analysis.sh                # End-to-end analysis entry point
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

The exact layout may evolve as the former research codebase is cleaned and migrated into this repository.

## Installation

### Clone the repository

```bash
git clone https://github.com/Telefonica-Scientific-Research/evs-llm-alignment.git
cd evs-llm-alignment
```

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### Install the dependencies

```bash
pip install -r requirements.txt
```

For an editable installation of the Python package:

```bash
pip install -e .
```

## Usage

### 1. Query multilingual LLMs

Use the survey-querying entry point to administer EVS items to one or more models:

```bash
python src/query_survey_llm.py \
  --languages es it en_gb \
  --models apertus gemma27b qwen3_30b \
  --csv-dir ./Surveys_parsed \
  --output-dir ./Surveys_responses
```

The querying component supports configurable server, port, timeout, language, model, input-directory, and output-directory arguments. Run the following command for the options available in the current version:

```bash
python src/query_survey_llm.py --help
```

### 2. Run the analysis pipeline

To execute the human- and LLM-response analyses:

```bash
bash run_analysis.sh
```

To include the available consistency checks:

```bash
bash run_analysis.sh --sanity-check
```

The pipeline can also run each stage independently:

```bash
bash run_analysis.sh --human-only
bash run_analysis.sh --llm-only
```

Additional options include:

```text
--no-pca            Skip PCA plots in the human analysis
--no-consensus      Skip the human consensus analysis
--sanity-check      Enable consistency checks
--no-sanity-check   Disable consistency checks
--help              Display the available options
```

### 3. Run individual analysis scripts

Human-response analysis:

```bash
python src/data_preprocessing/human_responses_analysis.py \
  --pca \
  --by-country \
  --consensus
```

LLM-response analysis:

```bash
python src/llm_responses_analysis/llm_responses_analysis.py
```

Restrict the analysis to selected models or countries:

```bash
python src/llm_responses_analysis/llm_responses_analysis.py \
  --models gemma4 qwen3-30B-A3B \
  --countries es fr de it
```

## Methodological Scope

The analysis is organised around three stages:

### Stage 1: Human value structure

Human EVS response distributions are aggregated to characterise societal-value patterns across European countries and language families. These profiles provide the empirical reference used in subsequent comparisons.

### Stage 2: Multilingual LLM survey responses

Selected EVS items are administered to multilingual LLMs using standardised language-specific prompts and constrained response formats. The resulting responses are parsed, validated, and aggregated into model value profiles.

### Stage 3: Descriptive alignment

Human and model profiles are compared using response-distribution measures, agreement statistics, and distances within the human-derived value space. The objective is to describe correspondence and divergence, rather than prescribe a universal normative target.

## Data

This project builds on the **European Values Survey (EVS)**. Users are responsible for obtaining and using EVS data in accordance with the dataset's applicable access conditions, documentation, and citation requirements.

The repository may contain derived metadata, parsing utilities, configuration files, and experiment outputs needed to reproduce the paper. Consult the documentation accompanying each data directory before running the pipeline.

## Reproducibility Notes

- Human consensus analysis uses human survey responses and canonical questionnaire-scale metadata.
- Human-only analysis does not aggregate or compare LLM response files.
- LLM analyses operate on previously collected and validated model responses.
- Analysis scripts use a non-interactive Matplotlib backend and can run in headless environments.
- Generated figures and tables are written to the output locations configured by the corresponding scripts.

## Testing and Code Quality

Run the test suite with:

```bash
pytest
```

Depending on the final project configuration, formatting, linting, and additional quality checks may also be available through the included `Makefile` and continuous-integration workflow.

## Citation

If you use this codebase, please cite the associated paper. Replace the placeholder fields below with the final workshop proceedings metadata once available:

```bibtex
@inproceedings{solansnoguero2026population,
  title     = {Population-Grounded Evaluation of Multilingual LLM Alignment with European Societal Values},
  author    = {Solans Noguero, David and Luque Serrano, Jordi and Sant Savall, Aleix},
  booktitle = {Proceedings of the PANDORA Workshop at EMNLP 2026},
  year      = {2026},
  url       = {TODO}
}
```

## License

This project is licensed under the terms specified in [LICENSE](LICENSE).

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Support

For bug reports, questions, and feature requests, please open an issue in the [GitHub issue tracker](https://github.com/Telefonica-Scientific-Research/evs-llm-alignment/issues).

## Acknowledgements

This repository consolidates and cleans the research code used to conduct the experiments reported in the associated paper. It also reuses components developed in the previous EVS and LLM integration codebase.
