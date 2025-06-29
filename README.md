# DSBL · Deferred Semantic Binding Language

_MAAIS for context-dependent symbol activation_

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15742505.svg)](https://doi.org/10.5281/zenodo.15742505)  
[![Site](https://img.shields.io/badge/site-dsbl.dev-blue.svg)](https://dsbl.dev)

---

DSBL introduces **deferred semantic binding**: symbols like `⟦VOTE:promote_alice⟧` keep their
meaning latent until runtime context activates them.  
The repository contains test implementation and experimental data
(34 MB), + tools to **reproduce the results in  
_"MAAIS via Deferred Semantic Binding."_**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Reproduce the Paper](#reproduce-the-paper)
3. [Directory Map](#directory-map)
4. [Dataset](#dataset)
5. [System Configuration](#system-configuration)
6. [Citation](#citation)
7. [License](#license)
8. [Maintainer](#maintainer)

---

## Quick Start

```bash
git clone https://github.com/dsbl-dev/clsh-core.git
cd clsh-core
pip install -r requirements.txt

cp .env .env.local        # add API_KEY
python main.py            # interactive demo
```

### Batch experiments

```bash
python batch_runner_parallel.py --runs 30 --tickets 60 --tag "research_batch"
```

---

## Reproduce the Paper

```bash
# Validate data integrity
python qc/qc_batch.py exp_output/published_data/*/events/*.jsonl

# Cross-batch analysis + key figures
python validation/cross_batch_analyzer.py exp_output/published_data/
```

---

## Directory Map

```
core/           — voting, gating, reputation, logging
agents/         — GPT-driven agent behaviours & personas
gates/          — security & civil content filters
experiments/    — AB-tests and baseline scripts
qc/             — batch / realtime data validation
validation/     — statistical & symbolic post-analysis
exp_output/     — published_data/ + ablation_data/ (34 MB)
```

---

## Dataset

- **Size**: ~30 MB of JSONL logs and summary metrics from 90 experiments.
- **License**: CC-BY-4.0 (see `DATA_LICENSE`) – free to reuse.
- Experimental data is included; regenerate data is possible with included batch scripts.

---

## System Configuration

Key parameters (`config/settings.yaml`):

```yaml
voting:
  promotion_threshold: 3
civil_gate:
  toxicity_threshold: 0.35
adaptive_immune:
  reflect_threshold: 0.15
ai:
  model: "gpt-4o-mini"
  max_tokens: 80
```

---

## Citation

If you use dataset, please cite:

```bibtex
@misc{petersson2025dsbl,
  title   = {MAAIS via Deferred Semantic Binding},
  author  = {Joel Petersson},
  year    = {2025},
  doi     = {10.5281/zenodo.15742505},
  url     = {https://doi.org/10.5281/zenodo.15742505}
}
```

A `CITATION.cff` file is included.

---

## License

Code © 2025 Joel Petersson, released under the **Apache 2.0 License**  
Dataset released under **CC-BY-4.0** (see `DATA_LICENSE`).

---

## Maintainer

Joel Petersson · [echo@joelpetersson.com](mailto:echo@joelpetersson.com) · [dsbl.dev](https://dsbl.dev)
