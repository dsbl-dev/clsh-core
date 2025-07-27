# DSBL · Deferred Semantic Binding Language

_Context-dependent symbol activation in multi-agent systems_

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15742504.svg)](https://doi.org/10.5281/zenodo.15742504)
[![Site](https://img.shields.io/badge/site-dsbl.dev-blue.svg)](https://dsbl.dev)
[![GitHub stars](https://img.shields.io/github/stars/dsbl-dev/clsh-core?style=social)](https://github.com/dsbl-dev/clsh-core/stargazers)

---

**DSBL introduces _deferred semantic binding_:** symbols such as  
`⟦VOTE:promote_alice⟧` remain latent until runtime context (actor, timing, social
state) binds their meaning.

- **Paper:** _"Deferred Semantic Binding Language: Theory & Prototype for Closed-Loop Social Homeostasis"_ → Zenodo DOI above.
- **This repo:** reference implementation + 90-run dataset & scripts to reproduce all figures.

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

cp .env.example .env      # add your OpenAI API key
python main.py            # interactive demo
```

### Batch experiments

```bash
python batch_runner_parallel.py --runs 30 --tickets 60 --tag "research"
```

---

## Reproduce the Paper

```bash
# 1 Validate data integrity
python qc/qc_batch.py exp_output/published_data/*/events/*.jsonl

# 2 Generate cross-batch analysis & key figures
python validation/cross_batch_analyzer.py exp_output/published_data/
```

---

## Directory Map

```
core/           – voting, gating, reputation, logging
agents/         – GPT-driven behaviours & personas
gates/          – safety & civil-content filters
experiments/    – AB-tests and baselines
qc/             – batch / realtime data validation
validation/     – statistical & symbolic post-analysis
exp_output/     – published_data/ + ablation_data/ (~30 MB)
```

---

## Documentation

- **Pseudocode core** → [dsbl_core_algorithm.md](dsbl_core_algorithm.md)

---

## Dataset

- **Size:** ≈ 30 MB JSONL + metrics (90 experiments)
- **License:** CC-BY-4.0 (see `DATA_LICENSE`) – free to reuse
- Data is version-controlled; regenerate via provided batch scripts.

---

## System Configuration (excerpt)

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

### How to cite

When citing DSBL, please use the **concept DOI**  
(10.5281/zenodo.15742504).  
The concept DOI always resolves to the latest version of the work.

```bibtex
@misc{petersson_dsbl_2025,
  title   = {Deferred Semantic Binding Language: Theory & Prototype},
  author  = {Petersson, Joel},
  year    = {2025},
  doi     = {10.5281/zenodo.15742504},
  url     = {https://doi.org/10.5281/zenodo.15742504}
}
```

(Or grab the latest version-specific BibTeX from Zenodo.)

See also: [CITATION.cff](./CITATION.cff)

---

## License

- **Code:** Apache-2.0
- **All `exp_output/` data:** CC-BY-4.0 (see `DATA_LICENSE`)

---

## Maintainer

Joel Petersson · [echo@joelpetersson.com](mailto:echo@joelpetersson.com) · [https://dsbl.dev](https://dsbl.dev)
