# Experiment Output Directory

This directory contains experiment output from system experiments batch_runner_parallel / main (interactive).

## Directory Structure

```
exp_output/
├── published_data/                   # Research datasets
│   ├── batch_09_adaptive_immune/     # Batch 09 experiments
│   ├── batch_10_adaptive_immune/     # Batch 10 experiments
│   └── batch_11_adaptive_immune/     # Batch 11 experiments
└── ablation_data/                    # Ablation study datasets
    ├── ablation_baseline/            # Baseline configuration
    └── ablation_reflect_only/        # Reflect-only configuration
```

## File Naming Convention

**Format:** `malice_YYMMDD_HHhMMmSSs_pPROCESS-ID_rRUN_tTICKETS_dDURATION.jsonl`

**Components:**

- `malice`: Experiment type identifier
- `250614`: Date (YYMMDD)
- `21h24m34s`: Start time
- `p25656`: Process ID
- `r01`: Run number
- `t60`: Tickets per run
- `d16m`: Duration

**Example:** `malice_250614_21h24m34s_p25656_r01_t60_d16m.jsonl`

## Data Structure

### Main Event Logs (`events/*.jsonl`)

- `VOTE_PROCESSING`: Vote events and processing
- `STATUS_CHANGE`: Agent promotion/demotion events
- `GATE_DECISION`: Security and civil gate decisions
- `REPUTATION_PENALTY`: Reputation changes
- `SYMBOL_INTERPRETATION`: Symbol processing timeline

### System Metrics (`metrics/*_metrics.jsonl`)

- `IMMUNE_RESPONSE_ADJUSTMENT`: Adaptive immune system data
- `TIMING_METRIC`: Performance monitoring

### Batch Results (`batch_results/*.json`)

- Experiment summaries and statistics

## Usage

### Running Experiments

```bash
# Run batch experiments
python batch_runner_parallel.py --runs 30 --tickets 60 --tag "experiment_name"
```

### Data Analysis

```python
# Load experiment data
from validation.data_parsing import load_experiment_logs_with_metrics

main_logs, metrics_logs, metrics_logs = load_experiment_logs_with_metrics(
    Path("exp_output/published_data/batch_09_adaptive_immune/")
)
```

### Quality Control

```bash
# Validate experiment logs
python qc/qc_batch.py exp_output/published_data/*/events/*.jsonl
```

## Data Organization

Each experiment batch contains:

1. **Events**: Primary events
2. **Metrics**: Telemetry and performance data
3. **Results**: Analysis summaries and statistics
