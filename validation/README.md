# Validation Scripts

Validation and analysis tools.

## Core Tools

### `data_parsing.py` - Standardized Parsing Functions

Reusable parsing functions for experiment data:

- `load_experiment_logs_with_metrics()` - Load main, and metrics logs
- `extract_binder_promotions()` - Parse STATUS_CHANGE events for promotions
- `detect_alliance_patterns()` - Identify coordinated voting behaviors
- `analyze_immune_stabilization()` - Extract immune system adjustments

**Usage**:

```python
from validation.data_parsing import load_experiment_logs_with_metrics, extract_binder_promotions
main_logs, metrics_logs, metrics_logs = load_experiment_logs_with_metrics(batch_dir)
promotions = extract_binder_promotions(main_logs)
```

## Validation Scripts

### `validate_symbol_data.py` - Core Data Validation

Validates data quality and coordination:

```bash
python validation/validate_symbol_data.py exp_output/batch_name/events/*.jsonl
```

### `analyze_symbol_journeys.py` - Symbol Evolution Analysis

Comprehensive symbol timeline analysis:

```bash
python validation/analyze_symbol_journeys.py exp_output/batch_name/events/*.jsonl --output analysis.md
```

### `analyze_validation_batch.py` - Batch Analysis

Automated batch analysis and comparison:

```bash
python validation/analyze_validation_batch.py exp_output/batch_name/events/*.jsonl --tag "batch_name"
```

### `cross_batch_analyzer.py` - Cross-Batch Analysis

Analysis across multiple experimental batches:

```bash
python validation/cross_batch_analyzer.py
```

### `autocorrelation_analysis.py` - Statistical Independence

Statistical validation for research requirements:

```bash
python validation/autocorrelation_analysis.py exp_output/published_data/*/events/*.jsonl --output report.txt
```

## Workflow

### Basic Analysis Pipeline

```bash
# 1. Validate data quality
python validation/validate_symbol_data.py exp_output/batch_name/events/*.jsonl

# 2. Generate analysis
python validation/analyze_symbol_journeys.py exp_output/batch_name/events/*.jsonl --output analysis.md

# 3. Create batch report
python validation/analyze_validation_batch.py exp_output/batch_name/events/*.jsonl --tag "batch_name"
```

### Integration with QC

```bash
python qc/qc_batch.py exp_output/batch_name/events/*.jsonl
python validation/validate_symbol_data.py exp_output/batch_name/events/*.jsonl
```

## Analysis Features

These scripts analyze:

- Symbol interpretation patterns
- Agent coordination behaviors
- System stability metrics
- Statistical independence
- Social dynamics evolution

## Output

Scripts generate reports in multiple formats:

- JSON data for programmatic analysis
- Markdown reports for documentation
- Statistical summaries for research
