# DSBL Experiment Output Directory - Multi-Agent Adaptive Immune System

This directory contains organized experiment output from parallel batch experiments with clear separation between published data, active experiments, and archived research.

## Organized Batch Structure

**Recommended approach**: Use `--tag` flag to organize experiments into dedicated directories.

```
exp_output/
├── published_data/                   # Data supporting arxiv publication
│   ├── batch_09_adaptive_immune/     # Publication dataset 1
│   │   ├── events/                   # Main experimental events
│   │   ├── metrics/                  # System telemetry data
│   │   └── batch_results/            # Analysis summaries
│   ├── batch_10_adaptive_immune/     # Publication dataset 2
│   └── batch_11_adaptive_immune/     # Publication dataset 3
├── ablation_baseline/                # Active experiments
│   ├── malice_*.jsonl                # Main event logs
│   ├── metrics/                      # System telemetry
│   └── batch_results/                # Analysis results
├── ablation_reflect_only/            # Active experiments
├── archive/                          # Historical/development data
│   ├── Arkiv/                        # Old experiments
│   ├── adaptive_immune_v2/           # Legacy versions
│   └── debug/                        # Old debug directories
└── [future_experiments]/             # New experiments go here directly
```

## File Naming Convention

**Format:** `malice_YYMMDD_HHhMMmSSs_pPROCESS-ID_rRUN_tTICKETS[_dDURATION].jsonl`

**Components:**

- `malice`: Experiment type (Multi-Agent Adaptive Immune System)
- `250614`: Date (YYMMDD)
- `21h24m34s`: Start time
- `p25656`: Process ID
- `r01`: Individual run number (not range)
- `t60`: Tickets per run
- `d16m`: Duration (added when complete)

**Example:** `malice_250614_21h24m34s_p25656_r01_t60_d16m.jsonl`

## Batch Experiment Usage

**Run organized batch experiments with tags:**

```bash
# Run Multi-Agent Adaptive Immune System experiment (adaptive immune automatically enabled)
python batch_runner_parallel.py --runs 30 --tickets 60 --tag "adaptive_test" --workers 8

# Creates directory structure:
exp_output/adaptive_test/
├── events/
│   ├── malice_250614_21h24m34s_p25656_r01_t60_d16m.jsonl
│   └── malice_250614_21h24m34s_p25657_r02_t60_d18m.jsonl
├── metrics/
│   ├── malice_250614_21h24m34s_p25656_r01_t60_metrics_d16m.jsonl
│   └── malice_250614_21h24m34s_p25657_r02_t60_metrics_d18m.jsonl
└── batch_results/
    └── adaptive_test_summary.json
```

**Benefits of tag organization:**

1. **Clean separation:** Each experiment batch in its own directory
2. **Easy analysis:** All related logs grouped together
3. **Quality control:** `python qc/qc_batch.py exp_output/adaptive_test/events/*.jsonl`
4. **Research workflow:** Clear experiment lineage for publication

## File States

### **Active Files** (no duration suffix)

- Experiment currently running
- Files being written to
- **Example:** `malice_250608_18h39m25s_p123456_r01-05_t8.jsonl`

### **Completed Files** (with `_dXXm` suffix)

- Experiment finished
- Complete dataset ready for analysis
- **Example:** `malice_250608_18h39m25s_p123456_r01-05_t8_d15m.jsonl`

## Data Contents - Dual-Stream Pipeline

### **Main Logs** (`exp_output/batch_tag/events/*.jsonl`) - Primary Analysis Stream

- `VOTE_PROCESSING`: Vote events with BINDER multipliers and adaptive frequencies
- `STATUS_CHANGE`: Promotion/demotion events with agent lifecycle patterns
- `GATE_DECISION`: Security and CIVIL gate decisions
- `REPUTATION_PENALTY`: Reputation changes and Mallory containment
- `SYMBOL_INTERPRETATION`: Complete symbol journey timeline including:
  - **VOTE** symbols: Standard voting interpretations
  - **VOTE_WEIGHTING**: Reputation-adjusted vote processing
  - **🆕 IMMUNE_ADJUSTMENT**: Multi-Agent Adaptive Immune System frequency adjustments
- Complete unified timeline for comprehensive analysis

### **Metrics Logs** (`exp_output/batch_tag/metrics/*_metrics.jsonl`) - System Telemetry Stream

- `IMMUNE_RESPONSE_ADJUSTMENT`: **Primary location** for immune system data
  - Agent coordination details (Eve, Dave, Zara)
  - Pressure level analysis (LOW/HIGH)
  - Promotion rate calculations
  - Frequency adjustment multipliers
- `TIMING_DEBUG`: Performance bottleneck analysis
  - Ticket processing timing
  - Message generation performance
  - Gate processing delays

### **Legacy Debug Files** (backward compatibility only)

- Some historical experiments may contain `*_debug.jsonl` files
- All current functionality is now in metrics files
- New experiments only generate events and metrics streams

### **🔗 Dual-Stream Pipeline Architecture**

**Clean Architecture**: Immune system events are logged to TWO semantically organized streams:

```
Adaptive Immune System Event
    ↓
    ├─ Main Log: SYMBOL_INTERPRETATION (adjust_frequency_eve/dave/zara)
    └─ Metrics Log: IMMUNE_RESPONSE_ADJUSTMENT (PRIMARY - system telemetry)
    ↓
Analysis tools read from both streams for complete data
    ↓
Semantically organized analysis with zero data gaps
```

**Key benefits:**

- **Semantic clarity**: Events = experimental data, Metrics = system telemetry
- **Clean organization**: No legacy compatibility overhead
- **Publication ready**: Professional data structure for research deployment

## Usage with Analysis Pipeline

**📖 Data Structure Reference**: [`validation/README_datastructure.md`](../validation/README_datastructure.md) - Complete JSON parsing guide for log analysis

```python
# Standardized parsing approach (recommended)
from validation.data_parsing import load_experiment_logs_with_metrics, extract_binder_promotions

# Load with dual-stream pipeline
main_logs, debug_logs, metrics_logs = load_experiment_logs_with_metrics(Path("exp_output/published_data/batch_10_adaptive_immune/"))

# Extract specific data using documented patterns
promotions = extract_binder_promotions(main_logs)
immune_adjustments = analyze_immune_stabilization(debug_logs, metrics_logs)  # Uses metrics

# Command-line validation (still supported)
python validation/analyze_symbol_journeys.py exp_output/published_data/batch_10_adaptive_immune/events/*.jsonl --output complete_analysis.txt
```

### **🎯 Key Benefits of Dual-Stream Architecture**

1. **Semantic Organization**: Clear separation between experimental events and system metrics
2. **Publication Ready**: Clean, professional data structure suitable for research deployment
3. **Data Consistency**: All analysis tools see identical immune system data
4. **Future Proof**: New analysis tools automatically get complete, well-organized dataset
5. **Zero Overhead**: No legacy compatibility burden on new experiments

## Multi-Batch Sequential Experiments

**Run multiple batches in sequence:**

```bash
# Run 3 sequential adaptive immune system batches (adaptive immune automatically enabled)
python batch_runner_parallel.py --runs 30 --tickets 60 --batches 3 --tag "adaptive_validation" --workers 8
```

**Creates organized structure:**

```
exp_output/adaptive_validation_1/
├── events/
│   ├── malice_250614_21h24m34s_p25656_r01_t60_d16m.jsonl
│   └── ... (30 experiments)
├── metrics/
│   ├── malice_250614_21h24m34s_p25656_r01_t60_metrics_d16m.jsonl
│   └── ... (30 metrics files)
└── batch_results/
    └── batch_1_summary.json
exp_output/adaptive_validation_2/
├── events/ ... (30 experiments)
├── metrics/ ... (30 metrics files)
└── batch_results/ ...
exp_output/adaptive_validation_3/
├── events/ ... (30 experiments)
├── metrics/ ... (30 metrics files)
└── batch_results/ ...
```

**Benefits:** Sequential analysis of system evolution across multiple experiment batches

## Quality Control

- Use `qc/qc_batch.py` to validate completed log files
- Timestamps enable correlation between QC results and batch summaries
- Both events and metrics logs share identical timing for cross-validation

---

## Metrics Logging (Always Enabled)

**Status:** Metrics logging is **permanently enabled** for complete immune system monitoring.

**Command:** `python batch_runner_parallel.py --runs 30 --tickets 60 --tag "batch_10_adaptive_immune"`

**Rationale:** Essential for capturing complete Multi-Agent Adaptive Immune System telemetry without data loss.

### Metrics Event Types

**🎫 TIMING_DEBUG:** Performance bottleneck analysis

- `checkpoint`: Timing at key stages (message generation, gate processing)
- `ticket_complete`: Full ticket timing summary
- **Alerts:** Operations >2s trigger console warnings

**🛡️ IMMUNE_RESPONSE_ADJUSTMENT:** System adaptation monitoring

- Multi-agent coordination details (Eve, Dave, Zara)
- Pressure level analysis and response timing
- Frequency adjustment multipliers and safety caps

### Performance Impact

- Minimal file size increase for essential telemetry
- No performance overhead on production data quality
- Complete system observability for research analysis
