# DSBL Validation Scripts - Multi-Agent Adaptive Immune System

Comprehensive validation and analysis tools for Multi-Agent Adaptive Immune System experiments.

## Data Structure Reference

### **`README_datastructure.md`** - Essential Parsing Reference
**Purpose**: Complete JSON structure reference and parsing patterns for all validation scripts  
**Features**:
- Exact JSON examples from production logs (batches 9-11)
- Copy-paste parsing templates for common operations
- Common pitfalls and solutions documentation
- Quick reference for event type locations

**Usage**: **Read this first** before developing any new validation script

### **`data_parsing.py`** - Standardized Parsing Functions
**Purpose**: Reusable parsing functions implementing documented data structure patterns  
**Features**:
- `load_experiment_logs_with_metrics()` - Load main, debug, and metrics logs with consistent structure
- `extract_binder_promotions()` - Parse STATUS_CHANGE events for BINDER promotions
- `detect_alliance_patterns()` - Identify coordinated voting behaviors
- `analyze_immune_stabilization()` - Extract immune system adjustments from debug logs

**Usage**:
```python
from validation.data_parsing import load_experiment_logs_with_metrics, extract_binder_promotions
main_logs, debug_logs, metrics_logs = load_experiment_logs_with_metrics(batch_dir)
promotions = extract_binder_promotions(main_logs)
```

---

## Validation Scripts

### `validate_symbol_data.py` - Core Data Validation
**Purpose**: Validates experiment data quality and Multi-Agent coordination
**Features**:
- Symbol interpretation tracking
- Adaptive immune system event validation
- Multi-Agent frequency adjustment verification
- Data quality scoring and recommendations

**Usage**:
```bash
# Validate recent adaptive immune experiments
python validation/validate_symbol_data.py exp_output/adaptive_test/events/*.jsonl

# Auto-discover and validate latest logs
python validation/validate_symbol_data.py
```

### `analyze_symbol_journeys.py` - Symbol Evolution Analysis
**Purpose**: Comprehensive symbol timeline analysis for adaptive systems
**Features**:
- Symbol journey evolution patterns
- Multi-Agent coordination analysis
- Adaptive frequency adjustment tracking
- BINDER emergence patterns with agent lifecycle analysis

**Usage**:
```bash
# Generate symbol journey report with Multi-Agent analysis
python validation/analyze_symbol_journeys.py exp_output/adaptive_test/events/*.jsonl --output symbol_analysis.md --json

# Analyze massive batch with coordinated agents
python validation/analyze_symbol_journeys.py exp_output/multi_agent_massive/events/*.jsonl --verbose
```

### `analyze_validation_batch.py` - Batch Comparison Analysis
**Purpose**: Automated validation batch analysis with era comparisons
**Features**:
- Multi-era comparison (Eve era, Balanced era, Adaptive immune era, Multi-Agent era)
- Alice dominance assessment with adaptive immune context
- Mallory integration analysis
- Publication-ready batch reports

**Usage**:
```bash
# Analyze validation batch with adaptive immune system context
python validation/analyze_validation_batch.py exp_output/adaptive_test/events/*.jsonl --tag "adaptive_validation"

# Generate comprehensive era comparison
python validation/analyze_validation_batch.py exp_output/multi_agent_massive/events/*.jsonl --tag "multi_agent_analysis"
```

### `cross_batch_analyzer.py` - Cross-Batch Consistency Analysis
**Purpose**: Comprehensive analysis across multiple experimental batches
**Features**:
- Multi-batch immune system consistency validation
- Social dynamics variation analysis across batches
- Cross-batch statistical comparisons
- Publication-ready comprehensive reports

**Usage**:
```bash
# Analyze all published batches (09-11)
python validation/cross_batch_analyzer.py
```

### `autocorrelation_analysis.py` - Statistical Independence Validation
**Purpose**: Statistical validation for academic publication requirements
**Features**:
- Durbin-Watson test for serial correlation
- Lag correlation analysis with confidence intervals
- Block-bootstrapping for robust statistics
- Agent-specific autocorrelation functions

**Usage**:
```bash
# Validate statistical independence
python validation/autocorrelation_analysis.py exp_output/published_data/batch_*/events/*.jsonl --output autocorr_report.txt
```

### `simple_autocorrelation_analysis.py` - Basic Statistical Validation
**Purpose**: Simplified statistical independence testing
**Features**:
- Basic autocorrelation analysis
- Durbin-Watson statistic computation
- Academic reviewer response validation

### `social_epoch_analysis.py` - Social Evolution Analysis
**Purpose**: Agent social behavior evolution over time
**Features**:
- Social epoch identification and classification
- Agent emergence and dominance patterns
- Temporal social dynamics analysis

## Multi-Agent Adaptive Immune System Features

All validation scripts support adaptive immune system features:

### **Adaptive Immune System Events**
- `IMMUNE_FREQUENCY_ADJUSTMENT` - Coordinated frequency changes (Eve, Dave, Zara)
- `ADAPTIVE_ADJUSTMENT` - System pressure detection and response
- Coordinated agent response tracking (0.18 ↔ 0.38)

### **Agent Lifecycle Analysis**
- **Alice dominance fatigue cycles** (67% → 0% → comeback patterns)
- **Bob opportunistic expansion** timing and competitive intelligence
- **Dave/Zara emergence** through adaptive coordination
- **Eve adaptive recovery** from 0% to meaningful participation

### **Multi-Agent Coordination Metrics**
- Frequency adjustment correlation between adaptive agents
- Synchronized response timing analysis
- Competitive balance maintenance tracking
- System self-regulation effectiveness

## Workflow Integration

### Production Analysis Pipeline
```bash
# 1. Validate data quality
python validation/validate_symbol_data.py exp_output/batch_tag/events/*.jsonl

# 2. Generate symbol journey analysis
python validation/analyze_symbol_journeys.py exp_output/batch_tag/events/*.jsonl --json --output reports/symbol_analysis.md

# 3. Create batch comparison report
python validation/analyze_validation_batch.py exp_output/batch_tag/events/*.jsonl --tag "batch_name"
```

### Quality Control Integration
```bash
# Combined with QC workflow
python qc/qc_batch.py exp_output/batch_tag/events/*.jsonl
python validation/validate_symbol_data.py exp_output/batch_tag/events/*.jsonl
```

## Output Integration

### Reports Directory
All validation scripts integrate with `reports/` directory structure:
- Automated report generation with timestamped filenames
- JSON and Markdown output formats
- Era comparison matrices
- Publication-ready analysis summaries

### Data Pipeline
```
exp_output/batch_tag/ → validation/ → reports/ → publication
```

## Analysis Capabilities

These validation scripts enable analysis of:
- **Multi-Agent adaptive coordination** in AI social systems
- **Coordinated frequency synchronization** across multiple agents
- **Agent lifecycle evolution** patterns over experimental sequences
- **System self-regulation** without human intervention
- **Adaptive competitive intelligence** between AI agents

---

**Contact**: Use these tools to validate and analyze Multi-Agent Adaptive Immune System experiments for publication-ready research.