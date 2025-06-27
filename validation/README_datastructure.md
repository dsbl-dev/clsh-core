# Log Data Structure Reference - MAA Immune System

Purpose: Comprehensive JSON structure reference for validation script development  
Updated: 2025-06-22 (added metrics separation, validated against batches 9, 10, 11)

This document provides exact JSON structures and parsing patterns for experiment logs, eliminating guesswork when developing new validation scripts.

---

## Log File Architecture

### Dual-Stream Pipeline

```
exp_output/batch_tag/
├── events/
│   └── malice_YYMMDD_HHhMMmSSs_pPID_rRUN_tTICKETS_dDURATION.jsonl    # Main logs
├── metrics/
│   ├── malice_YYMMDD_HHhMMmSSs_pPID_rRUN_tTICKETS_metrics_dDURATION.jsonl  # System metrics
│   └── malice_YYMMDD_HHhMMmSSs_pPID_rRUN_tTICKETS_metrics_dDURATION.jsonl   # Legacy metrics logs
└── batch_results/
    └── batch_summary.json    # Analysis results
```

Data Distribution:

- Main logs: Analysis events (SYMBOL_INTERPRETATION, VOTE_PROCESSING, GATE_DECISION)
- Metrics logs: System telemetry (IMMUNE_RESPONSE_ADJUSTMENT, TIMING_METRIC, etc.)
- Legacy metrics logs: Historical metrics information (CONFIG_CHECK, etc.)

---

## Main Log Event Structures

### SYMBOL_INTERPRETATION - Vote Events

```json
{
  "timestamp": "2025-06-17T17:45:14",
  "event_type": "SYMBOL_INTERPRETATION",
  "details": {
    "symbol_type": "VOTE",
    "symbol_content": "promote_alice_+1",
    "interpreter": "vote_processor",
    "context": {
      "ticket": "#1001",
      "author": "bob",
      "vote_value": 1,
      "raw_symbol": "⟦VOTE:promote_alice⟧ +1"
    },
    "interpretation": {
      "action": "vote_parsed",
      "target": "promote_alice", // Extract vote target here
      "vote_value": 1,
      "valid": true
    }
  },
  "ticket": "#1001",
  "timestamp_detailed": "2025-06-17T17:45:14.108948"
}
```

Parsing Pattern:

```python
# Extract vote target
if (event.get('event_type') == 'SYMBOL_INTERPRETATION' and
    event.get('details', {}).get('symbol_type') == 'VOTE'):
    target = event.get('details', {}).get('interpretation', {}).get('target')
    ticket = event.get('details', {}).get('context', {}).get('ticket', '#0')
```

### SYMBOL_INTERPRETATION - BINDER Promotions

```json
{
  "timestamp": "2025-06-17T17:45:32",
  "event_type": "SYMBOL_INTERPRETATION",
  "details": {
    "symbol_type": "STATUS_CHANGE", // BINDER promotions have this type
    "symbol_content": "BINDER_PROMOTION",
    "interpreter": "status_processor",
    "context": {
      "ticket": "#1015", // Ticket number here
      "agent": "alice",
      "promotion_reason": "vote_threshold_reached"
    },
    "interpretation": {
      "new_status": "BINDER", // New status here
      "previous_status": "regular",
      "status_change": "PROMOTION"
    }
  }
}
```

Parsing Pattern:

```python
# Extract BINDER promotions
if (event.get('event_type') == 'SYMBOL_INTERPRETATION' and
    event.get('details', {}).get('symbol_type') == 'STATUS_CHANGE'):
    interpretation = event.get('details', {}).get('interpretation', {})
    if interpretation.get('new_status') == 'BINDER':
        context = event.get('details', {}).get('context', {})
        ticket_str = context.get('ticket', '#0')
        ticket_num = int(ticket_str.replace('#', '')) if ticket_str.startswith('#') else 0
```

### VOTE_PROCESSING - Vote Mechanics

```json
{
  "timestamp": "2025-06-17T17:45:14",
  "event_type": "VOTE_PROCESSING",
  "details": {
    "voter": "bob",
    "target": "promote_alice",
    "value": 1,
    "actual_value": 1,
    "is_self_vote": false,
    "self_vote_weight": null,
    "is_binder_vote": false,
    "binder_multiplier": null,
    "message_id": 0,
    "ticket": "#1001"
  }
}
```

### GATE_DECISION - Security and Civility

```json
{
  "timestamp": "2025-06-17T17:45:32",
  "event_type": "GATE_DECISION",
  "details": {
    "gate": "civil",
    "result": "BLOCKED",
    "reason": "Toxic pattern detected: 'pity points'",
    "content_hash": "828597c80b",
    "detection_method": "pattern_fallback"
  }
}
```

---

## Metrics Log Event Structures

### IMMUNE_RESPONSE_ADJUSTMENT - Multi-Agent Coordination

```json
{
  "timestamp": "2025-06-17T17:45:45",
  "event_type": "IMMUNE_RESPONSE_ADJUSTMENT",
  "details": {
    "ticket": 8, // Ticket where adjustment occurred
    "pressure_level": "LOW",
    "recent_promotions": 0,
    "monitoring_window": 12,
    "agents_adjusted": ["eve", "dave", "zara"],
    "agent_details": {
      "eve": {
        "frequency_before": 0.26, // Before/after frequencies
        "frequency_after": 0.18,
        "adjustment_reason": "promotion_drought_detected",
        "safety_cap_applied": true
      },
      "dave": {
        "frequency_before": 0.26,
        "frequency_after": 0.18,
        "adjustment_reason": "promotion_drought_detected",
        "safety_cap_applied": true
      },
      "zara": {
        "frequency_before": 0.26,
        "frequency_after": 0.18,
        "adjustment_reason": "promotion_drought_detected",
        "safety_cap_applied": true
      }
    },
    "boost_multiplier_used": null,
    "reduction_multiplier_used": 0.65, // Actual multiplier applied
    "memory_dampening_applied": false
  }
}
```

Parsing Pattern:

```python
# Extract immune system adjustments from metrics logs (preferred) or legacy metrics logs (fallback)
if event.get('event_type') == 'IMMUNE_RESPONSE_ADJUSTMENT':
    ticket = event.get('details', {}).get('ticket', 0)
    agent_details = event.get('details', {}).get('agent_details', {})
    for agent, details in agent_details.items():
        freq_before = details.get('frequency_before', 0)
        freq_after = details.get('frequency_after', 0)
```

### TIMING_METRIC - Performance Analysis

```json
{
  "timestamp": "2025-06-17T17:45:14",
  "event_type": "TIMING_METRIC",
  "details": {
    "checkpoint": "ticket_complete",
    "ticket": "#1001",
    "duration_seconds": 1.235,
    "operation": "full_ticket_processing",
    "performance_warning": false
  }
}
```

---

## Legacy Metrics Log Event Structures

Legacy metrics logs now contain primarily technical information and configuration events:

### CONFIG_CHECK - Settings Validation

```json
{
  "timestamp": "2025-06-17T17:45:14",
  "event_type": "CONFIG_CHECK",
  "details": {
    "enable_calibrate": true,
    "source": "settings.yaml",
    "timestamp": "2025-06-17 17:45:14.108948"
  }
}
```

### CONFIG_ERROR - Configuration Issues

```json
{
  "timestamp": "2025-06-17T17:45:14",
  "event_type": "CONFIG_ERROR",
  "details": {
    "error": "Failed to load settings",
    "defaulting_to": true
  }
}
```

---

## Parsing Implementation Templates

### Complete Log Loading Function

```python
import json
from pathlib import Path
from typing import List, Dict, Tuple

def load_experiment_logs_with_metrics(batch_dir: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load main, legacy, and metrics logs from batch directory."""
    main_logs, legacy_logs, metrics_logs = [], [], []

    # Load main logs from events/ directory
    events_dir = batch_dir / "events"
    if events_dir.exists():
        for main_file in events_dir.glob("*.jsonl"):
            with open(main_file, 'r') as f:
                for line in f:
                    try:
                        main_logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    # Load legacy logs from metrics/ directory (legacy location)
    metrics_dir = batch_dir / "metrics"
    if metrics_dir.exists():
        for legacy_file in metrics_dir.glob("*_metrics.jsonl"):
            with open(legacy_file, 'r') as f:
                for line in f:
                    try:
                        legacy_logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    # Load metrics logs from metrics/ directory
    if metrics_dir.exists():
        for metrics_file in metrics_dir.glob("*_metrics.jsonl"):
            with open(metrics_file, 'r') as f:
                for line in f:
                    try:
                        metrics_logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    return main_logs, legacy_logs, metrics_logs
```

### BINDER Promotion Analysis

```python
def extract_binder_promotions(main_logs: List[Dict]) -> List[Dict]:
    """Extract all BINDER promotion events with timing."""
    promotions = []

    for event in main_logs:
        if (event.get('event_type') == 'SYMBOL_INTERPRETATION' and
            event.get('details', {}).get('symbol_type') == 'STATUS_CHANGE'):

            interpretation = event.get('details', {}).get('interpretation', {})
            if interpretation.get('new_status') == 'BINDER':
                context = event.get('details', {}).get('context', {})

                # Extract ticket number (handle #1001 format)
                context_ticket = context.get('ticket', '#0')
                event_ticket = event.get('ticket', '#0')
                ticket_str = context_ticket if context_ticket != '#0' else event_ticket
                ticket_num = int(ticket_str.replace('#', '')) if ticket_str.startswith('#') else 0

                promotions.append({
                    'ticket': ticket_num,
                    'agent': context.get('agent', 'unknown'),
                    'timestamp': event.get('timestamp'),
                    'previous_status': interpretation.get('previous_status', 'unknown')
                })

    return sorted(promotions, key=lambda x: x['ticket'])
```

### Alliance Formation Detection

```python
def detect_alliance_patterns(main_logs: List[Dict], time_window: int = 3) -> List[Dict]:
    """Detect coordinated voting patterns (alliances)."""
    vote_events = []

    # Extract all vote events
    for event in main_logs:
        if (event.get('event_type') == 'SYMBOL_INTERPRETATION' and
            event.get('details', {}).get('symbol_type') == 'VOTE'):

            context = event.get('details', {}).get('context', {})
            interpretation = event.get('details', {}).get('interpretation', {})

            # Extract ticket number
            ticket_str = context.get('ticket', '#0')
            ticket_num = int(ticket_str.replace('#', '')) if ticket_str.startswith('#') else 0

            vote_events.append({
                'ticket': ticket_num,
                'voter': context.get('author', 'unknown'),
                'target': interpretation.get('target', 'unknown'),
                'timestamp': event.get('timestamp')
            })

    # Group by target within time windows
    alliances = []
    for target in set(vote['target'] for vote in vote_events):
        target_votes = [v for v in vote_events if v['target'] == target]
        target_votes.sort(key=lambda x: x['ticket'])

        # Detect clusters within time_window
        current_cluster = []
        for vote in target_votes:
            if not current_cluster or vote['ticket'] - current_cluster[-1]['ticket'] <= time_window:
                current_cluster.append(vote)
            else:
                if len(current_cluster) >= 2:  # Alliance = 2+ coordinated votes
                    alliances.append({
                        'target': target,
                        'participants': [v['voter'] for v in current_cluster],
                        'tickets': [v['ticket'] for v in current_cluster],
                        'duration': max(v['ticket'] for v in current_cluster) - min(v['ticket'] for v in current_cluster)
                    })
                current_cluster = [vote]

        # Handle final cluster
        if len(current_cluster) >= 2:
            alliances.append({
                'target': target,
                'participants': [v['voter'] for v in current_cluster],
                'tickets': [v['ticket'] for v in current_cluster],
                'duration': max(v['ticket'] for v in current_cluster) - min(v['ticket'] for v in current_cluster)
            })

    return alliances
```

### Immune System Stabilization Analysis

```python
def analyze_immune_stabilization(legacy_logs: List[Dict], metrics_logs: Optional[List[Dict]] = None) -> List[Dict]:
    """Analyze pressure detection -> stabilization timing from metrics or legacy logs."""
    immune_events = []

    # Prefer metrics_logs if available, fallback to legacy_logs
    source_logs = metrics_logs if metrics_logs else legacy_logs

    for event in source_logs:
        if event.get('event_type') == 'IMMUNE_RESPONSE_ADJUSTMENT':
            details = event.get('details', {})

            immune_events.append({
                'ticket': details.get('ticket', 0),
                'pressure_level': details.get('pressure_level', 'UNKNOWN'),
                'recent_promotions': details.get('recent_promotions', 0),
                'agents_adjusted': details.get('agents_adjusted', []),
                'reduction_multiplier': details.get('reduction_multiplier_used'),
                'boost_multiplier': details.get('boost_multiplier_used'),
                'timestamp': event.get('timestamp')
            })

    return sorted(immune_events, key=lambda x: x['ticket'])
```

---

## Common Parsing Pitfalls

### 1. Ticket Number Extraction

```python
# WRONG - ticket not always at event level
ticket = event.get('ticket', 0)

# CORRECT - check context first, fallback to event level
context_ticket = event.get('details', {}).get('context', {}).get('ticket', '#0')
event_ticket = event.get('ticket', '#0')
ticket_str = context_ticket if context_ticket != '#0' else event_ticket
ticket_num = int(ticket_str.replace('#', '')) if ticket_str.startswith('#') else 0
```

### 2. Vote Target Extraction

```python
# WRONG - target not directly in details
target = event.get('details', {}).get('target')

# CORRECT - target in interpretation sub-object
target = event.get('details', {}).get('interpretation', {}).get('target')
```

### 3. Legacy Log File Loading

```python
# WRONG - legacy logs have different naming pattern
legacy_file = main_file.replace('.jsonl', '_metrics.jsonl')

# CORRECT - legacy logs in metrics directory with specific pattern
metrics_dir = batch_dir / "metrics"
legacy_files = list(metrics_dir.glob("*_metrics.jsonl"))
```

### 4. Event Type Filtering

```python
# INCOMPLETE - only checking event_type
if event.get('event_type') == 'SYMBOL_INTERPRETATION':

# COMPLETE - also check symbol_type for specificity
if (event.get('event_type') == 'SYMBOL_INTERPRETATION' and
    event.get('details', {}).get('symbol_type') == 'VOTE'):
```

---

## Quick Reference: Event Type Location Map

| Analysis Need      | Event Type                                               | Log Stream                | Key Field                         |
| ------------------ | -------------------------------------------------------- | ------------------------- | --------------------------------- |
| Vote patterns      | `SYMBOL_INTERPRETATION` + `symbol_type: "VOTE"`          | Main                      | `interpretation.target`           |
| BINDER promotions  | `SYMBOL_INTERPRETATION` + `symbol_type: "STATUS_CHANGE"` | Main                      | `interpretation.new_status`       |
| Immune adjustments | `IMMUNE_RESPONSE_ADJUSTMENT`                             | Metrics (Legacy fallback) | `agent_details.*.frequency_after` |
| Gate decisions     | `GATE_DECISION`                                          | Main                      | `details.result`                  |
| Alliance timing    | `SYMBOL_INTERPRETATION` + `symbol_type: "VOTE"`          | Main                      | `context.ticket`                  |
| Performance data   | `TIMING_METRIC`                                          | Metrics (Legacy fallback) | `details.duration_seconds`        |
| Configuration      | `CONFIG_CHECK`, `CONFIG_ERROR`                           | Legacy                    | `details.enable_calibrate`        |

---

## Testing Your Parsing Code

### Validation Function

```python
def validate_log_parsing(batch_dir: Path) -> Dict[str, int]:
    """Validate parsing against known batch structure."""
    main_logs, legacy_logs = load_experiment_logs(batch_dir)

    stats = {
        'main_events': len(main_logs),
        'legacy_events': len(legacy_logs),
        'vote_events': len([e for e in main_logs if e.get('event_type') == 'SYMBOL_INTERPRETATION'
                           and e.get('details', {}).get('symbol_type') == 'VOTE']),
        'binder_promotions': len([e for e in main_logs if e.get('event_type') == 'SYMBOL_INTERPRETATION'
                                 and e.get('details', {}).get('symbol_type') == 'STATUS_CHANGE'
                                 and e.get('details', {}).get('interpretation', {}).get('new_status') == 'BINDER']),
        'immune_adjustments': len([e for e in legacy_logs if e.get('event_type') == 'IMMUNE_RESPONSE_ADJUSTMENT'])
    }

    return stats

# Test against known batch
test_stats = validate_log_parsing(Path("exp_output/published_data/batch_11_adaptive_immune"))
print(f"Validation results: {test_stats}")
```

---

## Usage Examples

### Complete Analysis Script Template

```python
#!/usr/bin/env python3
"""
Template for new validation scripts.
Copy this structure for consistent log parsing.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple

def main():
    # 1. Load logs using standard pattern (with metrics)
    batch_dir = Path("exp_output/published_data/batch_11_adaptive_immune")
    main_logs, metrics_logs, metrics_logs = load_experiment_logs_with_metrics(batch_dir)

    # 2. Extract specific data using documented patterns
    binder_promotions = extract_binder_promotions(main_logs)
    immune_adjustments = analyze_immune_stabilization(metrics_logs, metrics_logs)  # Prefer metrics
    alliances = detect_alliance_patterns(main_logs)

    # 3. Analyze and report
    print(f"Found {len(binder_promotions)} BINDER promotions")
    print(f"Found {len(immune_adjustments)} immune adjustments")
    print(f"Found {len(alliances)} alliance formations")
    print(f"Metrics events: {len(metrics_logs)}")

    # 4. Validate results
    validation = validate_log_parsing(batch_dir)
    print(f"Validation: {validation}")

if __name__ == "__main__":
    main()
```

---

## Integration with Analysis Pipeline

This data structure reference integrates with:

- Standardized Functions: `validation/data_parsing.py` - Implements these patterns as reusable functions
- Validation Scripts: All scripts in `validation/` directory follow these parsing patterns
- Research Publication: Complete analysis methodology and visualization examples available in the accompanying research paper

Usage Workflow:

1. Reference this document when developing new validation scripts
2. Import functions from `data_parsing.py` for consistent parsing
3. Follow patterns in existing validation scripts for analysis implementation

---

Status: Complete reference for batches 9-11 data structure  
Validated: 2025-06-21 against production logs  
Implementation: Available in `validation/data_parsing.py`  
Next Update: When new event types or structure changes are introduced
