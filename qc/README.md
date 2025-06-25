# DSBL Quality Control (QC) Tools

**Note**: These QC tools are primarily of historical value for the Multi-Agent Adaptive Immune System. They remain functional for data validation but are rarely needed due to system stability.

QC scripts validate JSONL audit logs from batch experiments to detect data integrity issues.

## Usage

```bash
# Check specific adaptive immune system batch
python qc/qc_batch.py exp_output/adaptive_test/events/malice_250614_21h24m34s_p25656_r01_t60_d16m.jsonl

# Check all completed adaptive experiments
python qc/qc_batch.py exp_output/adaptive_test/events/*.jsonl

# Real-time monitoring of active experiments (legacy)
python qc/qc_realtime.py exp_output/
```

## Quality Checks Performed

| Check | Description | Failure Criteria |
|-------|-------------|-------------------|
| **Ticket coverage** | Sufficient experimental data | < 60% coverage of ticket range |
| **Message ID** | Unique & ascending message_id | Duplicates or non-monotonic |
| **Ticket chronology** | Proper ticket ordering | > 10% out-of-order |
| **Status consistency** | Demoted users cannot vote | Vote from demoted user |
| **BINDER emergence** | At least one promotion | 0 BINDER promotions (warning) |
| **Gate consistency** | ALLOWED gates → no penalties | ALLOWED followed by penalty |
| **Adaptive responses** | Immune system adjustments | Missing frequency adjustments (v2.11) |

### Ticket Coverage Calculation

QC calculates tickets as **coverage of ticket range** rather than exact count:
- **Range**: `max(ticket_numbers) - min(ticket_numbers) + 1`  
- **Coverage**: `unique_tickets / ticket_range`
- **Normal**: 65-82% coverage (due to agent voting_frequency < 1.0)
- **Problem**: < 60% coverage indicates experiment too short or inactive agents

### What Gets Tickets (v2.8+ with Adaptive Immune System)

**Only agent messages receive tickets:**
- ✅ Agent-generated messages during simulation
- ✅ All gate-processing events for same agent message use same ticket
- ✅ Adaptive immune system frequency adjustments (v2.11)
- ❌ System events (STATUS_CHANGE, isolated gate decisions)  
- ❌ Human messages in interactive mode

**Note**: System ensured 100% of tickets represent agent actions → predictable BINDER emergence and adaptive responses

## Output

- **🟢 PASS**: All checks approved
- **🟡 PASS (with warnings)**: Approved but with warnings  
- **🔴 FAIL**: Critical issues found

### Save QC Reports

```bash
# Save QC batch report to file
python qc/qc_batch.py exp_output/adaptive_test/events/*.jsonl > qc/reports/qc_batch_adaptive_test_$(date +%Y%m%d_%H%M%S).txt

# Save real-time analysis report  
python qc/qc_realtime.py exp_output/ > qc/reports/qc_realtime_$(date +%Y%m%d_%H%M%S).txt

# Create reports directory if it doesn't exist
mkdir -p qc/reports
```

## Common Issues (Legacy Context)

**Low ticket coverage**: Normal agent activity gaps
- **Explanation**: Agents have voting_frequency < 1.0, creating natural gaps
- **Solution**: If < 60% coverage, check experiment length or agent frequencies

**Gate inconsistencies**: ALLOWED gates followed by penalties  
- **Solution**: Check gate logic in core/gate_processor.py

**No BINDER emergence**: Threshold too high for ticket count
- **Solution**: Lower promotion_threshold or increase tickets

**Missing adaptive responses** (v2.11): No immune system adjustments detected
- **Solution**: Verify adaptive immune system is enabled and functioning

## Dependencies

Only Python standard library required.

## Legacy Status

These QC tools were essential during system development but are rarely needed with the stable Multi-Agent Adaptive Immune System. They remain available for troubleshooting or validating experimental data integrity.