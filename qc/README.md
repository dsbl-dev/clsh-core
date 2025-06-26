# DSBL Quality Control Tools

QC scripts validate JSONL audit logs from batch experiments to detect data integrity issues.

## Usage

```bash
# Check specific batch
python qc/qc_batch.py exp_output/batch_name/events/malice_*.jsonl

# Check all completed experiments
python qc/qc_batch.py exp_output/batch_name/events/*.jsonl

# Real-time monitoring
python qc/qc_realtime.py exp_output/
```

## Quality Checks

| Check | Description | Failure Criteria |
|-------|-------------|-------------------|
| **Ticket coverage** | Sufficient experimental data | < 60% coverage of ticket range |
| **Message ID** | Unique & ascending message_id | Duplicates or non-monotonic |
| **Ticket chronology** | Proper ticket ordering | > 10% out-of-order |
| **Status consistency** | Demoted users cannot vote | Vote from demoted user |
| **BINDER emergence** | At least one promotion | 0 BINDER promotions (warning) |
| **Gate consistency** | Gate decisions consistency | Inconsistent gate behavior |
| **Adaptive responses** | System adjustments present | Missing expected adjustments |

## Output

- **🟢 PASS**: All checks approved
- **🟡 PASS (with warnings)**: Approved with warnings  
- **🔴 FAIL**: Critical issues found

## Common Issues

**Low ticket coverage**: Normal agent activity gaps
- Agents have voting_frequency < 1.0, creating natural gaps
- If < 60% coverage, check experiment length

**Gate inconsistencies**: Check gate logic consistency

**No BINDER emergence**: Threshold may be too high for ticket count

## Dependencies

Python standard library only.