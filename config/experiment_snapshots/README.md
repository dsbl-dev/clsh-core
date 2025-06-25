# Experiment Configuration Snapshots

This directory contains frozen configuration files used for specific research experiments, ensuring reproducibility of published results.

## Available Snapshots

### `batch_09_11_settings.yaml`
**Used for**: DSBL Multi-Agent Adaptive Immune System experiments (Batches 09-11)  
**Paper**: "Deferred Semantic Binding Language: Enabling Closed-Loop Social Homeostasis in Multi-Agent Systems"  
**Key Results**: 
- 270/270 immune system activations (100% reliability)
- 5,400 experimental tickets across 90 runs
- 33% computational social mobility (Mallory redemption)

**Reproduce experiments**:
```bash
python main.py --config config/experiment_snapshots/batch_09_11_settings.yaml
```

## Key Parameters (Batch 09-11)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `promotion_threshold` | 5 | Votes needed for BINDER promotion |
| `binder_vote_multiplier` | 1.5 | Vote weight multiplier for BINDER agents |
| `toxicity_threshold` | 0.7 | CIVIL gate threshold (balanced for Mallory participation) |
| `enable_reflect` | true | Pressure detection (Event A) |
| `enable_calibrate` | true | Adaptive frequency adjustment (Event B) |

For complete parameter list, see the actual snapshot file.

## Usage Notes

- **Current development**: Use `../settings.yaml` for ongoing experiments
- **Research reproducibility**: Use snapshot files for exact replication
- **Academic citations**: Reference specific snapshot file in methodology sections

---
**Created**: 2025-06-23  
**Commit**: Will be tagged when first public commit is made