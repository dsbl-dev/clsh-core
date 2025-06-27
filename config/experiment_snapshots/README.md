# Experiment Configuration Snapshots

Frozen configuration files directory

## Available Snapshots

### `batch_09_11_settings.yaml`

**Used for**: Batches 09-11  
**Paper**: "Deferred Semantic Binding Language: Enabling Closed-Loop Social Homeostasis in Multi-Agent Systems"  
**Results**:

- 270/270 immune system activations
- 5,400 experimental tickets / 90 runs
- 33% computational social mobility

**Reproduce experiments**:

```bash
python main.py --config config/experiment_snapshots/batch_09_11_settings.yaml
```

## Key Parameters (Batch 09-11)

| Parameter                | Value | Description                              |
| ------------------------ | ----- | ---------------------------------------- |
| `promotion_threshold`    | 5     | Votes needed for BINDER promotion        |
| `binder_vote_multiplier` | 1.5   | Vote weight multiplier for BINDER agents |
| `toxicity_threshold`     | 0.7   | CIVIL gate threshold                     |
| `enable_reflect`         | true  | Pressure detection (Event A)             |
| `enable_calibrate`       | true  | Adaptive frequency adjustment (Event B)  |

For complete parameter list, see the actual snapshot file.

## Usage Notes

- **Current development**: Use `../settings.yaml` for ongoing experiments
- **Research reproducibility**: Use snapshot files for exact replication
