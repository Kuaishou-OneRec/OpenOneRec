# Benchmark

---

## Quick Start

### Requirements

- **Python**: >= 3.10
- **CUDA**: GPU environment with CUDA support
- **Hardware**: Multi-GPU setup recommended for faster evaluation

**Main Dependencies**:
- PyTorch 2.5.1
- Transformers 4.52.0
- Ray 2.43.0
- vLLM 0.7.3

### Step 1: Prepare Evaluation Data

Place the evaluation data in the project directory:

```bash
# Data directory structure
./data_v1.0/
├── rec_reason/
├── item_understand/
├── ad/
├── product/
├── label_cond/
├── video/
├── interactive/
└── label_pred/
```

### Step 2: Install Dependencies

```bash
# Navigate to benchmarks directory
cd benchmarks

# Install project (without dependencies to avoid overwriting existing environment)
pip install -e . --no-deps --no-build-isolation

# For full installation with all dependencies
pip install -e .
```

### Step 3: Start Ray Cluster

```bash
# Initialize multi-node multi-GPU environment
bash scripts/init_ray_cluster.sh
```

### Step 4: Run Evaluation

```bash
# Set environment variables
export BENCHMARK_BASE_DIR="./benchmarks"
export DATA_VERSION="v1.0"

# Run evaluation script
bash eval_script.sh <model_path> <result_name> <enable_thinking>
```

**Parameters**:
| Parameter | Description | Example |
|-----------|-------------|---------|
| model_path | Path to the model to evaluate | `model_output/sft/global_step10/converted` |
| result_name | Name identifier for output directory | `sft_nonthink` |
| enable_thinking | `true` or `false` | `false` |

**Examples**:
```bash
# Without thinking mode
bash eval_script.sh \
    model_output/sft/global_step10/converted \
    sft_nonthink \
    false

# With thinking mode
bash eval_script.sh \
    model_output/sft/global_step10/converted \
    sft_think \
    true
```

### Step 5: View Results

After evaluation completes, results are saved in:
```
./benchmarks/results/v1.0/results_<result_name>/
```

Log files are located at:
```
./benchmarks/auto_eval_logs/v1.0/<result_name>.log
```


---

## Evaluation Tasks

| Task Name | Source | Description |
|-----------|--------|-------------|
| label_cond | Kuaishou Internal | 34,891 | Predict next video given specified consumption behavior |
| video | Kuaishou Internal | 38,781  | Next video prediction |
| product | Kuaishou Internal | 28,536 | Predict next clicked product |
| ad | Kuaishou Internal | 30,131 | Predict next clicked advertisement |
| interactive | Kuaishou Internal | 1,000 | Predict next interacted video |
| label_pred | Kuaishou Internal | 346,190 | Predict user engagement with video content |
| item_understand | Kuaishou Internal | 474 | Video SID to Caption generation task |
| rec_reason | Kuaishou Internal | 300 | Recommendation reason inference |

