# Benchmark

For an optional optimization of long-history, short-SID beam search, see
[compact beam search](COMPACT_BEAM_SEARCH.md), including compatibility checks
and a paired native/compact comparison command.

## Quick Start

### Step 1: Install Dependencies

```bash
cd benchmarks

conda create -n benchmark python=3.10 
conda activate benchmark
pip install uv
uv pip install torch==2.5.1 transformers==4.52.0 vllm==0.7.3
pip install -r requirements.txt
pip install -e . --no-deps --no-build-isolation
```

### Step 2: Start Ray Cluster (Optional)

```bash
# Initialize multi-node multi-GPU environment
# Skip this step if using single-node multi-GPU setup
bash scripts/init_ray_cluster.sh
```


### Step 3: Configure LLM API

Edit `api/config/llm_config.json` to fill in your Gemini configuration:

```json
{
  "gemini": {
    "project": "<your-project>",
    "location": "<your-location>",
    "credentials_path": "<path-to-credentials>",
    ...
  }
}
```

**Note**: Only `project`, `location`, and `credentials_path` need to be configured. 

Test the configuration:

```python
from api import get_client_from_config

# Create client
client = get_client_from_config("gemini")

# Generate text
response = client.generate("Tell me a joke")
print(response)
```

### Step 4: Run Evaluation

```bash
export BENCHMARK_BASE_DIR="."
export BENCHMARK_DATA_DIR="../raw_data/onerec_data/benchmark_data"
export DATA_VERSION="v1.0"

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
    /path/to/model \
    model_nonthink \
    false

# With thinking mode
bash eval_script.sh \
    /path/to/model \
    model_think \
    true
```

For debugging purposes, you can add `--sample_size 10` to each python command in `eval_script.sh` to run evaluation on a smaller subset of data.


### Step 5: View Results

After evaluation completes, results are saved in:
```
./results/v1.0/results_<result_name>/
```

Log files are located at:
```
./auto_eval_logs/v1.0/<result_name>.log
```


---

## Evaluation Tasks

| Task Name | Source | Description |
|-----------|--------|-------------|
| ad | Kuaishou Internal | 27,677 | Predict next clicked advertisement |
| product | Kuaishou Internal | 27,910 | Predict next clicked product |
| interactive | Kuaishou Internal | 1,000 | Predict next interacted video |
| video | Kuaishou Internal | 38,781  | Next video prediction |
| label_cond | Kuaishou Internal | 34,891 | Predict next video given specified consumption behavior |
| label_pred | Kuaishou Internal | 346,190 | Predict user engagement with video content |
| item_understand | Kuaishou Internal | 500 | Video SID to Caption generation task |
| rec_reason | Kuaishou Internal | 470 | Recommendation reason inference |



