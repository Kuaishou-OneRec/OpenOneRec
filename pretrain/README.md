# OpenOneRec Pretraining Module

The OpenOneRec pretraining module is based on the Qwen3 architecture, supporting a two-stage pretraining pipeline (Itemic-Text Alignment → Full-parameter Co-Pretraining) and SFT training workflow.

> **⚠️ Important Notice**
>
> The distributed training in this module supports **torchrun (recommended)** and **MPI (legacy)** for multi-node communication. The torchrun scripts live in `examples/*_torchrun.sh` and rely on standard PyTorch environment variables (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`), while the legacy scripts use `mpirun` and MPI hostfiles.
>
> To simplify environment configuration and improve reproducibility, we plan to release in future versions:
> - **Pre-configured Docker/Apptainer images**: Including all necessary dependencies and a ready-to-use distributed environment


## Quick Start

### Prerequisites

- **Hardware**: CUDA-enabled GPUs (multi-GPU or multi-node recommended)
- **Software**:
  - Python 3.8+
  - PyTorch (with FSDP and distributed training support)
  - OpenMPI (only if you use the legacy `mpirun` scripts)
  - NCCL (for GPU communication)
- **Data**: Training data converted to Parquet format (refer to `../data/README.md`)
- **Model**: Qwen3 base model (HuggingFace format)

### 1. Environment Setup

First, configure the training environment:

```bash
# Set environment variables
source set_env_torchrun.sh
```

This script sets necessary environment variables, including Python path, CUDA path, etc.

### 2. Qwen3 Model Vocabulary Expansion

Before starting training, you need to expand the vocabulary of the Qwen3 base model to support recommendation system-specific item ID encoding (itemic tokens).

#### 2.1 Configure Parameters

Edit `scripts/expand_qwen3_vocab.sh` and set the following parameters:

```bash
HF_MODEL_DIR=/path/to/Qwen3-0.6B          # Original Qwen3 HuggingFace model path
OUTPUT_MODEL_DIR=/path/to/Qwen3-0.6B_itemic  # Output model path with expanded vocabulary
ITEMIC_LAYER_N=3                          # Number of layers for itemic tokens
VOCAB_SIZE_PER_LAYER=8192                 # Vocabulary size expansion per layer
```

#### 2.2 Execute Expansion

```bash
bash scripts/expand_qwen3_vocab.sh
```

This script will:
- Add new itemic tokens on top of the original vocabulary
- Align vocabulary size to multiples of 256
- Initialize embedding weights for new tokens
- Save the expanded model to the specified directory

**Note**: The expanded model path needs to be used in the data configuration file for subsequent training (`base_model_dir` field).

### 3. Data Preparation

Training data needs to be converted to Parquet format. Please refer to `../data/README.md` for format specifications.

Data configuration is specified through JSON files located in the `examples/dataset_config/` directory.

#### Data Configuration Format

Each data configuration file contains the following main fields:

```json
{
    "name": "chat_completion_parquet",
    "sources": "/path/to/file_list.json",
    "base_model_dir": "/path/to/Qwen3-1.7B_itemic",
    "max_length": 30000,
    "num_epochs": 3,
    "num_workers": 2,
    "itemic_id_range": [151669, 176246],
    "add_think_pattern": false,
    "local_shuffle_buffer_size": 100000
    ...
}
```

### 4. Training

Training scripts are located in the `examples/` directory, and data configuration files are in the `examples/dataset_config/` directory.

#### 4.1 Stage1 Pretraining

Stage1 is mainly used for training itemic embeddings, typically freezing LLM parameters and only optimizing the embedding layer.

```bash
# Edit examples/pretrain_stg1_torchrun.sh to set model path, output path, and other parameters
bash examples/pretrain_stg1_torchrun.sh
```

Main training parameters (configured in `pretrain_stg1_torchrun.sh`):
- `--dataset_config examples/dataset_config/stg1.json`: Specify data configuration
- `--freeze_llm`: Freeze LLM parameters
- `--start_optimize_embedding_index 151669`: Start optimizing embeddings from the specified token ID
- `--model_dir`: Base model path with expanded vocabulary
- `--output_dir`: Model output path

**Note**: After training, convert the checkpoint to HuggingFace format (see [Model Conversion](#model-conversion)).

#### 4.2 Stage2 Pretraining

Stage2 is used for full-parameter pretraining to further optimize model performance. This stage unfreezes all model parameters and performs co-pretraining on a mixed domain of recommendation data and general text data.

```bash
# Edit examples/pretrain_stg2_torchrun.sh to set model path, output path, and other parameters
# MODEL_DIR should point to the converted hf model path from Stage1 training output
bash examples/pretrain_stg2_torchrun.sh
```

Main training parameters (configured in `pretrain_stg2_torchrun.sh`):
- `--dataset_config examples/dataset_config/pretrain.json`: Specify data configuration (including recommendation data and general text data)
- `--model_dir`: Converted model path from Stage1 output
- `--output_dir`: Model output path
- Note: **Does not include** `--freeze_llm` parameter, indicating full-parameter training

**Note**: After training, convert the checkpoint to HuggingFace format (see [Model Conversion](#model-conversion)).

#### 4.3 SFT Fine-tuning

SFT (Supervised Fine-Tuning) is used for instruction fine-tuning to improve model performance on specific tasks. This stage performs supervised learning on instruction-following data, enabling the model to better understand and execute recommendation-related instructions.

```bash
# Edit examples/posttrain_sft_torchrun.sh to set model path, output path, and other parameters
# MODEL_DIR should point to the converted hf model path from Stage2 training output
bash examples/posttrain_sft_torchrun.sh
```

Main training parameters (configured in `posttrain_sft_torchrun.sh`):
- `--dataset_config examples/dataset_config/sft.json`: Specify SFT data configuration
- `--model_dir`: Converted model path from Stage2 output
- `--output_dir`: Model output path
- `add_think_pattern: true` in data configuration enables thinking mode, which automatically adds `<think>` `</think>` tags and `/think` and `/no_think` instructions (for reasoning tasks)

**Note**: After training, convert the checkpoint to HuggingFace format (see [Model Conversion](#model-conversion)).

## Training Configuration

### Data Configuration Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Data loader name, default is `"chat_completion_parquet"` |
| `sources` | str | Data file list path (JSON file) or directory path list |
| `base_model_dir` | str | Base model path (with expanded vocabulary), used for tokenizing data |
| `max_length` | int | Maximum sequence length |
| `num_epochs` | int | Number of training epochs |
| `num_workers` | int | Number of dataloader workers |
| `model_class` | str | Model class name, default is `"Qwen3ForCausalLM"` |
| `itemic_id_range` | list | Itemic token ID range `[start, end]`, only used for metrics statistics |
| `only_assistant_loss` | bool | Whether to only compute loss for assistant responses, applies to chat format data |
| `local_shuffle_buffer_size` | int | Local sample-level shuffle buffer size |
| `add_think_pattern` | bool | Whether to add think tags (add `/think` `/no_think` in prompt, and `<think>` `</think>` in response) |

Notes:
* The default dataset is implemented based on torch.utils.data.IterableDataset
* By default, one GPU is bound to one process, each process creates `num_workers` workers. The dataset distributes files from `sources` to each worker at file granularity based on total worker count. The file list is shuffled before distribution, and sample-level shuffle is performed according to `local_shuffle_buffer_size` when reading data
* If `num_epochs` > 1, file distribution is performed twice, with the file list reshuffled each time


### Training Parameters

Main training parameters are passed via command line to `recipes/train_qwen3.py`:

| Parameter | Description |
|-----------|-------------|
| `--model_dir` | Base model path (HuggingFace format) |
| `--output_dir` | Model output path |
| `--dataset_config` | Data configuration file path |
| `--freeze_llm` | Whether to freeze LLM parameters |
| `--learning_rate` | Learning rate |
| `--max_length` | Sequence length per step |
| `--min_lr` | Minimum learning rate |
| `--lr_scheduler_type` | Learning rate scheduler type (e.g., `cosine`) |
| `--num_training_steps` | Number of training steps |
| `--save_checkpoint_per_step` | Save checkpoint every N steps |
| `--minibatch_size` | LLM head chunk size for chunked loss computation to save memory |
| `--resume_from` | Checkpoint directory path to resume training from |
| `--resume_from_tag` | Checkpoint tag to resume from (e.g., `global_step1000`) |
| `--resume_training_state` | Whether to restore full training state (including optimizer, lr scheduler, and dataloader state) |
| `--start_optimize_embedding_index` | Start optimizing embeddings from the specified token ID (for Stage1 training, typically set to the starting ID of itemic tokens, e.g., 151669) |
| `--use_tie_weights` | Tie embedding and lm_head weights (required for smaller models like 0.6B / 1.7B / 4B to align with Qwen3 model configuration) |

Notes:
* `resume_from` is used to load checkpoints produced by the framework. When `resume_from` is configured, it takes priority; only model structure parameters from `model_dir` are loaded for initialization. If not configured, parameters from `model_dir` are also loaded
* `num_training_steps` only affects the lr decay steps. This configuration ensures that when training reaches `num_training_steps`, lr decays to minimum, but training will not stop. It is recommended to configure based on token count and `max_length` to calculate the maximum training steps
* `max_length` represents the maximum sequence length per GPU per step; the framework will perform packing based on this configuration

## Utility Scripts

### Model Conversion

Convert trained checkpoints to HuggingFace format:

```bash
bash scripts/convert_checkpoint_to_hf.sh <base_model_dir> <model_home> <step>
```

Parameter description:
- `base_model_dir`: Qwen base model directory with expanded vocabulary (output from vocabulary expansion stage)
- `model_home`: Training output directory (i.e., `OUTPUT_DIR` in training script)
- `step`: Checkpoint step number to convert

**Example:**
```bash
# Assuming the vocabulary-expanded model is in ./qwen_extended
# Training output is in ./output
# Converting the checkpoint at step 4000
bash scripts/convert_checkpoint_to_hf.sh ./qwen_extended ./output 4000
```

Conversion process:
1. The script automatically locates the `{model_home}/step{step}/global_step{step}` directory
2. Reads the training checkpoint from that directory
3. Saves the converted HuggingFace format model to `{model_home}/step{step}/global_step{step}/converted/`

The converted model can be directly used for:
- Loading and inference with HuggingFace Transformers
- Subsequent SFT or other fine-tuning stages
- Model evaluation and deployment

### Model Testing

Test the converted HuggingFace model:

```bash
bash scripts/test_hf_model.sh <hf_model_dir>
```

Parameter description:
- `hf_model_dir`: Converted HuggingFace model directory

**Example:**
```bash
# Test the converted model at step 4000
bash scripts/test_hf_model.sh ./output/step4000/global_step4000/converted/
```

This script will verify:
- Whether model weights are loaded correctly
- Whether forward pass works normally
- Whether generation functionality is available

### Training Monitoring

Logs and outputs during training:

- **Standard output/error**: Saved in `$OUTPUT_DIR/stdout.log` and `$OUTPUT_DIR/stderr.log`
- **Training logs**: Contains loss values, learning rate, training steps, and other information
- **TensorBoard**: The model supports TensorBoard visualization. You can start TensorBoard with:
  ```bash
  tensorboard --logdir=$OUTPUT_DIR
  ```
- **Checkpoint**: Saved at configured step intervals (`--save_checkpoint_per_step`)

### Checkpoint Management

Checkpoints are saved periodically during training with the following directory structure:

```
output_dir/
├── step50/
│   └── global_step50/
│       ├── model/          # Model weights
│       ├── optimizer/      # Optimizer state
│       └── ...
├── step100/
│   └── global_step100/
│       └── ...
└── ...
```

**Resuming Training**:
To resume training from a checkpoint, add the following to the training script:
```bash
--resume_from $OUTPUT_DIR/step1000 \
--resume_from_tag global_step1000 \
--resume_training_state
```

## Notes


1. **Distributed Environment**:
   - Torchrun scripts use `torchrun` and standard PyTorch env vars (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`)
   - For multi-node torchrun, ensure a shared hostfile (e.g., `/etc/mpi/hostfile`) and a consistent `MASTER_ADDR:MASTER_PORT`
   - Legacy MPI scripts use `mpirun` with `OMPI_COMM_WORLD_*` environment variables

2. **Data Format**:
   - Ensure training data conforms to Parquet format specifications, refer to `../data/README.md`
   - It is recommended that each Parquet file contains approximately 1000 samples for efficient loading and shuffling
   - Data file lists are specified through JSON files, supporting both local paths and HDFS paths

3. **Vocabulary Expansion**:
   - Vocabulary expansion must be performed before training, using the expanded model as `base_model_dir`
   - The expanded model path needs to be specified in the `base_model_dir` field of the data configuration file
   - Ensure `itemic_id_range` is consistent with the configuration during vocabulary expansion

4. **Model Size**:
   - For smaller models like 0.6B / 1.7B / 4B, the `--use_tie_weights` parameter is required to align with Qwen3 model configuration
   - Different model sizes may require different learning rate and training step configurations

## Related Documentation

- [OpenOneRec Main README](../README.md): Project overview and complete workflow
- [Data Format Specification](../data/README.md): Training data format requirements and preprocessing methods
