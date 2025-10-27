# Three-Stage Training Pipeline

The training process consists of three stages:

- **Stage 1**: Initialize token embeddings for the newly added modality
- **Stage 2**: Train the model to learn the speech-to-gesture task
- **Stage 3**: Enable the model to generate gestures following motion example prompts

## Important Notes

### Checkpoint Configuration

Before training Stage 2 and Stage 3, you need to update the checkpoint path in the configuration files:

- Modify `checkpointer.checkpoint_dir` in `recipes/configs/qwen2_5_meco/stage2.yaml` to point to your Stage 1 checkpoint (usually located in output/qwen2.5_0.5b-stage1/epoch_xx)
- Modify `checkpointer.checkpoint_dir` in `recipes/configs/qwen2_5_meco/stage3.yaml` to point to your Stage 2 checkpoint (usually located in output/qwen2.5_0.5b-stage2/epoch_xx)

### Memory Management

If you encounter out-of-memory (OOM) errors during training, please change your yaml config:

- Reduce `batch_size`
- Increase `gradient_accumulation_steps`

**Recommended setting**: Keep the product `num_gpus × batch_size × gradient_accumulation_steps ≈ 256` for optimal results.

## Training Commands

### Stage 1: Initialize Token Embeddings
```bash
cd torchtune
CUDA_VISIBLE_DEVICES=0,1 tune run --master_port 11111 --nproc_per_node 2 \
  full_finetune_distributed --config recipes/configs/qwen2_5_meco/stage1.yaml
```

### Stage 2: Speech-to-Gesture Learning
```bash
cd torchtune
CUDA_VISIBLE_DEVICES=0,1 tune run --master_port 11111 --nproc_per_node 2 \
  full_finetune_distributed --config recipes/configs/qwen2_5_meco/stage2.yaml
```

### Stage 3: Motion-Prompted Generation
```bash
cd torchtune
CUDA_VISIBLE_DEVICES=0,1 tune run --master_port 11111 --nproc_per_node 2 \
  full_finetune_distributed --config recipes/configs/qwen2_5_meco/stage3.yaml
```

## Customization

### Multi-GPU Training

To use a different number of GPUs, adjust the `CUDA_VISIBLE_DEVICES` and `nproc_per_node` parameters:
```bash
# Example: Using 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 tune run --master_port 11111 --nproc_per_node 4 \
  full_finetune_distributed --config recipes/configs/qwen2_5_meco/stage1.yaml
```

### Port Configuration

If port 11111 is occupied, change the `--master_port` parameter to an available port.