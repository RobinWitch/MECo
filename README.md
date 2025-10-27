# MECO: Motion-example-controlled Co-speech Gesture Generation Leveraging Large Language Models

## Environment Setup

Create a conda environment with Python 3.13:
```bash
conda create -n meco python=3.13
conda activate meco
```

Install torchtune as a submodule:
```bash
git submodule add https://github.com/RobinWitch/torchtune.git torchtune
cd torchtune
pip install -e .
cd ../
```

Install required dependencies:
```bash
pip install -r requirement_list.txt
```

## Model Download

### Download Base Model
```bash
mkdir ckpt
cd ckpt
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
git clone https://huggingface.co/robinwitch/MECo_BEAT2_2_qwen2.5_0.5b_stage1
git clone https://huggingface.co/robinwitch/MECo_BEAT2_2_qwen2.5_0.5b_stage2
git clone https://huggingface.co/robinwitch/MECo_BEAT2_2_qwen2.5_0.5b_stage3
cd ../
```

### Download Dataset

Download BEAT2 dataset:
```bash
huggingface-cli download H-Liu1997/BEAT2 \
  --repo-type dataset \
  --local-dir ./dataset/BEAT2 \
  --include "beat_english_v2.0.0/*"
```

Download additional dataset from Google Drive:
```bash
gdown https://drive.google.com/drive/folders/1MCks7CMNBtAzU2XihYezNmiGT_6pWex8?usp=drive_link -O ./dataset/hub --folder
```

### Download mHuBERT Model
```bash
mkdir ckpt/mhubert_base_1000
wget -P ckpt/mhubert_base_1000 https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt
wget -P ckpt/mhubert_base_1000 https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin
```

Clone fairseq repository:
```bash
mkdir refer
git clone https://github.com/facebookresearch/fairseq.git refer/fairseq
```

### Python 3.13 Compatibility Fix

When using fairseq with Python 3.13, you may encounter a `ValueError` related to mutable defaults. To fix this issue:

1. Locate your Python dataclasses file: `/path/to/your/env/lib/python3.13/dataclasses.py`
2. Comment out lines 859-861:
```python
# if f._field_type is _FIELD and f.default.__class__.__hash__ is None:
#     raise ValueError(f'mutable default {type(f.default)} for field '
#                      f'{f.name} is not allowed: use default_factory')
```

## Training

Follow these steps to train the model:

### Step 1: Calculate Mean and Standard Deviation
```bash
python scripts_vqvae/0_get_mean_std.py
```

### Step 2: Train RVQVAE
```bash
python scripts_vqvae/1_train_vqvae.py
```

### Step 3: Build LLM with Expanded Vocabulary
```bash
python scripts/0_build_qwen2.5_0.5b.py
```

### Step 4: Disable Tied Embedding
```bash
python scripts/1_detied_embedding.py
```

### Step 5: Build Training Dataset
```bash
python scripts/2_build_dataset_mhubert1000.py
```

### Step 6: Three-Stage LLM Training

Execute the three training commands in [scripts/3_train_llm.md](scripts/3_train_llm.md) sequentially to complete the three-stage training of the LLM.

## Evaluation

### Audio-Only Speech-to-Gesture Evaluation
```bash
python scripts/4_eval_gpt_decode_mhubert1000.py
```

### Speech-to-Gesture Evaluation with Motion Prompts
```bash
python scripts/5_eval_gpt_decode_mhubert1000_withprompt.py
```

## Performance Benchmarks

Due to the inherent diversity of GPT model inference:

- **Stage 2 (Audio-only generation)**: FID scores typically range from **0.32 to 0.36**
- **Stage 3 (With motion example control)**: FID scores typically range from **0.27 to 0.29**

---

## Citation

If you find this work useful, please consider citing:
```bibtex
@inproceedings{chen2025meco,
  author = {Bohong Chen and Yumeng Li and Youyi Zheng and Yao-Xiang Ding and Kun Zhou},
  title = {Motion-example-controlled Co-speech Gesture Generation Leveraging Large Language Models},
  year = {2025},
  isbn = {9798400715402},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3721238.3730611},
  doi = {10.1145/3721238.3730611},
  booktitle = {Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  series = {SIGGRAPH Conference Papers '25}
}
```

## Acknowledgments

Thanks to [EMAGE](https://github.com/PantoMatrix/PantoMatrix), [torchtune](https://github.com/meta-pytorch/torchtune), [ichigo](https://github.com/menloresearch/ichigo),  [T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [MoMask](https://github.com/EricGuo5513/momask-codes), [SynTalker](https://github.com/RobinWitch/SynTalker), our code is partially borrowing from them. Please check these useful repos.