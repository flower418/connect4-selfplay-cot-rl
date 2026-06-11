# connect4-selfplay-cot-rl

四子棋自博弈 CoT-RL 项目。当前第一阶段已经完成：用强规则 oracle 生成标签，用 DeepSeek 生成中文推理，再经过程序校验导出冷启动 SFT 数据。

## 当前状态

第一阶段 SFT 数据已经完成：

```text
data/seed/seed_positions_candidates_10k.jsonl      19078 个候选局面
data/seed/seed_positions_raw_10k.jsonl             10691 条 DeepSeek 原始样本
data/seed/seed_positions_raw_10k_errors.jsonl       2809 条失败记录
data/seed/seed_positions_verified_10k.jsonl        10691 条验证通过样本
data/train/seed_sft_10k.jsonl                      10691 条最终 SFT 样本
data/seed/seed_sft_10k_manifest.jsonl                  1 条数据生成 manifest
```

所有 `seed_sft_10k` 样本的 `move_quality` 都是 `best`。采集过程中的 batch 断点目录已经清理；最终训练、复现和审计所需文件保留在 `data/seed/` 和 `data/train/`。

## 数据管线架构

数据管线把“正确答案”和“自然语言推理”分开：

```text
候选局面生成
  -> 程序 oracle 标注正确落子
  -> DeepSeek 只负责生成中文分析和最终落子格式
  -> 解析模型回复
  -> 规则校验和 oracle 校验
  -> 导出 SFT JSONL
  -> 导出 verl Parquet
```

核心原则：

- 正确动作不由 DeepSeek 决定，而由本地 oracle 给出。
- DeepSeek 只生成训练用的中文分析文本。
- 样本必须通过解析、合法性、oracle 最优动作和基本忠实性校验后才进入 SFT。
- API 采集默认不重试，避免失败样本重复消耗 token。
- `verified` 数据保留 oracle 标签，后续做 RL reward 校验和数据审计时仍然有用。

## 关键模块

```text
connect4/env.py                  四子棋规则、合法动作、胜负判断、canonical id
connect4/oracle.py               深度受限 minimax oracle，用于 seed 过滤和 baseline
connect4/strong_oracle.py        exact solver，用于 frozen benchmark 真值
seed/generate_seed_positions.py  生成候选局面
seed/collect_seed_cot.py         调 DeepSeek 生成中文 CoT 样本
seed/build_seed_verified.py      raw -> verified
seed/build_large_seed_sft.py     10k 级 seed SFT 一键管线
training/build_sft.py            verified -> SFT JSONL
training/export_verl_sft.py      SFT JSONL -> verl parquet/jsonl
evaluation/                      frozen benchmark 和 rule baselines
scripts/run_verl_sft.sh          verl SFT 启动脚本
```

## 重新生成 SFT 数据

已有 `seed_sft_10k` 数据时不需要重新跑这一段。只有在要扩充数据时才执行。

```bash
export DEEPSEEK_MODEL=deepseek-v4-flash

python3 seed/build_large_seed_sft.py --step candidates --oracle-games 2500

python3 seed/build_large_seed_sft.py --step collect \
  --target-sft 10000 \
  --batch-size 1000 \
  --concurrency 100 \
  --max-tokens 3200 \
  --max-retries 1 \
  --resume
```

如果只想补少量数据，降低 batch 和并发：

```bash
python3 seed/build_large_seed_sft.py --step collect \
  --target-sft 12000 \
  --batch-size 500 \
  --concurrency 50 \
  --max-tokens 3200 \
  --max-retries 1 \
  --resume
```

## Benchmark

Benchmark 测的是固定局面的单步决策能力，不是完整对局胜率。它用 exact oracle 作为真值，主要指标是：

- `format_success_rate`: 是否能解析出最终落子
- `legal_rate`: 落子是否合法
- `oracle_best_acc`: 是否命中 exact oracle 最优动作
- `mean_value_regret`: 动作价值相对最优动作损失多少

构建 benchmark：

```bash
python3 evaluation/build_frozen_benchmark.py \
  --target-per-split 500 \
  --max-empty 14 \
  --max-candidates 1000000 \
  --progress-every 1000
```

跑 rule baseline：

```bash
python3 evaluation/evaluate_baselines.py \
  --input data/eval/frozen_benchmark.jsonl \
  --output data/metrics/baseline_results.jsonl
```

测 Hugging Face 模型：

```bash
python3 evaluation/evaluate_hf_model.py \
  --model /path/to/Qwen2.5-0.5B-Instruct-or-sft \
  --input data/eval/frozen_benchmark.jsonl \
  --output data/metrics/model_results.jsonl

python3 evaluation/summarize_results.py \
  --input data/metrics/model_results.jsonl \
  --actor-key model
```

## verl SFT

训练入口已经接到 `seed_sft_10k`：

```bash
export MODEL_PATH=/root/Qwen2.5-0.5B-Instruct
bash scripts/run_verl_sft.sh
```

脚本会先执行：

```bash
python3 training/export_verl_sft.py \
  --input data/train/seed_sft_10k.jsonl \
  --output data/train/seed_sft_verl.parquet
```

然后调用：

```bash
python3 -m verl.trainer.sft_trainer
```

### 可调参数

`scripts/run_verl_sft.sh` 暴露这些环境变量：

```bash
export MODEL_PATH=/root/Qwen2.5-0.5B-Instruct
export SFT_INPUT=data/train/seed_sft_10k.jsonl
export TRAIN_FILE=data/train/seed_sft_verl.parquet
export OUTPUT_DIR=outputs/seed_sft_qwen25_05b
export PROJECT_NAME=connect4-cot-rl
export EXPERIMENT_NAME=seed-sft-qwen25-05b
export TOTAL_EPOCHS=3
export TRAIN_BATCH_SIZE=8
export MICRO_BATCH_SIZE=1
export MAX_LENGTH=2048
export NNODES=1
export N_GPUS_PER_NODE=1
export SAVE_FREQ=-1
export TEST_FREQ=after_each_epoch
export LR=1e-5
export LOGGER='["console","wandb"]'

bash scripts/run_verl_sft.sh
```

### 可视化和日志

默认 `LOGGER='["console","wandb"]'`，训练指标会同时打印到控制台并上报 wandb。服务器需要提前执行 `wandb login` 或设置 `WANDB_API_KEY`。是否包含 `train loss`、`eval/validation loss` 取决于安装的 verl 版本和它的 SFT trainer 日志字段；当前脚本把 `data.val_files` 指到同一个 parquet，因此如果 trainer 支持 validation logging，会有验证侧指标，但这不是严格独立验证集。

显式设置 wandb：

```bash
export WANDB_API_KEY=...
export LOGGER='["console","wandb"]'
export PROJECT_NAME=connect4-cot-rl
export EXPERIMENT_NAME=seed-sft-qwen25-05b
bash scripts/run_verl_sft.sh
```

这样训练曲线会进入 wandb。常见可看指标包括训练 loss、学习率、step、吞吐，以及 verl 版本支持的验证 loss。

## 服务器环境

服务器上建议：

```bash
conda activate connect4-cot-rl
pip install -r requirements.txt
pip install -r requirements-verl.txt
```

当前脚本适配 `verl==0.8.x` 的入口 `verl.trainer.sft_trainer`。如果后续 verl 版本调整入口，只需要改 `scripts/run_verl_sft.sh` 里的模块名和 Hydra 字段，数据导出部分不需要改。

## 目录约定

保留：

```text
data/seed/seed_positions_candidates_10k.jsonl
data/seed/seed_positions_raw_10k.jsonl
data/seed/seed_positions_raw_10k_errors.jsonl
data/seed/seed_positions_verified_10k.jsonl
data/seed/seed_sft_10k_manifest.jsonl
data/train/seed_sft_10k.jsonl
```

可重新生成，不建议提交：

```text
data/train/seed_sft_verl.parquet
outputs/
wandb/
```

本地私密文件不提交：

```text
.env.local
models/
```
