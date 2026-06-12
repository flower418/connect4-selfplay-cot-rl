# connect4-selfplay-cot-rl

四子棋自博弈 CoT-RL 项目。目标是验证小模型在程序化规则、oracle 标注、过程监督和后续自博弈迭代下，是否能提升未见局面的单步决策能力和 CoT/action 一致性。

## 当前进度

已完成：

- 四子棋环境、合法动作、胜负判断、镜像 canonical id
- 深度受限 minimax oracle 和 late-game exact oracle
- Seed 数据管线：候选局面 -> DeepSeek 中文 CoT -> 解析 -> verifier -> SFT JSONL
- `seed_sft_10k` 数据集
- verl SFT 数据导出和训练脚本
- Seed SFT 训练
- frozen benchmark / rule baseline / HF 模型评估脚本

待做：

- Seed SFT checkpoint 的 frozen benchmark 评估结果归档
- self-play generation runner
- RS-SFT 代际数据过滤和再训练
- verl GRPO reward / rollout 接入

## 完整链路

```text
候选局面生成
  -> 本地 oracle 标注 best move
  -> DeepSeek 生成中文分析和最终落子格式
  -> 解析模型回复
  -> 规则校验、oracle 校验、faithfulness v1
  -> 导出 Seed SFT JSONL
  -> 导出 verl train/val parquet
  -> verl Seed SFT
  -> frozen benchmark 评估
  -> self-play 生成新数据
  -> verifier 过滤高质量样本
  -> RS-SFT 下一代
  -> 可选 GRPO
```

核心原则：

- 正确动作由本地 oracle 给出，不由 DeepSeek 决定。
- DeepSeek 只负责生成 cold-start 阶段的中文推理文本。
- 进入训练集的样本必须可解析、合法，并命中 oracle best/good move。
- `verified` 数据保留 oracle 标签，后续可复用到评估、RL reward 和数据审计。

## 数据状态

```text
data/seed/seed_positions_candidates_10k.jsonl      19078 candidates
data/seed/seed_positions_raw_10k.jsonl             10691 raw DeepSeek samples
data/seed/seed_positions_raw_10k_errors.jsonl       2809 failed calls
data/seed/seed_positions_verified_10k.jsonl        10691 verified samples
data/train/seed_sft_10k.jsonl                      10691 final SFT samples
data/seed/seed_sft_10k_manifest.jsonl                  1 manifest
```

## 关键模块

```text
connect4/env.py                  Connect4 rules and canonical ids
connect4/oracle.py               depth-limited minimax oracle
connect4/strong_oracle.py        exact late-game solver
seed/generate_seed_positions.py  seed candidate generation
seed/collect_seed_cot.py         DeepSeek CoT collection
seed/build_seed_verified.py      raw -> verified
seed/build_large_seed_sft.py     10k seed pipeline
verification/cleaner.py          verifier and faithfulness v1
training/build_sft.py            verified -> SFT JSONL
training/export_verl_sft.py      SFT JSONL -> verl parquet/jsonl
training/build_grpo.py           GRPO prompt export scaffold
evaluation/                      benchmark, baselines, model eval
scripts/run_verl_sft.sh          verl SFT entrypoint
```

## Seed 数据管线

已有 `seed_sft_10k` 时不需要重跑。扩充或复现时使用：

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

## verl Seed SFT

训练入口：

```bash
export MODEL_PATH=/root/Qwen2.5-0.5B-Instruct
bash scripts/run_verl_sft.sh
```

脚本会先导出 train/val parquet：

```bash
python3 training/export_verl_sft.py \
  --input data/train/seed_sft_10k.jsonl \
  --train-output data/train/seed_sft_verl.parquet \
  --val-output data/train/seed_sft_verl_val.parquet \
  --val-ratio 0.1
```

然后调用：

```bash
torchrun --standalone \
  -m verl.trainer.sft_trainer \
  data.train_files=data/train/seed_sft_verl.parquet \
  data.val_files=data/train/seed_sft_verl_val.parquet \
  data.messages_key=messages
```

常用参数：

```bash
export SFT_INPUT=data/train/seed_sft_10k.jsonl
export OUTPUT_DIR=outputs/seed_sft_qwen25_05b
export TOTAL_EPOCHS=3
export TRAIN_BATCH_SIZE=8
export MICRO_BATCH_SIZE=1
export MAX_LENGTH=4096
export TEST_FREQ=25
export SAVE_FREQ=-1
export MAX_CKPT_TO_KEEP=1
export LOGGER='["console","wandb"]'
```

## 评估

构建 frozen benchmark：

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

评估 base 或 SFT checkpoint：

```bash
python3 evaluation/evaluate_hf_model.py \
  --model /path/to/model-or-sft-checkpoint \
  --input data/eval/frozen_benchmark.jsonl \
  --output data/metrics/model_results.jsonl

python3 evaluation/summarize_results.py \
  --input data/metrics/model_results.jsonl \
  --actor-key model
```

核心指标：

- `format_success_rate`
- `legal_rate`
- `oracle_best_acc`
- `mean_value_regret`

## 环境

```bash
conda activate connect4-cot-rl
pip install -r requirements.txt
pip install -r requirements-verl.txt
```

默认使用 `sdpa` attention，不强依赖 `flash-attn`。当前脚本适配 `verl==0.8.x` 的 `verl.trainer.sft_trainer`。

## 仓库约定

保留并提交：

```text
data/seed/seed_positions_candidates_10k.jsonl
data/seed/seed_positions_raw_10k.jsonl
data/seed/seed_positions_raw_10k_errors.jsonl
data/seed/seed_positions_verified_10k.jsonl
data/seed/seed_sft_10k_manifest.jsonl
data/train/seed_sft_10k.jsonl
```

不提交：

```text
data/train/seed_sft_verl.parquet
data/train/seed_sft_verl_val.parquet
outputs/
wandb/
.env.local
models/
```
