# 四子棋 Self-Play CoT-RL 项目方案

  ## Summary

  目标不是泛泛证明“涌现推理”，而是验证：Qwen2.5-0.5B 能否通过 self-play policy iteration + 过程监督 + 可选 RL/preference 优化，在未见四子棋局面上
  获得可量化的战略能力提升。

  主路线建议采用三阶段，而不是直接 RL：

  1. Seed SFT: 用强模型/搜索器生成高质量 cold-start 数据，让 0.5B 先学会格式、规则、基础战术。
  2. Iterative RS-SFT: 每代 self-play 生成数据，用程序 verifier 过滤高质量轨迹，再 SFT 下一代。
  3. Preference/RL Extension: 在 RS-SFT 稳定提升后，再加入 DPO 或 GRPO，对比是否带来额外收益。

  这样既有 RL sense，又不会把项目押在直接 RL 的不稳定性上。

  ## Core Design

  训练分支固定为四组，保证实验结论可归因：

  - Base: 原始 Qwen2.5-0.5B-Instruct，不训练。
  - Seed-SFT: 只用种子数据训练，验证 cold-start 效果。
  - SelfPlay-RS-SFT: 主实验路线，每代 self-play、过滤、SFT。
  - SelfPlay-RS-SFT+DPO/GRPO: 扩展路线，验证 RL/preference 是否超过纯 SFT policy iteration。

  四个假说对应指标：

  - H1: generation vs ELO、vs fixed baselines win rate、tactical puzzle accuracy。
  - H2: generation vs CoT faithfulness score。
  - H3: exact/symmetry/motif/depth holdout 上的泛化能力。
  - H4: faithfulness vs ELO correlation，加上 CoT intervention ablation 的因果证据。

  ## Data Pipeline

  整体数据目录建议：

  data/
    seed/
      seed_games_raw.jsonl
      seed_positions_verified.jsonl
      seed_sft_sharegpt.json
    selfplay/
      gen_000_raw_games.jsonl
      gen_001_raw_games.jsonl
    verified/
      gen_000_positions.jsonl
      gen_001_positions.jsonl
    train/
      sft_gen_000_sharegpt.json
      sft_gen_001_sharegpt.json
      dpo_gen_001.json
    eval/
      tactical_puzzles.jsonl
      holdout_exact.jsonl
      holdout_symmetry.jsonl
      holdout_motif.jsonl
      holdout_depth.jsonl
    metrics/
      arena_results.jsonl
      faithfulness_results.jsonl
      generation_summary.csv

  每条 self-play 原始样本保存完整信息：

  {
    "game_id": "gen_003_game_0127",
    "generation": 3,
    "move_index": 18,
    "player": "X",
    "board_before": [[...]],
    "legal_moves": [0, 2, 3, 5],
    "prompt": "...",
    "raw_response": "...",
    "parsed_cot": "...",
    "parsed_action": 3,
    "is_legal": true,
    "board_after": [[...]],
    "winner": "X",
    "terminal": false,
    "model_path": "models/gen_003",
    "decode_config": {
      "temperature": 0.7,
      "top_p": 0.9
    }
  }

  Verifier 负责追加程序化标签：

  {
    "position_id": "sha256(board+player)",
    "canonical_id": "sha256(symmetry_canonical_board+player)",
    "outcome": "win",
    "move_quality": "best|good|neutral|blunder|illegal",
    "tactical_tags": ["immediate_win", "must_block", "double_threat"],
    "oracle_best_moves": [3],
    "oracle_value_before": 0.42,
    "oracle_value_after": 0.91,
    "cot_action_consistent": true,
    "cot_mentions_immediate_win": true,
    "cot_mentions_must_block": false,
    "faithfulness_score": 0.83
  }

  SFT 数据只从 verified 数据生成：

  {
    "conversations": [
      {
        "from": "human",
        "value": "当前棋盘... 轮到 X。请先分析局面，再选择一个合法列。"
      },
      {
        "from": "gpt",
        "value": "分析：...\n最终落子列: 3"
      }
    ]
  }

  DPO 数据从同一局面构造 preferred/rejected：

  {
    "prompt": "当前棋盘...",
    "chosen": "分析正确...\n最终落子列: 3",
    "rejected": "分析错误...\n最终落子列: 5",
    "preference_reason": "chosen blocks opponent immediate win"
  }

  ## Implementation Changes

  需要实现 7 个模块：

  - connect4/env.py: 四子棋规则、合法动作、胜负检测、棋盘 canonicalization、左右镜像去重。
  - connect4/oracle.py: minimax/alpha-beta 搜索器，至少支持 depth 4/6，用于 best move、局面 value、战术标签。
  - generation/selfplay.py: 加载当前 generation 模型，生成 400-2000 局 self-play，保存 raw JSONL。
  - verification/cleaner.py: 解析动作、过滤非法输出、标注 move quality、计算 faithfulness、去除测试集重叠。
  - training/build_sft.py: 从 verified positions 生成 LLaMA-Factory ShareGPT 格式。
  - training/build_dpo.py: 从同局面多响应、胜负轨迹、oracle best/worst move 构造 preference pairs。
  - evaluation/arena.py: 固定评估池，输出 ELO、胜率、非法率、CoT 指标和 puzzle accuracy。

  主控循环：

  gen_0 = Seed-SFT model

  for generation in 1..N:
    1. self-play current model for K games
    2. verify and label every move
    3. remove eval leakage by exact board + mirror board canonical_id
    4. keep high-quality samples:
       - legal move
       - winner move or oracle-good move
       - no tactical blunder
       - CoT/action consistent
    5. build SFT dataset with mixture:
       - previous best data 30%
       - current gen high-quality data 60%
       - seed data replay 10%
    6. train next model
    7. run arena evaluation
    8. optionally build DPO/GRPO data after gen_2

  推荐第一版参数：

  - seed games: 50-100 局，由 DeepSeek/强模型 + oracle 校验生成。
  - self-play games: 每代 800 局起步，多卡服务器可提升到 2000-5000。
  - generations: 5-8 代。
  - training: LoRA/QLoRA 优先；全量微调作为扩展，不作为第一版默认。
  - decoding: self-play 用 temperature=0.7 保持探索，evaluation 用 temperature=0 保持稳定。

  ## Metrics And Figures

  必须产出的核心图：

  - Generation vs ELO: 每代模型 round-robin，含置信区间。
  - Generation vs Win Rate: 分别对 random、greedy、minimax-depth2、minimax-depth4。
  - Generation vs Illegal Move Rate: 验证规则掌握。
  - Generation vs Tactical Accuracy: immediate win、must block、double threat、trap avoidance。
  - Generation vs Faithfulness Score: 验证 H2。
  - Faithfulness vs ELO: 验证 H4 相关性。
  - CoT Intervention Drop: no-CoT、normal-CoT、shuffled-CoT、wrong-CoT 的棋力下降。
  - Holdout Accuracy: exact、mirror、motif、depth holdout，验证 H3。

  CoT faithfulness score 建议组合：

  faithfulness =
    0.25 * action_consistency
  + 0.25 * tactical_claim_correctness
  + 0.20 * threat_recognition
  + 0.20 * counterfactual_sensitivity
  + 0.10 * no_hallucinated_illegal_claims

  不要只依赖 LLM judge。LLM judge 可以作为辅助，但主指标必须来自程序 verifier。

  ## Test Plan

  基础正确性测试：

  - 四子棋落子、满盘、横/竖/双对角胜利检测。
  - 合法列解析和非法输出处理。
  - 镜像 canonicalization 是否稳定。
  - oracle 在人工构造局面中能找到 immediate win 和 must block。
  - 数据 cleaner 不允许 eval canonical_id 泄漏进 train。
  - ShareGPT/DPO 输出能被 LLaMA-Factory 配置读取。

  实验验收标准：

  - SelfPlay-RS-SFT 相比 Seed-SFT 在 fixed baselines 上稳定提升。
  - 至少 3 个连续 generation 的 ELO 或 tactical accuracy 不下降，或总体趋势显著上升。
  - holdout 集不低于 in-distribution 太多，否则不能声称泛化。
  - CoT intervention 中 shuffled/wrong CoT 明显降低表现，才支持“CoT 与动作有因果关系”。
  - DPO/GRPO 若无明显增益，也要报告为负结果；这反而能体现实验严谨性。

  ## Assumptions

  - 默认使用多卡服务器，但第一版仍建议 LoRA/QLoRA，避免全量微调成本过高。
  - 默认主路线是 Seed-SFT -> Iterative RS-SFT -> optional DPO/GRPO，不建议直接 RL 起步。
  - DeepSeek/强模型只用于 seed/cold-start，不参与最终 evaluation。
  - 所有测试集必须先冻结，再开始 self-play generation，避免后验挑题。
  - 项目结论表述为“受控 self-play 和过程监督提升了小模型的战术泛化与 CoT-action 一致性”，不要直接写成“证明小模型涌现真正推理”。