# TRL Training Methods Overview

TRL (Transformer Reinforcement Learning) provides multiple training methods for fine-tuning and aligning language models. This reference provides a brief overview of each method.

## Supervised Fine-Tuning (SFT)

**What it is:** Standard instruction tuning with supervised learning on demonstration data.

**When to use:**
- Initial fine-tuning of base models on task-specific data
- Teaching new capabilities or domains
- Most common starting point for fine-tuning

**Dataset format:** Conversational format with "messages" field, OR text field, OR prompt/completion pairs

**Example:**
```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="my-model",
        push_to_hub=True,
        hub_model_id="username/my-model",
        eval_strategy="no",  # Disable eval for simple example
        # max_length=1024 is the default - only set if you need different length
    )
)
trainer.train()
```

**Note:** For production training with evaluation monitoring, see `scripts/train_sft_example.py`

**Documentation:** `hf_doc_fetch("https://huggingface.co/docs/trl/sft_trainer")`

## Direct Preference Optimization (DPO)

**What it is:** Alignment method that trains directly on preference pairs (chosen vs rejected responses) without requiring a reward model.

**When to use:**
- Aligning models to human preferences
- Improving response quality after SFT
- Have paired preference data (chosen/rejected responses)

**Dataset format:** Preference pairs with "chosen" and "rejected" fields

**Example:**
```python
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",  # Use instruct model
    train_dataset=dataset,
    args=DPOConfig(
        output_dir="dpo-model",
        beta=0.1,  # KL penalty coefficient
        eval_strategy="no",  # Disable eval for simple example
        # max_length=1024 is the default - only set if you need different length
    )
)
trainer.train()
```

**Note:** For production training with evaluation monitoring, see `scripts/train_dpo_example.py`

**Documentation:** `hf_doc_fetch("https://huggingface.co/docs/trl/dpo_trainer")`

## Group Relative Policy Optimization (GRPO)

**What it is:** Online RL method that optimizes relative to group performance, useful for tasks with verifiable rewards.

**When to use:**
- Tasks with automatic reward signals (code execution, math verification)
- Online learning scenarios
- When DPO offline data is insufficient

**Dataset format:** Prompt-only format (model generates responses, reward computed online)

**Example:**
```python
# Use TRL maintained script
hf_jobs("uv", {
    "script": "https://raw.githubusercontent.com/huggingface/trl/main/examples/scripts/grpo.py",
    "script_args": [
        "--model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct",
        "--dataset_name", "trl-lib/math_shepherd",
        "--output_dir", "grpo-model"
    ],
    "flavor": "a10g-large",
    "timeout": "4h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**Documentation:** `hf_doc_fetch("https://huggingface.co/docs/trl/grpo_trainer")`

## Reward Modeling

**What it is:** Train a reward model to score responses, used as a component in RLHF pipelines.

**When to use:**
- Building RLHF pipeline
- Need automatic quality scoring
- Creating reward signals for PPO training

**Dataset format:** Preference pairs with "chosen" and "rejected" responses

**Documentation:** `hf_doc_fetch("https://huggingface.co/docs/trl/reward_trainer")`

## Proximal Policy Optimization (PPO)

**What it is:** Classic RLHF method using policy gradient optimization with a learned reward model. The gold standard for RL-based alignment, used in InstructGPT and the original GPT-3 RLHF work.

**When to use:**
- Full RLHF pipeline with a trained reward model
- When you have a reward model scoring response quality
- Tasks where reward can be computed from output (summarization, helpfulness, safety)
- Most data-efficient when reward model is accurate

**Requirements:**
- A trained reward model (from `RewardModeling` or external)
- Reference model for KL divergence penalty
- Reward function that scores responses

**Dataset format:** Prompts only — the model generates responses, reward model scores them

**Minimal example:**
```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

model = AutoModelForCausalLMWithValueHead.from_pretrained("Qwen/Qwen2.5-0.5B")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

trainer = PPOTrainer(
    config=PPOConfig(
        output_dir="ppo-model",
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        total_episodes=10000,
    ),
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    train_dataset=dataset,  # prompts only
    reward_function=your_reward_model,
)
trainer.train()
```

**Via HF Jobs script:**
```python
hf_jobs("uv", {
    "script": "https://raw.githubusercontent.com/huggingface/trl/main/examples/scripts/ppo/ppo.py",
    "script_args": [
        "--model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct",
        "--dataset_name", "trl-internal-testing/descriptiveness-sentiment-trl-style",
        "--output_dir", "ppo-model",
        "--reward_model_path", "your/reward-model",
        "--sft_model_path", "Qwen/Qwen2.5-0.5B",
    ],
    "flavor": "a10g-large",
    "timeout": "4h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

**Full RLHF pipeline with PPO:**
1. **SFT** — Fine-tune base model on task data
2. **Reward Modeling** — Train reward model on preference pairs
3. **PPO** — Fine-tune SFT model using reward model signal
4. **Optional: GGUF conversion** — Deploy locally

**Documentation:** `hf_doc_fetch("https://huggingface.co/docs/trl/ppo_trainer")`

**References:**
- [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) (PPO original)
- [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2204.05862)
- [The N Implementation Details of RLHF with PPO](https://arxiv.org/abs/2404.17990)

## Method Selection Guide

| Method | Complexity | Data Required | Use Case |
|--------|-----------|---------------|----------|
| **SFT** | Low | Demonstrations | Initial fine-tuning |
| **DPO** | Medium | Paired preferences | Post-SFT alignment |
| **GRPO** | Medium | Prompts + reward fn | Online RL with automatic rewards |
| **PPO** | High | Prompts + reward model | Full RLHF with reward model |
| **Reward** | Medium | Paired preferences | Building RLHF pipeline |

## Recommended Pipeline

**For most use cases:**
1. **Start with SFT** - Fine-tune base model on task data
2. **Follow with DPO** - Align to preferences using paired data
3. **Optional: GGUF conversion** - Deploy for local inference

**For full RLHF with PPO:**
1. **Start with SFT** - Fine-tune base model on task data
2. **Train reward model** - On preference data
3. **PPO** - Fine-tune SFT model using reward model signal
4. **Optional: GGUF conversion** - Deploy locally

## Dataset Format Reference

For complete dataset format specifications, use:
```python
hf_doc_fetch("https://huggingface.co/docs/trl/dataset_formats")
```

Or validate your dataset:
```bash
uv run https://huggingface.co/datasets/mcp-tools/skills/raw/main/dataset_inspector.py \
  --dataset your/dataset --split train
```

## See Also

- `references/training_patterns.md` - Common training patterns and examples
- `scripts/train_sft_example.py` - Complete SFT template
- `scripts/train_dpo_example.py` - Complete DPO template
- [Dataset Inspector](https://huggingface.co/datasets/mcp-tools/skills/raw/main/dataset_inspector.py) - Dataset format validation tool
