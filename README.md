# hf-skills

> Hugging Face Skills — ported for Hermes Agent

This repo mirrors the [huggingface/skills](https://github.com/huggingface/skills) library, adapted for [Hermes Agent](https://github.com/chaichungsang/HermesSweeper).

## Skills Included

| Skill | Description |
|-------|-------------|
| `hf-cli` | HF Hub CLI — download models/datasets, upload files, manage repos |
| `huggingface-best` | Find best model for any task via leaderboards & benchmarks |
| `huggingface-community-evals` | Run local evals with inspect-ai / lighteval |
| `huggingface-datasets` | Explore/query HF datasets via REST API |
| `huggingface-gradio` | Build Gradio web UIs and demos |
| `huggingface-llm-trainer` | Fine-tune LLMs (SFT/DPO/GRPO) on HF Jobs infrastructure |
| `huggingface-local-models` | Run GGUF models locally with llama.cpp |
| `huggingface-paper-publisher` | Publish papers to HF Hub, link to models/datasets |
| `huggingface-papers` | Look up HF paper pages and arXiv metadata |
| `huggingface-tool-builder` | Build reusable HF API scripts |
| `huggingface-trackio` | Track ML training experiments with Trackio |
| `huggingface-vision-trainer` | Train object detection & image classification models |
| `transformers-js` | Run ML models in JavaScript/TypeScript |

## Blog Post

The launch announcement: [blog_hf-skills-training.md](./blog_hf-skills-training.md)  
Original HTML: [blog_hf-skills-training.html](./blog_hf-skills-training.html)

## Repository Structure

```
hf-skills/
├── hf-cli/                     # Individual skill folders
├── huggingface-best/
├── huggingface-community-evals/
├── huggingface-datasets/
├── huggingface-gradio/
├── huggingface-llm-trainer/    # ← main skill from the blog post
├── huggingface-local-models/
├── huggingface-paper-publisher/
├── huggingface-papers/
├── huggingface-tool-builder/
├── huggingface-trackio/
├── huggingface-vision-trainer/
├── transformers-js/
├── scripts/                    # HF-provided helper scripts
├── HUGGINGFACE_SKILLS_README.md
├── blog_hf-skills-training.md
└── blog_hf-skills-training.html
```

## Credits

All skills are from the official [huggingface/skills](https://github.com/huggingface/skills) repository, created by the Hugging Face team. This repo is just a local mirror adapted for personal use with Hermes Agent.

## License

MIT — see individual skill licenses and original Hugging Face repo for details.
