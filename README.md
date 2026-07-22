# abir-001-aerospace-llm

### abir-001: QLoRA Fine-Tuned Aerospace Engineering Assistant

[![Model on Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-abir--001-yellow)](https://huggingface.co/abirh19/abir-001)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abirh19/abir-001-aerospace-llm/blob/main/fine_tune_abir001.ipynb)

A parameter-efficient fine-tune of `Qwen/Qwen2.5-1.5B-Instruct` on a hand-built
aerospace/engineering Q&A dataset, using QLoRA (4-bit quantization + LoRA adapters)
via Hugging Face `transformers`, `peft`, and `trl`. Trained weights are published
at [`abirh19/abir-001`](https://huggingface.co/abirh19/abir-001) on the Hub.

## Motivation

Most public instruction-tuning tutorials fine-tune on generic, off-the-shelf
datasets (Guanaco, Alpaca, etc.) that don't demonstrate anything about the
person doing the fine-tuning. This project instead uses a dataset built from
my own aerospace engineering coursework (mechanics of materials, orbital
mechanics, aerodynamics, propulsion, spacecraft structures, and applied
numerical methods) plus write-ups of two of my own engineering projects
(a Warman Design and Build Competition crane subsystem, and a 1U CubeSat
structural frame). The goal is a small model that can discuss these topics
and my own project decisions coherently -- and a project I can actually
explain end-to-end, including its limitations.

## Dataset

- **145 Q&A pairs**, hand-written (not scraped or copied from any textbook
  or third-party source), covering:
  - Mechanics of materials (20)
  - Orbital mechanics (15)
  - Aerodynamics (15)
  - Propulsion (15)
  - Spacecraft structures, general + CubeSat-specific (15)
  - Numerical methods applied to engineering problems (15)
  - Materials science for aerospace (15)
  - Design process / systems engineering (10)
  - My Warman crane subsystem project, including its actual failure and
    root-cause analysis (15)
  - My CubeSat structural frame project (10)
- Stored as conversational `messages` (system/user/assistant turns) rather
  than a pre-formatted chat-template string, so `SFTTrainer` applies the
  tokenizer's own chat template and can train with loss computed only on
  the assistant's tokens (see Method below).
- Split 88/12 into `train.jsonl` (128 examples) and `val.jsonl` (17 examples),
  held out and never trained on, used for both loss tracking and qualitative
  before/after comparison.

**Honest limitation**: 145 examples is small. This is enough to demonstrate
the full fine-tuning pipeline and produce a noticeable shift in how the model
responds to in-domain questions, but it is not enough data to expect robust,
general aerospace-domain competence comparable to a much larger fine-tune.
Treat this as a proof-of-concept / methodology demonstration, not a
production-quality domain model -- and say so if asked, rather than
overclaiming what a 145-example fine-tune can do.

## Method

- **Base model**: `Qwen/Qwen2.5-1.5B-Instruct` (small enough to iterate
  quickly on a single T4 GPU)
- **Technique**: QLoRA -- 4-bit NF4 quantization of the base model, with
  LoRA adapters (r=32, alpha=32, dropout=0.1) trained on top
- **Assistant-only loss**: loss is computed only on the assistant's answer
  tokens, not the system/user prompt tokens (`assistant_only_loss=True` in
  `SFTConfig`, using the conversational `messages` dataset format). Training
  on the full sequence (including memorizing the fixed system prompt and
  question wording) wastes gradient signal that should go toward the actual
  answer content -- this was a real issue in an earlier version of this
  project and is worth calling out if asked, since it's a common mistake in
  QLoRA tutorials that use a flat pre-formatted text field.
- **Eval-based checkpoint selection**: trains for up to 10 epochs but keeps
  the checkpoint with the *best held-out validation loss*
  (`load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`), with
  early stopping if validation loss stops improving for 4 consecutive eval
  rounds. This directly targets the overfitting/hallucination seen in an
  earlier run, where the final-epoch checkpoint had picked up stylistic
  patterns faster than factual reliability.
- **Learning rate**: 1e-4 (lowered from an initial 2e-4, which was likely
  too aggressive for a dataset this small and contributed to the earlier
  hallucination issue), cosine schedule, ~3% warmup, gradient checkpointing
  enabled
- **Framework versions used for this run**: transformers 5.14.1, trl 1.9.0,
  peft 0.19.1, accelerate 1.14.0 (see `requirements.txt`; this stack changes
  fast, see Reproducibility note below)

## Evaluation

Two things are checked, deliberately kept modest and honest for a dataset
this size:

1. **Held-out eval loss**, tracked during training via `eval_dataset` /
   `eval_strategy="steps"` in the notebook -- watch this in the TensorBoard
   cell rather than trusting a single final training-loss number.
2. **Qualitative base-vs-fine-tuned comparison** on held-out validation
   questions (never seen during training) -- the notebook loads both the
   original base model and the fine-tuned model and prints both answers
   side by side against the reference answer, so you can actually read
   whether fine-tuning changed the response in a meaningful way, rather
   than relying on a loss number alone.

This is not a rigorous benchmark (no automatic scoring, no large held-out
test set) -- it's an honest, inspectable check appropriate to the dataset's
scale. If you want to make stronger claims, expanding the dataset and adding
an actual scored eval set (e.g. multiple-choice or exact-match questions)
would be the next step.

## Reproducibility note

Hugging Face's `transformers`/`trl`/`peft` stack has broken API changes
frequently across versions (e.g. `SFTConfig` field renames, `TrainingArguments`
argument changes, `tokenizer` -> `processing_class`). The notebook has already
been adjusted to work with the versions listed above; if you run this later
and hit a `TypeError` about an unexpected keyword argument, check the
installed version (`import transformers; transformers.__version__`, etc.)
against current docs and adjust field names accordingly -- this is a known,
recurring characteristic of this ecosystem, not a sign anything is
fundamentally wrong with the approach.

## Files

- `fine_tune_abir001.ipynb` -- the training notebook (Colab-ready, T4 GPU)
- `train.jsonl` / `val.jsonl` -- the dataset splits
- `README.md` -- this file
- `requirements.txt` -- package versions this was last run and verified against

## Using the trained model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("abirh19/abir-001")
tokenizer = AutoTokenizer.from_pretrained("abirh19/abir-001")

messages = [
    {"role": "system", "content": "You are abir-001, an assistant specialized in aerospace engineering fundamentals and design."},
    {"role": "user", "content": "Why do rocket nozzles have a converging-diverging shape?"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
output = model.generate(inputs, max_new_tokens=200)
print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))
```

## Possible next steps

- Expand the dataset further (aim for 300+ examples) for a more convincing
  demonstration of domain shift
- Add a small scored evaluation set (e.g. 20-30 held-out multiple-choice or
  short-answer questions with known correct answers) for a quantitative
  accuracy number, not just qualitative comparison
- Try a second base model (e.g. Phi-3.5-mini) and compare results
- Write up the base-vs-fine-tuned comparison output (once you have a clean
  run) as a short "Results" section here, including at least one honest
  example of a remaining failure mode -- this is more credible than only
  showing successes
