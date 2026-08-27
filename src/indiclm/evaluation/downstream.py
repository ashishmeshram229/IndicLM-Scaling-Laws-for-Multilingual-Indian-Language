"""Downstream task evaluation: zero-shot sentiment classification.

Perplexity (`indiclm.evaluation.perplexity`) measures how well a model
predicts held-out text, but it can't answer the actual research question
this project asks: does mixture ratio, tokenizer choice, or data quality
change *usable model quality*, not just loss? This module adds one real
downstream task so mixture/tokenizer/ablation experiments can be compared
on something other than perplexity.

Method: label scoring via length-normalized log-likelihood, the standard
zero-shot classification approach for a causal LM with no classification
head (used by GPT-2/GPT-3-style zero-shot evaluations). For each example,
the prompt is `"{text}\\nSentiment:"` and the model scores each candidate
label as a continuation; the label with the higher *average per-token*
log-probability wins (length-normalized so "positive" and "negative"
having different tokenized lengths under a given tokenizer doesn't bias
the comparison).

Label words are fixed English strings ("positive"/"negative") across all
languages rather than per-language translations. This is a deliberate
simplification, not an oversight: at this project's vocabulary scale
(1-2K token BPE/Unigram tokenizers trained on a few hundred documents),
translated label words would frequently fall back to byte/UNK
fragments, which would make the scoring artifact-driven rather than
task-driven. Using one fixed anchor pair isolates the thing being
measured (does the model represent sentiment in context) from tokenizer
coverage of a specific label string.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

from indiclm.models.config import ModelConfig
from indiclm.models.transformer import DecoderOnlyTransformer
from indiclm.training.checkpoint import load_checkpoint

LABELS = ("positive", "negative")
PROMPT_TEMPLATE = "{text}\nSentiment:"


@dataclass
class SentimentExample:
    text: str
    label: str


@dataclass
class LanguageSentimentResult:
    language: str
    n_examples: int
    n_correct: int
    accuracy: float
    # per-example predictions, for error analysis / the report's "failure cases" section
    predictions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "n_examples": self.n_examples,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "predictions": self.predictions,
        }


@dataclass
class DownstreamReport:
    task: str
    checkpoint: str
    tokenizer: str
    overall_accuracy: float
    macro_avg_accuracy: float
    n_examples: int
    per_language: dict[str, LanguageSentimentResult]
    label_words: tuple[str, ...]
    chance_accuracy: float

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "checkpoint": self.checkpoint,
            "tokenizer": self.tokenizer,
            "overall_accuracy": self.overall_accuracy,
            "macro_avg_accuracy": self.macro_avg_accuracy,
            "n_examples": self.n_examples,
            "label_words": list(self.label_words),
            "chance_accuracy": self.chance_accuracy,
            "per_language": {k: v.to_dict() for k, v in self.per_language.items()},
        }


def load_sentiment_examples(eval_dir: Path) -> dict[str, list[SentimentExample]]:
    """Loads `{language}.jsonl` files from `eval_dir` (see
    `data/eval/sentiment/README.md` for provenance and format)."""
    by_language: dict[str, list[SentimentExample]] = {}
    for path in sorted(Path(eval_dir).glob("*.jsonl")):
        examples = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(SentimentExample(text=row["text"], label=row["label"]))
        if examples:
            by_language[path.stem] = examples
    return by_language


@torch.no_grad()
def _label_logprob(
    model: DecoderOnlyTransformer,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    label: str,
    device: torch.device,
    max_len: int,
) -> float:
    """Length-normalized mean log-probability of `label`'s tokens,
    teacher-forced on a single [prompt_ids + label_ids] forward pass."""
    prompt_ids = tokenizer.encode(prompt, out_type=int)
    label_ids = tokenizer.encode(" " + label, out_type=int)
    if not label_ids:
        return float("-inf")

    ids = (prompt_ids + label_ids)[-max_len:]
    n_label = min(len(label_ids), len(ids))  # in case truncation ate into the label
    if n_label == 0:
        return float("-inf")

    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x, targets=None)  # (1, T, V)
    log_probs = F.log_softmax(logits[0], dim=-1)  # (T, V)

    # Position i's logits predict token i+1, so the label's own tokens are
    # predicted by the logits one position before each label token.
    label_start = len(ids) - n_label
    total = 0.0
    for offset in range(n_label):
        target_pos = label_start + offset
        pred_pos = target_pos - 1
        if pred_pos < 0:
            continue
        target_token = ids[target_pos]
        total += log_probs[pred_pos, target_token].item()
    return total / n_label


def _evaluate_language(
    model: DecoderOnlyTransformer,
    tokenizer: spm.SentencePieceProcessor,
    examples: list[SentimentExample],
    language: str,
    device: torch.device,
    max_len: int,
) -> LanguageSentimentResult:
    predictions = []
    n_correct = 0
    for ex in examples:
        prompt = PROMPT_TEMPLATE.format(text=ex.text)
        scores = {
            label: _label_logprob(model, tokenizer, prompt, label, device, max_len)
            for label in LABELS
        }
        predicted = max(scores, key=lambda label: scores[label])
        correct = predicted == ex.label
        n_correct += int(correct)
        predictions.append(
            {"text": ex.text, "gold": ex.label, "predicted": predicted, "correct": correct, "scores": scores}
        )
    n = len(examples)
    return LanguageSentimentResult(
        language=language,
        n_examples=n,
        n_correct=n_correct,
        accuracy=round(n_correct / n, 4) if n else 0.0,
        predictions=predictions,
    )


def evaluate_downstream_sentiment(
    checkpoint_path: Path,
    tokenizer_path: Path,
    eval_dir: Path = Path("data/eval/sentiment"),
    device: str = "cpu",
) -> DownstreamReport:
    by_language = load_sentiment_examples(eval_dir)
    if not by_language:
        raise ValueError(
            f"No sentiment eval examples found under {eval_dir}. "
            "Expected one {{language}}.jsonl file per language."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig(**checkpoint["config"]["model_config"])
    model = DecoderOnlyTransformer(model_config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    torch_device = torch.device(device)
    max_len = model_config.max_seq_len

    per_language = {
        lang: _evaluate_language(model, tokenizer, examples, lang, torch_device, max_len)
        for lang, examples in sorted(by_language.items())
    }

    n_total = sum(r.n_examples for r in per_language.values())
    n_correct_total = sum(r.n_correct for r in per_language.values())
    macro_avg = sum(r.accuracy for r in per_language.values()) / len(per_language)

    return DownstreamReport(
        task="sentiment_classification",
        checkpoint=str(checkpoint_path),
        tokenizer=str(tokenizer_path),
        overall_accuracy=round(n_correct_total / n_total, 4) if n_total else 0.0,
        macro_avg_accuracy=round(macro_avg, 4),
        n_examples=n_total,
        per_language=per_language,
        label_words=LABELS,
        chance_accuracy=round(1.0 / len(LABELS), 4),
    )
