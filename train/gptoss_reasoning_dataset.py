"""Dataset builders that preserve GPT-OSS Harmony reasoning structure."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


GENERATION_REGEX = re.compile(r"\{%-?\s+generation\s+-?%\}")


def _has_chat_template(tokenizer: Any) -> bool:
    """Return True when the tokenizer exposes a callable chat template."""
    return getattr(tokenizer, "chat_template", None) is not None and callable(
        getattr(tokenizer, "apply_chat_template", None)
    )


def _ensure_pad_token(tokenizer: Any) -> int:
    """Ensure the tokenizer has a usable pad token id."""
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if (
        getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    return int(tokenizer.pad_token_id)


def _pad_to_seq_length(
    values: list[int], pad_value: int, seq_length: int
) -> list[int]:
    """Pad one sequence to the requested fixed length."""
    missing = seq_length - len(values)
    if missing <= 0:
        return values
    return values + [pad_value] * missing


def _normalize_scalar(value: Any) -> str:
    """Normalize scalar content values to strings."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
    """Normalize one conversation while preserving assistant thinking."""
    if not isinstance(messages, list):
        raise TypeError(
            f"Expected messages list, got {type(messages).__name__}"
        )

    normalized: list[dict[str, Any]] = []
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            raise TypeError(
                "Expected each message to be a dict, got "
                f"{type(raw_message).__name__}"
            )

        message: dict[str, Any] = {
            "role": str(raw_message.get("role", "")),
            "content": _normalize_scalar(raw_message.get("content")),
        }
        if "thinking" in raw_message and raw_message["thinking"] is not None:
            message["thinking"] = _normalize_scalar(raw_message["thinking"])
        if (
            "tool_calls" in raw_message
            and raw_message["tool_calls"] is not None
        ):
            message["tool_calls"] = raw_message["tool_calls"]
        if "recipient" in raw_message and raw_message["recipient"] is not None:
            message["recipient"] = raw_message["recipient"]
        if "name" in raw_message and raw_message["name"] is not None:
            message["name"] = _normalize_scalar(raw_message["name"])
        normalized.append(message)

    return normalized


def _package_tokenized_example(
    tokenizer: Any,
    input_ids: list[int],
    assistant_masks: list[int],
    seq_length: int | None,
    truncation: str | bool,
    padding: str | bool,
) -> dict[str, list[int] | dict[str, int]]:
    """Convert token ids and assistant masks into SFT tensors."""
    labels = input_ids.copy()
    shifted_input_ids = input_ids[:-1]
    attention_mask = [1] * len(shifted_input_ids)
    labels = [
        label if bool(mask) else -100
        for label, mask in zip(labels, assistant_masks)
    ]
    labels = labels[1:]

    if (
        isinstance(seq_length, int)
        and padding not in [None, "do_not_pad", False]
    ):
        pad_token_id = int(tokenizer.pad_token_id)
        shifted_input_ids = _pad_to_seq_length(
            shifted_input_ids, pad_token_id, seq_length
        )
        labels = _pad_to_seq_length(labels, -100, seq_length)

    attention_mask += [0] * (len(labels) - len(attention_mask))
    return {
        "input_ids": shifted_input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "___PAD_TOKEN_IDS___": {
            "input_ids": int(tokenizer.pad_token_id),
            "labels": -100,
            "attention_mask": 0,
        },
    }


def _format_chat_example(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    seq_length: int | None,
    truncation: str | bool,
    padding: str | bool,
    answer_only_loss_mask: bool,
) -> dict[str, list[int] | dict[str, int]]:
    """Tokenize one GPT-OSS conversation and preserve assistant masks."""
    if not _has_chat_template(tokenizer):
        raise ValueError(
            "Tokenizer lacks a usable chat template. GPT-OSS formatting "
            "requires tokenizer.apply_chat_template(...)."
        )

    template_has_generation_kwd = bool(
        GENERATION_REGEX.search(str(tokenizer.chat_template))
    )
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=template_has_generation_kwd,
        padding=padding,
        truncation=truncation,
        max_length=seq_length,
    )
    input_ids = list(tokenized_chat["input_ids"])

    if template_has_generation_kwd:
        assistant_masks = list(tokenized_chat["assistant_masks"])
    elif answer_only_loss_mask:
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError(
                "Expected the last message to be an assistant message."
            )
        prompt_messages = messages[:-1]
        prompt_tokens = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=False,
            padding=padding,
            truncation=truncation,
            max_length=seq_length,
        )
        prompt_length = len(list(prompt_tokens.get("input_ids", [])))
        assistant_masks = [0] * prompt_length + [1] * (
            len(input_ids) - prompt_length
        )
    else:
        assistant_masks = [1] * len(input_ids)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if (
        eos_token_id is not None
        and input_ids
        and input_ids[-1] != eos_token_id
    ):
        input_ids.append(int(eos_token_id))
        assistant_masks.append(1)

    return _package_tokenized_example(
        tokenizer=tokenizer,
        input_ids=input_ids,
        assistant_masks=assistant_masks,
        seq_length=seq_length,
        truncation=truncation,
        padding=padding,
    )


def make_gptoss_multilingual_thinking_dataset(
    tokenizer: Any,
    path_or_dataset_id: str = "HuggingFaceH4/Multilingual-Thinking",
    split: str = "train",
    name: str | None = None,
    seq_length: int | None = 1024,
    padding: str | bool = "do_not_pad",
    truncation: str | bool = True,
    limit_dataset_samples: int | None = None,
    answer_only_loss_mask: bool = True,
    cache_dir: str | None = None,
) -> Dataset:
    """Return a HF dataset that preserves GPT-OSS reasoning traces."""
    _ensure_pad_token(tokenizer)
    raw_dataset = load_dataset(
        path_or_dataset_id,
        name=name,
        split=split,
        cache_dir=cache_dir,
    )

    if (
        limit_dataset_samples is not None
        and limit_dataset_samples > 0
        and len(raw_dataset) > limit_dataset_samples
    ):
        raw_dataset = raw_dataset.select(range(limit_dataset_samples))

    def _map_example(
        example: dict[str, Any]
    ) -> dict[str, list[int] | dict[str, int]]:
        messages = _normalize_messages(example["messages"])
        return _format_chat_example(
            tokenizer=tokenizer,
            messages=messages,
            seq_length=seq_length,
            truncation=truncation,
            padding=padding,
            answer_only_loss_mask=answer_only_loss_mask,
        )

    return raw_dataset.map(
        _map_example,
        remove_columns=list(raw_dataset.column_names),
        desc=f"Formatting {path_or_dataset_id} ({split}) for GPT-OSS",
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset inspection."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one GPT-OSS reasoning dataset example after "
            "tokenization/label construction."
        )
    )
    parser.add_argument(
        "--tokenizer-path",
        default="openai/gpt-oss-20b",
        help="Tokenizer path or model id used for chat formatting.",
    )
    parser.add_argument(
        "--dataset-id",
        default="HuggingFaceH4/Multilingual-Thinking",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--split",
        default="train[:1]",
        help="Dataset split expression for inspection.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Row index to inspect from the loaded split.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=1024,
        help="Sequence length passed into chat formatting.",
    )
    parser.add_argument(
        "--preview-tokens",
        type=int,
        default=350,
        help="How many tokens to decode for previews.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache dir for dataset loading.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for tokenizer loading.",
    )
    return parser.parse_args()


def inspect_dataset_sample(args: argparse.Namespace) -> None:
    """Print a readable inspection view for one formatted dataset sample."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )
    _ensure_pad_token(tokenizer)

    dataset_path = Path(args.dataset_id).expanduser()
    if dataset_path.is_file() and dataset_path.suffix == ".arrow":
        raw_dataset = Dataset.from_file(str(dataset_path))
    else:
        raw_dataset = load_dataset(
            args.dataset_id,
            split=args.split,
            cache_dir=args.cache_dir,
        )
    row = raw_dataset[args.row_index]
    messages = _normalize_messages(row["messages"])
    packaged = _format_chat_example(
        tokenizer=tokenizer,
        messages=messages,
        seq_length=args.seq_length,
        truncation=True,
        padding="do_not_pad",
        answer_only_loss_mask=True,
    )
    input_ids = packaged["input_ids"]
    labels = packaged["labels"]
    attention_mask = packaged["attention_mask"]
    visible_label_ids = [token for token in labels if token != -100]
    masked_count = sum(1 for token in labels if token == -100)

    print("num_messages", len(messages))
    print("roles", [message["role"] for message in messages])
    print(
        "has_thinking",
        [bool(message.get("thinking")) for message in messages],
    )
    print("input_len", len(input_ids))
    print("labels_len", len(labels))
    print("attention_len", len(attention_mask))
    print("masked_labels", masked_count)
    print("supervised_labels", len(visible_label_ids))
    print("rendered_input_preview_start")
    print(
        tokenizer.decode(
            input_ids[: args.preview_tokens],
            skip_special_tokens=False,
        )
    )
    print("rendered_input_preview_end")
    print("supervised_target_preview_start")
    print(
        tokenizer.decode(
            visible_label_ids[: args.preview_tokens],
            skip_special_tokens=False,
        )
    )
    print("supervised_target_preview_end")
    print("assistant_thinking_field_start")
    print(messages[-1].get("thinking"))
    print("assistant_thinking_field_end")
    print("assistant_content_field_start")
    print(messages[-1].get("content"))
    print("assistant_content_field_end")


def main() -> int:
    """CLI entrypoint for dataset inspection."""
    inspect_dataset_sample(_parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
