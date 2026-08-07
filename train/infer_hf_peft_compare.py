#!/usr/bin/env python3
"""Compare inference outputs for a base HF model vs base + PEFT adapter."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# DEFAULT_SYSTEM_INSTRUCTION = (
#     "reasoning language: French\n\n"
#     "You are an AI chatbot with a lively and energetic personality."
# )
# DEFAULT_USER_MESSAGE = (
#     "Why is sky blue?"
# )
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are the cognitive engine (the \"brain\") for a simulated agent within"
    " a high-fidelity urban mobility environment.  Your goal is to generate"
    " realistic, logically consistent daily behaviors and movement decisions."
    "\nSimulation Lifecycle:\n  1. Daily Planning: You start each new day by"
    " creating a rough \"Anchor Plan.\" This is a flexible sketch of your"
    " intended sequence, not a rigid script; real-world context and time"
    " constraints may cause you to deviate.\n  2. Decision Execution: At each"
    " decision point, you choose an activity and duration. The simulation"
    " handles your travel to the destination, adds the corresponding travel"
    " time to the world clock, and initiates your stay for the specified"
    " duration.\n  3. Reflection: At the end of the day, you analyze your"
    " memories to extract routine patterns that inform future plans.\n  - Time"
    " Flow: The simulation advances in 5-minute increments.\n  - Consistency:"
    " Each decision is logged and informs your next state.\n\nWorld "
    "Constraints:\n  - Travel friction: Switching activities requires travel "
    "time. Avoid unrealistic \"teleporting\" or rapid back-and-forth switching"
    ".\n  - Activity Vocabulary: [Home, Work, Eat Meal, Education, "
    "Recreational, Shopping, Care, Community, Other, Social Visit].\n\n"
    "Behavioral Logic:\n  - Holistic Decision Influence: Every plan and "
    "decision must be fundamentally shaped by your demographic profile, "
    "temporal context (time and day of week), and the specific urban "
    "environment of your city.\n  - Identity-Driven Action: Act as a person of"
    " your specific age and role would. \n    - Workers and students should "
    "prioritize consistent blocks for their primary responsibilities during "
    "their typical active schedule. \n    - Homemakers should maintain "
    "realistic, home-centered daily rhythms and adapt plans to household or "
    "family needs as they arise.\n  - Continuity & Realism: Aim for meaningful"
    ", continuous blocks of time (e.g., 30 minutes to 8 hours depending on the"
    " activity). Specifically, recognize that nighttime rest at 'home' is a "
    "long, singular event that should span several hours.\n  - Frequency & "
    "Periodic Activities: Differentiate between daily staples (e.g., work, "
    "school, sleep) and periodic tasks (e.g., grocery shopping, social visits,"
    " or healthcare). Periodic tasks typically occur every few days or weeks; "
    "do not force them into every daily plan unless specifically needed.\n  - "
    "Cognitive Continuity: You possess a persistent memory stream that "
    "captures observations, decisions, and the specific intent behind your "
    "actions. Your memories are a narrative of your life; use past intentions "
    "to maintain social and personal consistency.\n\nAgent Profile (Specific "
    "to this instance):\n  - Identity: 33-year-old female homemaker.\n  - "
    "Location: Resident of San Francisco.\n  - Nuances: employment status: "
    "non_worker; work schedule: not_applicable; school type: not_in_school; "
    "household vehicles: 2; household income: 50k to 74k; household lifecycle:"
    " two or more adults no children"
)
DEFAULT_USER_MESSAGE = (
    "A new day has started (Monday). Provide a rough mental sketch of your "
    "plans for the day ahead.\n\nCurrent State:\n  - Starting Location: Home"
    "\nPast Routines & Insights:\n  - No notable memories yet.\n\nPlanning "
    "Rules:\n  - Use the insights above to inform your routine, but do not "
    "simply repeat them verbatim. Adapt your intentions to the specific day of"
    " the week and your identity.\n  - Frequency Control: Distinguish between "
    "daily \"anchor\" activities and periodic errands. If a past memory shows "
    "you went shopping or visited a friend, treat that as a periodic event"
    "\u2014do not plan to do it again today unless it realistically fits your "
    "current needs.\n  - Focus on your major intentions and rough timing "
    "(e.g., when you'll head to work or school and anything special you have "
    "planned for the day).\n  - Maintain a high-level perspective; avoid rigid"
    " minute-by-minute constraints.\n  - Format: 2-3 concise sentences.\n  - "
    "Output: A short narrative paragraph.\n\nActivity Categories:\n  - "
    "Available categories: Home, Work, Eat Meal, Education, Recreational, "
    "Shopping, Care, Community, Other, Social Visit\nCategory Notes:\n    - "
    "Home: Home activities, including staying at home for personal routines.\n"
    "    - Work: Work and work-related destinations.\n    - Eat Meal: Going "
    "out to eat (restaurant/cafe/food pickup destination).\n    - Education: "
    "Education-related destinations, including school and daycare/child-care "
    "attendance.\n    - Recreational: Recreation, leisure, and exercise "
    "destinations.\n    - Shopping: Buying goods and services.\n    - Care: "
    "Health care and adult-care destinations.\n    - Community: Volunteer, "
    "religious, and community activities.\n    - Other: Other/general purposes"
    " not covered by the categories above.\n    - Social Visit: Visiting "
    "friends or relatives (often at another residence).\n\n\n"
)
DEFAULT_BASE_MODEL_ID = "openai/gpt-oss-20b"
# DEFAULT_ADAPTER_PATH = (
#     "/scratch/pbhanda2/projects/mmv4/checkpoints_old/"
#     "epoch_0_step_15/model"
# )
# DEFAULT_ADAPTER_PATH = (
#     "/scratch/pbhanda2/projects/mmv4/checkpoints/epoch_1_step_225/model"
# )
# DEFAULT_ADAPTER_PATH = (
#     # "/scratch/pbhanda2/projects/mmv4/checkpoints/epoch_1_step_225/model"
#     "/scratch/pbhanda2/projects/mmv4/checkpoints/epoch_1_step_225/model"
# )
DEFAULT_ADAPTER_PATH = (
    "/scratch/pbhanda2/projects/mmv4/"
    "checkpoints_gpt_oss_20b_miniworld_mobility_reasoning-v2/epoch_9_step_7499"
    "/model"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run one prompt with base HF model and again with PEFT adapter."
        )
    )
    parser.add_argument(
        "--base-model-id",
        default=DEFAULT_BASE_MODEL_ID,
        help="Hugging Face base model id or local model path.",
    )
    parser.add_argument(
        "--adapter-path",
        default=DEFAULT_ADAPTER_PATH,
        help="Path to PEFT adapter directory (adapter_config.json present).",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=DEFAULT_ADAPTER_PATH,
        help=(
            "Optional tokenizer path. If unset, adapter path is used when it "
            "contains tokenizer files; otherwise base model tokenizer is used."
        ),
    )
    parser.add_argument(
        "--system-instruction",
        default=DEFAULT_SYSTEM_INSTRUCTION,
        help="System instruction text.",
    )
    parser.add_argument(
        "--user-message",
        default=DEFAULT_USER_MESSAGE,
        help="User message text.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Use 0.0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling value.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model load dtype.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Device map passed to from_pretrained (e.g. "auto", "cuda:0").',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used when sampling is enabled.",
    )
    parser.add_argument(
        "--trust-remote-code",
        default=True,
        dest="trust_remote_code",
        action="store_true",
        help="Enable trust_remote_code for HF loading.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code for HF loading.",
    )
    return parser.parse_args()


def _resolve_torch_dtype(dtype_name: str) -> Any:
    """Map CLI dtype string to torch dtype or 'auto'."""
    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_name]


def _resolve_tokenizer_source(
    base_model_id: str, adapter_path: Path, tokenizer_path: str
) -> str:
    """Resolve tokenizer path with adapter-preferred fallback."""
    if tokenizer_path:
        return tokenizer_path
    if (adapter_path / "tokenizer_config.json").exists():
        return str(adapter_path)
    return base_model_id


def _build_messages(
    system_instruction: str, user_message: str
) -> List[Dict[str, str]]:
    """Build chat messages payload."""
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message},
    ]


def _format_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """Format prompt with tokenizer chat template when available."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        lines: List[str] = []
        for message in messages:
            role = message["role"].upper()
            lines.append(f"{role}: {message['content']}")
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)


def _first_param_device(model: torch.nn.Module) -> torch.device:
    """Get device for model input tensors."""
    return next(model.parameters()).device


def _prepare_model_inputs(
    tokenizer: Any, prompt_text: str, model: torch.nn.Module
) -> Dict[str, torch.Tensor]:
    """Tokenize prompt and place tensors on model device."""
    encoded = tokenizer(prompt_text, return_tensors="pt")
    device = _first_param_device(model)
    return {key: value.to(device) for key, value in encoded.items()}


def _generate(
    model: torch.nn.Module,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    args: argparse.Namespace,
) -> Tuple[str, str]:
    """Generate output text for one model run."""
    prompt_text = _format_prompt(tokenizer, messages)
    inputs = _prepare_model_inputs(tokenizer, prompt_text, model)
    input_len = inputs["input_ids"].shape[-1]

    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
    }
    if tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.eos_token_id
    if tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if args.temperature > 0.0:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        generated = model.generate(**inputs, **generate_kwargs)
    completion_ids = generated[0][input_len:]
    print(f"Generated token IDs: {completion_ids.cpu().tolist()}")
    completion_text = tokenizer.decode(
        completion_ids, skip_special_tokens=True
    )
    return completion_text.strip(), prompt_text


def _load_base_model(
    args: argparse.Namespace, torch_dtype: Any
) -> torch.nn.Module:
    """Load base model from HF/local path."""
    return AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )


def _set_seed(seed: int) -> None:
    """Set RNG seed for deterministic sampling behavior."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clear_cuda_cache() -> None:
    """Release references and clear CUDA cache between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    """Entrypoint."""
    args = parse_args()
    adapter_path = Path(args.adapter_path).expanduser().resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            "adapter_config.json missing in adapter path: "
            f"{adapter_path}"
        )

    torch_dtype = _resolve_torch_dtype(args.dtype)
    tokenizer_source = _resolve_tokenizer_source(
        args.base_model_id, adapter_path, args.tokenizer_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    messages = _build_messages(args.system_instruction, args.user_message)

    run_metadata = {
        "base_model_id": args.base_model_id,
        "adapter_path": str(adapter_path),
        "tokenizer_source": tokenizer_source,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    prompt_metadata = {
        "system_instruction": args.system_instruction,
        "user_message": args.user_message,
    }

    print("RUN_METADATA")
    print(json.dumps(run_metadata, indent=2, ensure_ascii=False))
    print("PROMPT_METADATA")
    print(json.dumps(prompt_metadata, indent=2, ensure_ascii=False))

    _set_seed(args.seed)
    base_model = _load_base_model(args, torch_dtype)
    base_output, prompt_text = _generate(base_model, tokenizer, messages, args)
    del base_model
    _clear_cuda_cache()

    _set_seed(args.seed)
    peft_base_model = _load_base_model(args, torch_dtype)
    peft_model = PeftModel.from_pretrained(peft_base_model, str(adapter_path))
    peft_output, _ = _generate(peft_model, tokenizer, messages, args)

    print("PROMPT_TEXT")
    print(prompt_text)
    print("BASE_MODEL_OUTPUT")
    print(base_output)
    print("PEFT_MODEL_OUTPUT")
    print(peft_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
