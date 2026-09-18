#!/usr/bin/env python3
"""CLI wrapper around the notebook PDF extraction functions.

Prompts for extracting questions and variables are language-specific and loaded
from ./prompts/survey_prompt/pdf_extraction/<language_code>.txt.
"""

# CLI examples:
# python src/extract_survey_csv_from_pdf.py ./Surveys/ZA7500_q_gb.pdf --language gb --host 127.0.0.1 --port 10106 --model gemma-4-31B-it-UD-Q8_K_XL.gguf
# python src/extract_survey_csv_from_pdf.py ./Surveys/ZA7500_q_es.pdf --language es --host 127.0.0.1 --port 10106 --model gemma-4-31B-it-UD-Q8_K_XL.gguf
# python src/extract_survey_csv_from_pdf.py ./Surveys/ZA7500_q_it.pdf --language it --host 127.0.0.1 --port 10106 --model gemma-4-31B-it-UD-Q8_K_XL.gguf
# Note: the extraction prompt is different for each language code and is read
# from ./prompts/survey_prompt/pdf_extraction/<language_code>.txt.

import argparse
import csv
import os
from pathlib import Path
from typing import Optional

import requests


LLAMA_SERVER_URL = "http://127.0.0.1:10106"
CHAT_ENDPOINT = f"{LLAMA_SERVER_URL}/v1/chat/completions"
MODEL_NAME = "gemma-4-31B-it-UD-Q8_K_XL.gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract survey questions and response options from a PDF into CSV "
            "using a local llama.cpp OpenAI-compatible server. "
            "The extraction prompt differs by language and is loaded from "
            "./prompts/survey_prompt/pdf_extraction/<language_code>.txt."
        ),
        epilog=(
            "Examples:\n"
            "  python src/extract_survey_csv_from_pdf.py ./Surveys/ZA7500_q_gb.pdf --language gb --host 127.0.0.1 --port 10106 --model gemma-4-31B-it-UD-Q8_K_XL.gguf\n"
            "  python src/extract_survey_csv_from_pdf.py ./Surveys/ZA7500_q_es.pdf --language es --host 127.0.0.1 --port 10106 --model gemma-4-31B-it-UD-Q8_K_XL.gguf\n"
            "  python src/extract_survey_csv_from_pdf.py ./Surveys/ZA7500_q_it.pdf --language it --host 127.0.0.1 --port 10106 --model gemma-4-31B-it-UD-Q8_K_XL.gguf"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pdf_path", type=Path, help="Path to input PDF file")
    parser.add_argument(
        "--language",
        required=True,
        help=(
            "Language code used to auto-load prompt file "
            "(expects <prompts-dir>/<language>.txt)"
        ),
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=Path("./prompts/survey_prompt/pdf_extraction"),
        help="Directory containing language prompt files (default: ./prompts/survey_prompt/pdf_extraction)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: ./Surveys_parsed/<pdf_stem>_<language>.csv)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host of llama.cpp server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10106,
        help="Port of llama.cpp server (default: 10106)",
    )
    parser.add_argument("--model", type=str, default="local-model", help="Model name (default: local-model)")
    parser.add_argument(
        "--timeout",
        type=int,
        default=30000,
        help="Request timeout in seconds (default: 30000)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature (default: 0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=-1,
        help="Max tokens for response (-1 means server default/no hard limit)",
    )
    return parser.parse_args()


def load_prompt(prompts_dir: Path, language_code: str) -> str:
    prompt_path = prompts_dir / f"{language_code}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def query_llm(
    question: str,
    temperature: float = 0,
    max_tokens: int = -1,
    timeout: int = 30000,
) -> Optional[str]:
    """Send a question to the local llama.cpp server and get a response."""
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        response = requests.post(
            CHAT_ENDPOINT,
            json=payload,
            timeout=timeout,
        )

        if response.status_code == 200:
            result = response.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return answer

        print(f"Error: HTTP {response.status_code}")
        print(f"Response: {response.text[:500]}")
        return None
    except requests.exceptions.Timeout:
        print(f"Error: Request timeout after {timeout}s")
        return None
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {LLAMA_SERVER_URL}")
        return None
    except Exception as exc:
        print(f"Error: {exc}")
        return None


def query_llm_with_pdf(
    pdf_file_path: str,
    prompt: str,
    temperature: float = 0,
    max_tokens: int = -1,
    timeout: int = 30000,
) -> Optional[str]:
    """Extract text from a PDF and send it with a prompt to local llama.cpp server."""
    try:
        if not os.path.exists(pdf_file_path):
            print(f"Error: PDF file not found: {pdf_file_path}")
            return None

        try:
            import pdfplumber
        except ImportError:
            print("Error: pdfplumber is not installed. Install with: pip install pdfplumber")
            return None

        pdf_text = ""
        with pdfplumber.open(pdf_file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    pdf_text += f"\n\n--- Page {page_num} ---\n{page_text}"

        if not pdf_text.strip():
            print("Error: Could not extract text from PDF")
            return None

        combined_content = f"PDF Content:\n{pdf_text}\n\n{prompt}"

        return query_llm(
            question=combined_content,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return None


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def validate_csv_text(csv_text: str) -> None:
    rows = list(csv.reader(csv_text.splitlines()))
    if not rows:
        raise ValueError("Model returned empty CSV content")


def resolve_output_path(output_csv: Path | None, pdf_path: Path, language_code: str) -> Path:
    if output_csv is not None:
        return output_csv
    return Path("./Surveys_parsed") / f"{pdf_path.stem}_{language_code}.csv"


def main() -> int:
    global LLAMA_SERVER_URL
    global CHAT_ENDPOINT
    global MODEL_NAME

    args = parse_args()

    LLAMA_SERVER_URL = f"http://{args.host}:{args.port}"
    CHAT_ENDPOINT = f"{LLAMA_SERVER_URL}/v1/chat/completions"
    MODEL_NAME = args.model

    if not args.pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {args.pdf_path}")

    prompt_text = load_prompt(args.prompts_dir, args.language)
    response = query_llm_with_pdf(
        pdf_file_path=str(args.pdf_path),
        prompt=prompt_text,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    if response is None:
        raise RuntimeError("Failed to get response from llama.cpp")

    csv_response = strip_code_fences(response)
    validate_csv_text(csv_response)

    output_path = resolve_output_path(args.output_csv, args.pdf_path, args.language)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(csv_response + "\n", encoding="utf-8")

    print(f"LLM Server URL: {LLAMA_SERVER_URL}")
    print(f"Language code: {args.language}")
    print(f"Prompt file: {(args.prompts_dir / f'{args.language}.txt').resolve()}")
    print(f"Output CSV: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
