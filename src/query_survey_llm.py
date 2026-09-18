#!/usr/bin/env python3
"""
Survey LLM Query Script - Iterate over Languages and Models
Queries a local LLM server with survey questions in multiple languages
and saves responses in CSV and JSON formats.
"""

import argparse
import json
import time
from typing import Optional
import pandas as pd
import requests
from jinja2 import Template


# ============================================================================
# Configuration
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Query LLM server with survey questions in multiple languages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_survey_llm.py --port 10000
  python query_survey_llm.py --port 8000 --languages gb es it
  python query_survey_llm.py --port 5000 --models minimistral3 apertus
        """
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=10000,
        help='Port number of the llama.cpp server (default: 10000)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host address of the llama.cpp server (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['gb', 'es', 'it', 'cz', 'hu', 'rs', 'ru'],
        help='Languages to process (default: gb es it cz hu rs ru)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['minimistral3'],
        help='LLM models to use (default: minimistral3)'
    )
    
    parser.add_argument(
        '--csv-dir',
        type=str,
        default='./Surveys_parsed',
        help='Directory containing survey CSV files (default: ./Surveys_parsed)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./Surveys_responses',
        help='Directory for output files (default: ./Surveys_responses)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Request timeout in seconds (default: 300)'
    )
    
    return parser.parse_args()


# ============================================================================
# Global Configuration
# ============================================================================

args = parse_arguments()
LLAMA_SERVER_URL = f"http://{args.host}:{args.port}"
CHAT_ENDPOINT = f"{LLAMA_SERVER_URL}/v1/chat/completions"

# Languages to process
LANGUAGES = args.languages

# LLM models to use (supports multiple models)
LLM_NAMES = args.models

# Paths
CSV_DIR = args.csv_dir
OUTPUT_DIR = args.output_dir
REQUEST_TIMEOUT = args.timeout


# ============================================================================
# Functions
# ============================================================================

def query_llm(
    question: str,
    temperature: float = 0.7,
    max_tokens: int = -1,
    timeout: Optional[int] = None
) -> Optional[str]:
    """
    Send a question to the local llama.cpp server and get a response.
    
    Args:
        question: The question/prompt to send
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = more random)
        max_tokens: Maximum tokens in response (-1 = no limit)
        timeout: Request timeout in seconds (uses global REQUEST_TIMEOUT if not specified)
    
    Returns:
        The LLM's response text, or None if error
    """
    if timeout is None:
        timeout = REQUEST_TIMEOUT
    try:
        payload = {
            "model": "local-model",  # llama.cpp uses this default
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        print(f"Querying: {question[:80]}...")
        response = requests.post(
            CHAT_ENDPOINT,
            json=payload,
            timeout=timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return answer
        else:
            print(f"  Error: HTTP {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"  Error: Request timeout after {timeout}s")
        return None
    except requests.exceptions.ConnectionError:
        print(f"  Error: Could not connect to {LLAMA_SERVER_URL}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def load_template(llm_name: str) -> Template:
    """Load Jinja2 template for the specified LLM."""
    template_path = f"./prompts/survey_prompt/survey_prompts_final_answer_{llm_name}.jinja2"
    with open(template_path, 'r', encoding='utf-8') as f:
        template_string = f.read()
    return Template(template_string)


def process_language(language: str, llm_name: str, template: Template) -> dict:
    """
    Process a single language for a specific LLM model.
    
    Returns:
        Dictionary with results and statistics
    """
    print(f"\n{'='*80}")
    print(f"Processing: Language={language}, Model={llm_name}")
    print(f"{'='*80}\n")
    
    # Load CSV
    csv_file_path = f"{CSV_DIR}/ZA7500_q_{language}.csv"
    df = pd.read_csv(csv_file_path)
    
    print(f"Loaded CSV with {len(df)} rows")
    print(f"Columns: {list(df.columns)}\n")
    
    # Prepare query items
    query_items = []
    for idx, row in df.iterrows():
        query_items.append({
            "question_id": row.get('Question Number', ''),
            "question_text": row.get('Question Text', ''),
            "variable": row.get('Variable_Name', ''),
            "option_text": row.get('Response_Options', ''),
            "response_scale": row.get('Response_Scale', ''),
            "csv_index": idx
        })
    
    print(f"Prepared {len(query_items)} query items\n")
    print("Sample items:")
    for i, item in enumerate(query_items[:3], 1):
        print(f"  {i}. [{item['question_id']}] {item['question_text']}")
        print(f"     Variable: {item['variable']} | Option: {item['option_text']}")
    if len(query_items) > 3:
        print(f"  ... and {len(query_items) - 3} more items\n")
    
    # Query LLM for all items
    results = []
    
    print(f"\nQuerying LLM with Survey Questions")
    print(f"{'='*80}\n")
    
    for idx, item in enumerate(query_items, 1):
        # Render prompt from template
        detailed_prompt = template.render(
            language=language,
            question_id=item['question_id'],
            question_text=item['question_text'],
            variable=item['variable'],
            option_text=item['option_text'],
            response_scale=item['response_scale']
        )
        
        print(f"[{idx}/{len(query_items)}] {item['question_id']} - {item['variable']}: {item['option_text']}")
        
        answer = query_llm(detailed_prompt)
        
        if answer:
            results.append({
                "question_id": item['question_id'],
                "question_text": item['question_text'],
                "variable": item['variable'],
                "option_text": item['option_text'],
                "response_scale": item['response_scale'],
                "model_response": answer.strip(),
                "status": "success"
            })
            print(f"  ✓ Response received ({len(answer.strip())} chars)\n")
        else:
            results.append({
                "question_id": item['question_id'],
                "question_text": item['question_text'],
                "variable": item['variable'],
                "option_text": item['option_text'],
                "response_scale": item['response_scale'],
                "model_response": None,
                "status": "failed"
            })
            print(f"  ✗ Failed to get answer.\n")
        
        # Small delay between requests
        time.sleep(0.2)
    
    # Summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"{'='*80}")
    print(f"Completed: {success_count}/{len(results)} items answered")
    print(f"{'='*80}\n")
    
    # Save results
    results_df = pd.DataFrame(results)
    
    print("Results Summary:")
    print(results_df.head())
    
    # CSV output
    csv_output = f"{OUTPUT_DIR}/llm_survey_{llm_name}_responses_{language}.csv"
    results_df.to_csv(csv_output, index=False)
    print(f"\nResults saved to: {csv_output}")
    
    # JSON output
    json_output = f"{OUTPUT_DIR}/llm_survey_{llm_name}_responses_{language}.json"
    with open(json_output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results also saved to: {json_output}")
    
    return {
        "language": language,
        "llm_name": llm_name,
        "total_items": len(results),
        "success_count": success_count,
        "failed_count": len(results) - success_count
    }


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function to process all languages and models."""
    print(f"\n{'='*80}")
    print("Survey LLM Query - Multi-Language Processing")
    print(f"{'='*80}\n")
    
    # Print configuration
    print("Configuration:")
    print(f"  Server URL: {LLAMA_SERVER_URL}")
    print(f"  Languages: {LANGUAGES}")
    print(f"  Models: {LLM_NAMES}")
    print(f"  CSV Directory: {CSV_DIR}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Request Timeout: {REQUEST_TIMEOUT}s\n")
    
    summary_stats = []
    
    for language in LANGUAGES:
        for llm_name in LLM_NAMES:
            try:
                # Load template
                template = load_template(llm_name)
                
                # Process language
                stats = process_language(language, llm_name, template)
                summary_stats.append(stats)
                
            except FileNotFoundError as e:
                print(f"Error: Template or CSV file not found - {e}")
                continue
            except Exception as e:
                print(f"Error processing {language} with {llm_name}: {e}")
                continue
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    for stat in summary_stats:
        print(f"Language: {stat['language']}, Model: {stat['llm_name']}")
        print(f"  Total: {stat['total_items']}, Success: {stat['success_count']}, Failed: {stat['failed_count']}")
        print(f"  Success Rate: {100*stat['success_count']/stat['total_items']:.1f}%\n")


if __name__ == "__main__":
    main()
