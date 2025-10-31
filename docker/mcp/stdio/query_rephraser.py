"""
Query Rephraser with Config Integration
Rephrases user queries for better database searching with LIKE suggestions
Uses LangChain clients from config.py
"""

import json
from time import perf_counter
from datetime import datetime
from pathlib import Path
from langchain_core.messages import HumanMessage
from config import QUERY_REPHRASER_CLIENT

# Timing log file
TIMING_LOG_FILE = "timing.json"


def get_model_name(client):
    """Extract model name from LangChain chat client"""
    # Try different attribute names used by different providers
    for attr in ['model', 'model_name', 'model_id', 'name']:
        if hasattr(client, attr):
            value = getattr(client, attr)
            if value:
                return value
    return "unknown"


def normalize_date_target(target: str) -> list[str]:
    """
    Convert natural language date expressions to SQL-friendly formats.
    Returns a list of possible SQL date patterns.
    
    The LLM should provide month names with years (e.g., "January 2025").
    If month is provided without year, keep as text - let LLM handle context.
    
    Examples:
        "March 2025" -> ["2025-03"]
        "January 2024" -> ["2024-01"]
        "2024" -> ["2024"]
        "January" -> ["January"] (LLM should have added year)
        "Abbott" -> ["Abbott"] (not a date)
        "5000" -> ["5000"] (not a date)
    """
    target = target.strip()
    patterns = []
    
    # Month names mapping
    months = {
        'january': '01', 'jan': '01',
        'february': '02', 'feb': '02',
        'march': '03', 'mar': '03',
        'april': '04', 'apr': '04',
        'may': '05',
        'june': '06', 'jun': '06',
        'july': '07', 'jul': '07',
        'august': '08', 'aug': '08',
        'september': '09', 'sep': '09', 'sept': '09',
        'october': '10', 'oct': '10',
        'november': '11', 'nov': '11',
        'december': '12', 'dec': '12'
    }
    
    # Swedish month names
    swedish_months = {
        'januari': '01',
        'februari': '02',
        'mars': '03',
        'april': '04',
        'maj': '05',
        'juni': '06',
        'juli': '07',
        'augusti': '08',
        'september': '09',
        'oktober': '10',
        'november': '11',
        'december': '12'
    }
    
    target_lower = target.lower()
    
    # Check for "Month YYYY" pattern (e.g., "March 2025", "January 2024")
    for month_name, month_num in {**months, **swedish_months}.items():
        if month_name in target_lower:
            # Extract year if present
            import re
            year_match = re.search(r'\b(20\d{2})\b', target)
            if year_match:
                year = year_match.group(1)
                patterns.append(f"{year}-{month_num}")  # 2025-03
                return patterns
            else:
                # Month without year - LLM should have provided year in context
                # Keep as-is and let it be handled as text
                return [target]
    
    # Check for year-only pattern (e.g., "2024", "2025")
    import re
    if re.match(r'^20\d{2}$', target):
        patterns.append(f"{target}")  # Just the year
        return patterns
    
    # Not a date pattern, return as-is
    return [target]


def normalize_targets(targets: list[str]) -> list[str]:
    """
    Normalize all targets, expanding date expressions into SQL-friendly formats.
    """
    normalized = []
    for target in targets:
        date_patterns = normalize_date_target(target)
        normalized.extend(date_patterns)
    return normalized


def load_timing_data():
    """Load existing timing data from JSON file with error recovery"""
    if not Path(TIMING_LOG_FILE).exists():
        return []
    
    try:
        with open(TIMING_LOG_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # Handle empty file
            if not content:
                return []
            
            data = json.loads(content)
            
            # Ensure it's a list
            if not isinstance(data, list):
                print(f"Warning: timing.json is not a list, resetting to empty list")
                return []
            
            return data
            
    except json.JSONDecodeError as e:
        print(f"ERROR: Corrupted timing.json file: {e}")
        
        # Try to backup the corrupted file
        backup_file = f"{TIMING_LOG_FILE}.corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            import shutil
            shutil.copy2(TIMING_LOG_FILE, backup_file)
            print(f"Corrupted file backed up to: {backup_file}")
        except Exception as backup_error:
            print(f"Could not backup corrupted file: {backup_error}")
        
        # Try to recover from backup file if it exists
        backup_file_pattern = f"{TIMING_LOG_FILE}.backup"
        if Path(backup_file_pattern).exists():
            print(f"Attempting to recover from backup...")
            try:
                with open(backup_file_pattern, 'r', encoding='utf-8') as f:
                    # Backup file has one JSON object per line
                    recovered_data = []
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                recovered_data.append(entry)
                            except:
                                continue
                    print(f"Recovered {len(recovered_data)} entries from backup")
                    return recovered_data
            except Exception as recover_error:
                print(f"Could not recover from backup: {recover_error}")
        
        # Return empty list as last resort
        print("Starting with empty timing data")
        return []
        
    except Exception as e:
        print(f"ERROR: Could not load timing.json: {e}")
        return []


def save_timing_data(timing_entry):
    """Append timing data to JSON file with robust error handling"""
    import tempfile
    import shutil
    import os
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Load existing data
            timing_data = load_timing_data()
            timing_data.append(timing_entry)
            
            # Write to temporary file first (safer than direct write)
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.json', 
                dir=os.path.dirname(os.path.abspath(TIMING_LOG_FILE)) or '.',
                text=True
            )
            
            try:
                with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
                    json.dump(timing_data, temp_file, indent=2, ensure_ascii=False)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                
                # Atomic move (overwrites existing file)
                shutil.move(temp_path, TIMING_LOG_FILE)
                
                # Successfully written
                return
                
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                raise e
                
        except Exception as e:
            if attempt < max_retries - 1:
                # Retry with a small delay
                import time
                time.sleep(0.1 * (attempt + 1))
                continue
            else:
                # Last attempt failed, log error
                print(f"ERROR: Failed to save timing data after {max_retries} attempts: {e}")
                print(f"Timing entry that failed to save: {timing_entry}")
                
                # Try to save to backup file as last resort
                try:
                    backup_file = f"{TIMING_LOG_FILE}.backup"
                    with open(backup_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(timing_entry, ensure_ascii=False) + '\n')
                    print(f"Saved to backup file: {backup_file}")
                except Exception as backup_error:
                    print(f"ERROR: Could not save to backup: {backup_error}")


def rephrase_query(user_query: str) -> str:
    """
    Send a query to the configured LLM and return a rephrased version
    suitable for SQL generation with LIKE instructions for identified targets.
    Times the LLM call and logs to JSON.
    
    Args:
        user_query: The user's original query
        
    Returns:
        Rephrased query with LIKE suggestions if targets are identified
    """
    print(f"User query: {user_query}")

    prompt = f"""
You are a query rephrasing assistant. Your job is to:
1. Rephrase the user's query into a clear, explicit natural language query suitable for SQL generation
2. Identify specific target values that should be searched for using SQL LIKE operators
3. Return ONLY a JSON object with two fields: "rephrased_query" and "targets"

Rules for rephrasing:
- Keep domain-specific words and proper nouns (e.g., "Abbott", "Siemens", "invoice")
- Add missing details like year if implied from context (e.g., "January" → "January 2024")
- Be explicit about what is being requested (e.g., "Show me" → "List the invoices")
- Translate to English if the query is in another language, but preserve proper nouns
- Do NOT generate SQL code - only natural language
- Avoid ambiguous terms without clarification

Rules for identifying targets:
- Targets are specific constraint values that should be searched using LIKE
- Include: company names, dates, amounts, status values, descriptions, keywords
- Include: numeric thresholds (e.g., "5000", "30 days")
- Include: specific terms the user mentions (e.g., "consulting", "hardware")
- For time periods: ALWAYS include the year with month names (e.g., "January 2025" not just "January")
- For time periods: If user says just "January", infer the year from context (usually current year)
- Year-only is fine (e.g., "2024")
- Do NOT include: table names, column names, SQL keywords, generic terms like "invoice" or "supplier"

Examples:

Input: "Show invoices from Abbott in 2024"
Output: {{"rephrased_query": "List the invoices from Abbott in 2024", "targets": ["Abbott", "2024"]}}

Input: "Which suppliers billed us more than 5000 for consulting between January and June 2024?"
Output: {{"rephrased_query": "List the suppliers that billed more than 5000 in total for consulting services between January and June 2024", "targets": ["5000", "consulting", "January 2024", "June 2024", "2024"]}}

Input: "Show invoices from January"
Output: {{"rephrased_query": "List the invoices from January 2025", "targets": ["January 2025"]}}

Input: "Show invoices without descriptions"
Output: {{"rephrased_query": "List the invoices where description field is empty or null", "targets": []}}

Input: "Hur mycket spenderade vi på övertid 2024?"
Output: {{"rephrased_query": "Lista det totala fakturabeloppet för 2024 för beskrivningar som innehåller övertid", "targets": ["övertid", "2024"]}}

Input: "Lista tyska eller svenska leverantörer med fakturor över 2000 under Q1"
Output: {{"rephrased_query": "Lista leverantörerna från länder som innehåller tyska eller svenska med fakturor över 2000 under Q1", "targets": ["tyska", "svenska", "2000", "Q1"]}}

Input: "Visa fakturor utan beskrivningar"
Output: {{"rephrased_query": "Lista fakturorna där beskrivningsfältet är tomt eller null", "targets": []}}

Now process this query:
Input: "{user_query}"
Output:"""

    # Start timing
    start_time = perf_counter()
    
    try:
        # Call the LLM using config client
        response = QUERY_REPHRASER_CLIENT.invoke([HumanMessage(content=prompt)])
        
        # Stop timing
        elapsed_time = perf_counter() - start_time
        
        # Get response content
        response_content = response.content
        
        print(f"Raw response: {response_content}")
        print(f"LLM call took: {elapsed_time:.4f} seconds")
        
        # Strip markdown code blocks if present (models like phi4 wrap JSON in ```json ... ```)
        cleaned_content = response_content.strip()
        if cleaned_content.startswith("```"):
            # Remove opening ```json or ``` and closing ```
            lines = cleaned_content.split('\n')
            # Remove first line if it's ```json or ```
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned_content = '\n'.join(lines).strip()
        
        # Parse JSON response
        try:
            result = json.loads(cleaned_content)
            rephrased_query = result.get("rephrased_query", "")
            targets = result.get("targets", [])
            
            # Normalize targets (convert dates to SQL-friendly format)
            normalized_targets = normalize_targets(targets)
            
            # Add LIKE instructions if there are targets
            if normalized_targets:
                like_instructions = ", ".join([f'LIKE "%{target}%"' for target in normalized_targets])
                final_query = f"{rephrased_query}\nSuggestions: Use {like_instructions}"
            else:
                final_query = rephrased_query
            
            print(f"Rephrased query: {final_query}")
            print(f"Original targets: {targets}")
            print(f"Normalized targets: {normalized_targets}")
            
            # Log timing data for successful call
            timing_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": "query_rephraser",
                "model": get_model_name(QUERY_REPHRASER_CLIENT),
                "operation": "rephrase_query",
                "elapsed_time_seconds": round(elapsed_time, 4),
                "status": "success",
                "original_query": user_query,
                "rephrased_query": final_query,
                "input_length": len(user_query),
                "output_length": len(final_query),
                "targets_count": len(targets),
                "normalized_targets_count": len(normalized_targets),
                "has_targets": len(targets) > 0,
                "targets": targets,
                "normalized_targets": normalized_targets
            }
            save_timing_data(timing_entry)
            
            return final_query
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            
            # Log timing data for JSON parse error
            timing_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": "query_rephraser",
                "model": get_model_name(QUERY_REPHRASER_CLIENT),
                "operation": "rephrase_query",
                "elapsed_time_seconds": round(elapsed_time, 4),
                "status": "json_parse_error",
                "original_query": user_query,
                "error_message": str(e),
                "raw_response": response_content[:500]  # First 500 chars
            }
            save_timing_data(timing_entry)
            
            # Return the raw response as fallback
            return response_content
            
    except Exception as e:
        elapsed_time = perf_counter() - start_time
        print(f"Error calling LLM: {e}")
        
        # Log timing data for LLM call error
        timing_entry = {
            "timestamp": datetime.now().isoformat(),
            "module": "query_rephraser",
            "model": get_model_name(QUERY_REPHRASER_CLIENT),
            "operation": "rephrase_query",
            "elapsed_time_seconds": round(elapsed_time, 4),
            "status": "llm_error",
            "original_query": user_query,
            "error_message": str(e)
        }
        save_timing_data(timing_entry)
        
        return f"Error: {e}"


def rephrase_query_simple(user_query: str) -> tuple[str, list[str]]:
    """
    Simplified version that returns both rephrased query and targets separately.
    
    Args:
        user_query: The user's original query
        
    Returns:
        Tuple of (rephrased_query, targets)
    """
    print(f"User query: {user_query}")

    prompt = f"""
You are a query rephrasing assistant. Your job is to:
1. Rephrase the user's query into a clear, explicit natural language query suitable for SQL generation
2. Identify specific target values that should be searched for using SQL LIKE operators
3. Return ONLY a JSON object with two fields: "rephrased_query" and "targets"

Rules for rephrasing:
- Keep domain-specific words and proper nouns (e.g., "Abbott", "Siemens", "invoice")
- Add missing details like year if implied from context (e.g., "January" → "January 2024")
- Be explicit about what is being requested (e.g., "Show me" → "List the invoices")
- Translate to English if the query is in another language, but preserve proper nouns
- Do NOT generate SQL code - only natural language
- Avoid ambiguous terms without clarification

Rules for identifying targets:
- Targets are specific constraint values that should be searched using LIKE
- Include: company names, dates, amounts, status values, descriptions, keywords
- Include: numeric thresholds (e.g., "5000", "30 days")
- Include: specific terms the user mentions (e.g., "consulting", "hardware")
- For time periods: ALWAYS include the year with month names (e.g., "January 2025" not just "January")
- For time periods: If user says just "January", infer the year from context (usually current year)
- Year-only is fine (e.g., "2024")
- Do NOT include: table names, column names, SQL keywords, generic terms like "invoice" or "supplier"

Examples:

Input: "Show invoices from Abbott in 2024"
Output: {{"rephrased_query": "List the invoices from Abbott in 2024", "targets": ["Abbott", "2024"]}}

Input: "Which suppliers billed us more than 5000 for consulting between January and June 2024?"
Output: {{"rephrased_query": "List the suppliers that billed more than 5000 in total for consulting services between January and June 2024", "targets": ["5000", "consulting", "January 2024", "June 2024", "2024"]}}

Input: "Show invoices from January"
Output: {{"rephrased_query": "List the invoices from January 2025", "targets": ["January 2025"]}}

Now process this query:
Input: "{user_query}"
Output:"""

    # Start timing
    start_time = perf_counter()
    
    try:
        # Call the LLM using config client
        response = QUERY_REPHRASER_CLIENT.invoke([HumanMessage(content=prompt)])
        
        # Stop timing
        elapsed_time = perf_counter() - start_time
        
        # Get response content
        response_content = response.content
        
        # Strip markdown code blocks if present
        cleaned_content = response_content.strip()
        if cleaned_content.startswith("```"):
            lines = cleaned_content.split('\n')
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned_content = '\n'.join(lines).strip()
        
        # Parse JSON response
        try:
            result = json.loads(cleaned_content)
            rephrased_query = result.get("rephrased_query", "")
            targets = result.get("targets", [])
            
            # Normalize targets (convert dates to SQL-friendly format)
            normalized_targets = normalize_targets(targets)
            
            print(f"Rephrased query: {rephrased_query}")
            print(f"Original targets: {targets}")
            print(f"Normalized targets: {normalized_targets}")
            print(f"LLM call took: {elapsed_time:.4f} seconds")
            
            # Log timing data
            timing_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": "query_rephraser",
                "model": get_model_name(QUERY_REPHRASER_CLIENT),
                "operation": "rephrase_query_simple",
                "elapsed_time_seconds": round(elapsed_time, 4),
                "status": "success",
                "original_query": user_query,
                "rephrased_query": rephrased_query,
                "targets_count": len(targets),
                "normalized_targets_count": len(normalized_targets),
                "has_targets": len(targets) > 0,
                "targets": targets,
                "normalized_targets": normalized_targets
            }
            save_timing_data(timing_entry)
            
            return rephrased_query, normalized_targets
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            
            # Log error
            timing_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": "query_rephraser",
                "model": get_model_name(QUERY_REPHRASER_CLIENT),
                "operation": "rephrase_query_simple",
                "elapsed_time_seconds": round(elapsed_time, 4),
                "status": "json_parse_error",
                "original_query": user_query,
                "error_message": str(e)
            }
            save_timing_data(timing_entry)
            
            # Return raw response as fallback
            return response_content, []
            
    except Exception as e:
        elapsed_time = perf_counter() - start_time
        print(f"Error calling LLM: {e}")
        
        # Log error
        timing_entry = {
            "timestamp": datetime.now().isoformat(),
            "module": "query_rephraser",
            "model": get_model_name(QUERY_REPHRASER_CLIENT),
            "operation": "rephrase_query_simple",
            "elapsed_time_seconds": round(elapsed_time, 4),
            "status": "llm_error",
            "original_query": user_query,
            "error_message": str(e)
        }
        save_timing_data(timing_entry)
        
        return f"Error: {e}", []


if __name__ == "__main__":
    # Test queries
    test_queries = [
        "Show invoices from Abbott in 2024",
        "Which suppliers billed us more than 5000 for consulting?",
        "Visa fakturor utan beskrivningar",
        "Lista tyska eller svenska leverantörer med fakturor över 2000 under Q1"
    ]
    
    print("=" * 60)
    print("QUERY REPHRASER TEST")
    print("=" * 60)
    print()
    
    for query in test_queries:
        print("-" * 60)
        rephrased = rephrase_query(query)
        print()
    
    print("=" * 60)
    print("SIMPLE VERSION TEST")
    print("=" * 60)
    print()
    
    query = "Show invoices from Abbott in 2024"
    rephrased, targets = rephrase_query_simple(query)
    print(f"Rephrased: {rephrased}")
    print(f"Targets: {targets}")