import os
import requests
# from dotenv import load_dotenv, find_dotenv

# # Load environment variables (optional)
# load_dotenv(find_dotenv())

# Default Ollama config (remote)
OLLAMA_HOST = "http://vs2153.vll.se"
OLLAMA_PORT = "11434"
OLLAMA_URL = f"{OLLAMA_HOST}:{OLLAMA_PORT}/api/chat"
MODEL = "gpt-oss:20b"  # change to your model if needed


def rephrase_query(user_query: str) -> str:
    print(f"User query: {user_query} ")
    """
    Send a query to a remote Ollama server and return a rephrased version
    suitable for SQL generation.
    """

    prompt = f"""
    You are a data assistant that rewrites user questions — written in either Swedish or English — into a clear,
    explicit, and simple query in the same language, so that a SQL generator model can understand it.

    Rules:
    - The rephrased query must always start with "List the" (if in English) or "Lista" (if in Swedish).
    - Keep domain-specific words (e.g., invoice, supplier, date, amount, tax).
    - Add details where missing, such as specifying year and filters, when they can be inferred from context.
    - Do not talk about database tables unless the user explicitly mentions it.
    - Avoid ambiguous terms like "overtime" or "recent" without explanation.
    - Output only the rephrased query — no commentary or explanation.
    - Do not change any number in the query. Preserve all numeric values exactly as in the input, including decimal places (if any). Do not round or reformat numbers.
    - Dates can be rephrased or clarified, but numbers (e.g., 100, 55.78, 0.123) must remain exactly as they appear in the input.

    Examples:
    Input: "Hur mycket spenderade vi på övertid 2024?"
    Output: "Lista det totala fakturabeloppet för 2024 för poster eller beskrivningar som innehåller ordet 'övertid'."

    Input: "How much did we spend on overtime 2024?"
    Output: "List the total invoice amount in 2024 for items or descriptions that contain the word 'overtime'."

    Now rephrase this query:
    "{user_query}"
    """

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.3},
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        repharsed_query=data.get("message", {}).get("content", "").strip()
        print(f"Rephrased query: {repharsed_query}")

        return repharsed_query

    except requests.exceptions.RequestException as e:
        return f"Error communicating with Ollama: {e}"


# Example usage:
if __name__ == "__main__":
    test_queries = [
        # English
        "How many suppliers issued invoices above 1000 in Q1 2024?",
        "Show all invoices for electronic items from German suppliers during the second half of 2023.",
        "How much VAT did we pay on all food-related purchases last year?",
        "Which suppliers billed us more than 5000.000 in total for consulting services between January and June 2024?",
        "Give me all invoices where the payment was delayed more than 30 days, issued before April 2023.",
        "What is the average invoice amount for each supplier in 2025?",
        "List all invoices that contain both 'hardware' and 'support' in the description, from the last 12 months.",
        "Find all credit notes issued in 2024 that relate to overpaid taxes.",
        "How much did we spend per country on logistics between March 2023 and March 2024?",
        "Show invoices that were split across multiple cost centers and totaled more than 2000.000.",
        
        # Swedish
        "Hur många leverantörer skickade fakturor över 1000 under första kvartalet 2024?",
        "Visa alla fakturor för elektronik från tyska leverantörer under andra halvan av 2023.",
        "Hur mycket moms betalade vi för alla matrelaterade inköp förra året?",
        "Vilka leverantörer fakturerade oss mer än 5000.000 totalt för konsulttjänster mellan januari och juni 2024?",
        "Ge mig alla fakturor där betalningen var försenad mer än 30 dagar, utfärdade före april 2023.",
        "Vad är den genomsnittliga fakturabeloppet per leverantör under 2025?",
        "Lista alla fakturor som innehåller både 'hårdvara' och 'support' i beskrivningen, från de senaste 12 månaderna.",
        "Hitta alla kreditfakturor utfärdade 2024 som rör överskjutande skatt.",
        "Hur mycket spenderade vi per land på logistik mellan mars 2023 och mars 2024?",
        "Visa fakturor som delats upp mellan flera kostnadsställen och som totalt överstiger 2000.000."
    ]

    for i, user_question in enumerate(test_queries, start=1):
        rephrased = rephrase_query(user_question)
        print(f"Test {i}")
        print("Original:  ", user_question)
        print("Rephrased: ", rephrased)
        print("-" * 60)

