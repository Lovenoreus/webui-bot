import requests
import json

BASE_URL = "http://localhost:8009"

def query_cosmic_database(query: str):
    """
    Send a query to the cosmic database endpoint and return the response.
    
    Args:
        query (str): The query string to search the cosmic database
        
    Returns:
        dict: Response from the endpoint containing success status, query, and result/error
    """
    endpoint = f"{BASE_URL}/qdrant/cosmic_database_tool"
    payload = {"query": query}
    
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "query": query,
            "error": str(e)
        }

# Example usage
if __name__ == "__main__":
    # Example query
    sample_query = "Förklara hur CDS FH Referral underlättar remittering till specialistvård och vilken information som inkluderas i den genererade remisstexten."
    result = query_cosmic_database(sample_query)
    
    print("Response from cosmic database:")
    print(json.dumps(result, indent=2))