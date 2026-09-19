"""
Uses a local Ollama model to parse a free-text user query and extract
the company names being asked about.

Setup: ollama pull llama3.2   (run once, in terminal)
Install: pip install ollama
"""

import json
import ollama

EXTRACTION_PROMPT = """You extract company names from a user's question about Indian stocks.

Return ONLY a JSON array of company names mentioned, nothing else. No explanation.

Examples:
Question: "show me trends for reliance and paytm"
Answer: ["reliance", "paytm"]

Question: "what's happening with HDFC bank stock"
Answer: ["HDFC bank"]

Question: "compare tcs infosys and wipro"
Answer: ["tcs", "infosys", "wipro"]

Question: "{query}"
Answer:"""


def extract_companies(user_query, model="llama3.2"):
    prompt = EXTRACTION_PROMPT.format(query=user_query)

    response = ollama.generate(model=model, prompt=prompt)
    raw_text = response["response"].strip()

    # Ollama sometimes wraps output in markdown code fences - strip those
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        companies = json.loads(raw_text)
        if isinstance(companies, list):
            return [str(c).strip() for c in companies if c]
    except json.JSONDecodeError:
        pass

    # Fallback: if JSON parsing fails, return empty list rather than crashing
    return []


if __name__ == "__main__":
    test_queries = [
        "show me trends and predictions for reliance, paytm etc",
        "what's the sentiment on tata motors",
        "compare hdfc bank and icici bank",
    ]
    for q in test_queries:
        companies = extract_companies(q)
        print(f"Query: {q!r}")
        print(f"Extracted: {companies}\n")