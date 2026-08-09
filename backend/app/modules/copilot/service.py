import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an AI Identity Verification Copilot for a digital trust platform.
You help security analysts understand identity verification results, explain fraud indicators,
and generate investigation summaries. Be concise, professional, and factual.
Only use the data provided to you — never make up information."""


def ask_copilot(question, context_data):
    """
    context_data: dict containing verification results (trust score, breakdown, graph info, etc.)
    """
    user_message = f"""Verification Data:
{context_data}

Analyst Question: {question}

Provide a clear, professional answer based only on the data above."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
        max_tokens=500
    )

    return {"answer": response.choices[0].message.content}


def generate_investigation_summary(context_data):
    """
    Automatically generates a summary without needing a specific question.
    """
    prompt = f"""Verification Data:
{context_data}

Generate a brief investigation summary (3-5 sentences) covering:
1. Overall trust assessment
2. Key risk factors (if any)
3. Recommended action (approve/review/reject)"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=400
    )

    return {"summary": response.choices[0].message.content}