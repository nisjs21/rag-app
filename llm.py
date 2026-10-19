from openai import OpenAI
import os



client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def create_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def ask_llm(question: str, context: str) -> str:

    prompt = f"""
        Answer the user's question using only the provided context.

        Rules:
        1. Do not use outside knowledge.
        2. Do not guess or invent facts.
        3. If the context does not contain enough information to answer,
        4.When the provided context contains a policy ID relevant to the answer, always include that policy ID in your response.
        respond: "I don't have enough information in the provided context."

        Context:
        {context}

        Question:
        {question}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
