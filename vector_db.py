
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from extraction import extract_text_from_pdf
from chunking import chunk
from llm import create_embedding, ask_llm

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

collection_name = "policy"
pdf_path = "policy.pdf"


def add_into_collection(embeddings: list[list[float]], chunks: list[str]):
    from qdrant_client import models

    points = [
        models.PointStruct(
            id=i,
            vector=embedding,
            payload={"text": chunk_text},
        )
        for i, (embedding, chunk_text) in enumerate(zip(embeddings, chunks))
    ]

    client.upsert(
        collection_name=collection_name,
        points=points,
    )


def initialize_collection():
    if not client.collection_exists(collection_name):
        print("Collection does not exist. Create it first.")
        return

    collection_info = client.get_collection(collection_name)

    if collection_info.points_count == 0:
        print("Collection is empty, creating new collection data")

        extracted_text = extract_text_from_pdf(pdf_path)
        chunks = chunk(extracted_text, 800, 100)

        embeddings = [
            create_embedding(text)
            for text in chunks
        ]

        add_into_collection(embeddings, chunks)
    else:
        print("Collection exists, using prebuilt collection")


initialize_collection()


def retrieve_context(query: str) -> str:
    query_embedding = create_embedding(query)

    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=7,
    )

    retrieved_documents = [
        point.payload["text"]
        for point in results.points
        if point.payload and "text" in point.payload
    ]

    context = "\n\n".join(retrieved_documents)
    return context


def generate_response(query: str) -> str:
    context = retrieve_context(query)
    response = ask_llm(query, context)
    return response
