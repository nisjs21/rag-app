import chromadb
from extraction import extract_text_from_pdf
from chunking import chunk
from llm import create_embedding, ask_llm

client = chromadb.Client()

collection = client.get_or_create_collection("policy")

pdf_path = "policy.pdf"

extracted_text = extract_text_from_pdf(pdf_path)
chunks = chunk(extracted_text, 800, 100)

embeddings = [
    create_embedding(text)
    for text in chunks
]

def add_into_collection(embeddings: list[list[float]], chunks: list[str]):
    collection.add(
        ids= [str(i) for i in range(len(embeddings))],
        embeddings=embeddings,
        documents=chunks
    )

add_into_collection(embeddings, chunks)


def retrieve_context(query: str) -> str:
    query_embedd = create_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedd],
        n_results=7
    )

    retrieved_documents = results["documents"][0]
    retrieved_distances = results["distances"][0]

    for document, distance in zip(retrieved_documents, retrieved_distances):
        print(f"Distance: {distance:.4f}")
        print(f"Document: {document}\n")

    context = "\n\n".join(retrieved_documents)

    return context


def generate_response(query: str) -> str:

    context = retrieve_context(query)

    response = ask_llm(query, context)

    return response   



