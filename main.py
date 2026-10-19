from fastapi import FastAPI, HTTPException
from chroma_db import generate_response


app = FastAPI()



@app.get("/ask")
def ask(query: str):

    if query is None:
        raise HTTPException(
            status_code=400,
            detail="Bad Request: query cannot be empty."
        )

    return generate_response(query)