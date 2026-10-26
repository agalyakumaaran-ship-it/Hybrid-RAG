
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

from backend.document_reader import read_document
from backend.text_chunker import split_text
from backend.hybrid_retriever import HybridRetriever

# Load environment variables from the project .env file.
PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

client = OpenAI()

app = FastAPI(
    title="Hybrid RAG Virtual Assistant",
    description="API for our AI virtual assistant",
    version="1.0.0",
)

DOCUMENT_DIR = PROJECT_DIR / "documents"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

retriever = HybridRetriever()

document_status = {
    "loaded": False,
    "filename": None,
    "chunks": 0,
}


def load_document():
    """Load and index a document automatically when the backend starts."""

    document_status.update(
        {"loaded": False, "filename": None, "chunks": 0}
    )

    if not DOCUMENT_DIR.exists():
        print(f"Documents folder not found: {DOCUMENT_DIR}")
        return

    files = sorted(
        (
            path
            for path in DOCUMENT_DIR.rglob("*")
            if path.is_file()
            and path.suffix.lower() in ALLOWED_EXTENSIONS
        ),
        key=lambda path: (
            path.suffix.lower() != ".pdf",
            path.name.lower(),
        ),
    )

    if not files:
        print(f"No PDF, TXT, or DOCX files found in {DOCUMENT_DIR}")
        return

    # Start with the first supported document found.
    # PDFs are preferred over TXT and DOCX files.
    document_path = files[0]

    try:
        print(f"Loading document: {document_path.name}")

        text = read_document(str(document_path))

        if not text or not text.strip():
            print(f"Document contains no readable text: {document_path.name}")
            return

        chunks = split_text(text)

        if not chunks:
            print("No text chunks were created.")
            return

        retriever.build(chunks)

        document_status.update(
            {
                "loaded": True,
                "filename": document_path.name,
                "chunks": len(chunks),
            }
        )

        print(f"Document indexed successfully: {document_path.name}")
        print(f"Chunks indexed: {len(chunks)}")

    except Exception as error:
        print(f"Document loading failed: {error}")


@app.on_event("startup")
def startup_event():
    load_document()


@app.get("/")
def home():
    return {
        "message": "Welcome to your Hybrid RAG Virtual Assistant!",
        "status": "Backend is running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/document-status")
def get_document_status():
    return document_status


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not document_status["loaded"]:
        raise HTTPException(
            status_code=503,
            detail=(
                "No document is indexed. Check the documents folder "
                "and the backend terminal for errors."
            ),
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    try:
        results = retriever.search(question)

        vector_context = "\n\n".join(results["vector_results"])
        graph_context = ", ".join(results["graph_results"])

        response = client.responses.create(
            model="gpt-4o-mini",
          
instructions=(
    "You are a document-only question-answering assistant. "
    "Answer the user's question using ONLY the supplied document context. "
    "Do not use your general knowledge or outside information. "
    "If the document context does not contain enough information to answer, "
    "respond only with: 'The provided document does not contain the answer "
    "to this question.' "
    "Never provide an answer after stating that the information is missing."
),

            input=(
                f"User question:\n{question}\n\n"
                f"Document context:\n{vector_context}\n\n"
                f"Knowledge graph matching terms:\n{graph_context}"
            ),
        )

        return {
            "question": question,
            "answer": response.output_text,
            "vector_results": results["vector_results"],
            "graph_results": results["graph_results"],
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Question processing failed: {error}",
        ) from error
