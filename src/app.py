from fastapi import FastAPI, Request, Form, HTTPException
from typing import Annotated
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from helper import ingest_documents, GetRAGResponse
import json
import uvicorn

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load Document and Split it into chunks
doc_file = STATIC_DIR / "docs/Resume.pdf"
# Load and store document
try:
    ingest_documents(doc_file)
except Exception as error:
    print(f"Document ingestion failed: {error}")
    raise HTTPException(status_code=500, detail="Document ingestion failed")


@app.get("/health")
async def CheckHealth():
    return JSONResponse({"success": True, "status": "OK"})


@app.get("/", response_class=HTMLResponse)
async def Home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/chat")
async def Chat(input: Annotated[str, Form()]):
    if not input or not input.strip():
        raise HTTPException(status_code=400, detail="please enter a question")

    return StreamingResponse(GetRAGResponse(input.strip()), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
