import os
import uuid
import json
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse

# Imports
from app.models.schemas import ChatRequest, DocumentResponse, DeleteResponse, HealthResponse
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import get_vector_store
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.langsmith_config import init_langsmith

logger = logging.getLogger("rag_app.routes")

router = APIRouter(prefix="/api")

# Singletons for services
embedding_service = EmbeddingService()
vector_store = get_vector_store()
llm_service = LLMService()
rag_service = RAGService(embedding_service, vector_store, llm_service)

# session memory is now managed by rag_service.session_memory

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    embedding_provider: str = Form("huggingface"),
    session_id: str = Form("default_session")
):
    """
    Accepts document, uploads it to uploads folder, runs document splitter,
    generates embeddings, and stores in the FAISS index.
    """
    filename = file.filename
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext not in [".pdf", ".docx", ".txt", ".md"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format: {ext}. Only PDF, DOCX, TXT, and MD are allowed."
        )

    # Make temporary path
    upload_id = str(uuid.uuid4())
    temp_filename = f"{upload_id}_{filename}"
    file_path = os.path.join(vector_store.uploads_dir, temp_filename)

    try:
        # Save file to disk
        with open(file_path, "wb") as buffer:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            buffer.write(content)

        # Trigger RAG ingestion
        logger.info(f"Uploading file {filename} to {file_path} for session {session_id}")
        result = rag_service.ingest_document(
            file_path=file_path,
            session_id=session_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            provider=embedding_provider
        )

        return DocumentResponse(
            id=result["id"],
            filename=result["filename"],
            uploaded_at=result["uploaded_at"],
            file_size=result["file_size"],
            chunk_count=result["chunk_count"],
            embedding_provider=result["embedding_provider"]
        )

    except ValueError as val_err:
        logger.error(f"Validation error during upload: {str(val_err)}")
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        logger.error(f"Failure during upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")


from app.services.guardrails import PIIMasker, PromptInjectionFilter, OutputGuard

pii_masker = PIIMasker()
prompt_filter = PromptInjectionFilter()
output_guard = OutputGuard()

@router.post("/chat")
async def chat_endpoint(request: ChatRequest, session_id: Optional[str] = Query(None)):
    """
    Query the knowledge base and stream back the LLM answer alongside retrieved sources.
    Uses Server-Sent Events (SSE) format to support real-time token streaming.
    """
    current_session = session_id or "default_session"

    # Quick validation of LLM credentials
    supported = llm_service.get_supported_info()
    if request.provider not in supported or not supported[request.provider]["available"]:
        raise HTTPException(
            status_code=400,
            detail=f"LLM Provider '{request.provider}' is not configured. Please supply its API key in the environment."
        )

    # 1. Input Guardrail: Prompt Injection Detection
    if prompt_filter.is_injection(request.query):
        raise HTTPException(
            status_code=400,
            detail="Safety Alert: Query was flagged as a potential prompt injection attack."
        )

    # 2. Input Guardrail: PII Masking
    masked_query = pii_masker.mask(request.query)

    # Store user message using memory service
    user_msg = {
        "role": "user",
        "content": masked_query,
        "timestamp": datetime.utcnow().isoformat()
    }
    rag_service.session_memory.save_message(current_session, user_msg)

    # Dynamic conversation summary generation (Phase 6 Memory)
    rag_service.summary_memory.update_summary_if_needed(
        session_id=current_session,
        llm_provider=request.provider,
        llm_model=request.model
    )

    async def sse_generator():
        bot_response_tokens = []
        retrieved_sources = []
        
        try:
            # We obtain RAG stream
            embedding_prov = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "huggingface")

            async for event in rag_service.generate_response_stream(
                query=masked_query,
                embedding_provider=embedding_prov,
                llm_provider=request.provider,
                session_id=current_session,
                llm_model=request.model,
                temperature=request.temperature,
                top_k=request.top_k,
                score_threshold=request.score_threshold
            ):
                if event["type"] == "sources":
                    retrieved_sources = event["content"]
                elif event["type"] == "token":
                    bot_response_tokens.append(event["content"])
                
                # Format to SSE standard: "data: <json>\n\n"
                yield f"data: {json.dumps(event)}\n\n"

            # 3. Output Guardrail: Content & Safety compliance check
            full_response = "".join(bot_response_tokens)
            safety_check = output_guard.validate_output(full_response)
            
            if not safety_check["is_safe"]:
                logger.warning(f"Output Guardrail triggered: {safety_check['violation_type']} - {safety_check['reason']}")
                # Yield error event
                err_msg = f"Safety Alert: Response blocked due to policy violation: {safety_check['reason']}"
                yield f"data: {json.dumps({'type': 'error', 'content': err_msg})}\n\n"
                # Override the saved message
                saved_content = "Response blocked: The generated content violated safety policy."
            else:
                saved_content = full_response

            # Store finished bot response in session db using memory service
            assistant_msg = {
                "role": "assistant",
                "content": saved_content,
                "sources": retrieved_sources,
                "timestamp": datetime.utcnow().isoformat()
            }
            rag_service.session_memory.save_message(current_session, assistant_msg)

        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@router.get("/documents", response_model=List[DocumentResponse])
def get_documents(embedding_provider: str = "huggingface", session_id: str = Query("default_session")):
    """
    Returns list of all indexed documents for the active embedding provider.
    """
    try:
        docs = vector_store.list_documents(embedding_provider, session_id=session_id)
        return [
            DocumentResponse(
                id=doc["id"],
                filename=doc["filename"],
                uploaded_at=doc["uploaded_at"],
                file_size=doc["file_size"],
                chunk_count=doc["chunk_count"],
                embedding_provider=doc["embedding_provider"]
            )
            for doc in docs
        ]
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str, embedding_provider: str = "huggingface", session_id: str = Query("default_session")):
    """
    Deletes the selected document from the FAISS vector index, deletes its source
    file from uploads, and updates the manifest.
    """
    try:
        # Load embedding model to recreate deletion index
        embeddings = embedding_service.get_embeddings(embedding_provider)
        success = vector_store.delete_document(doc_id, embedding_provider, embeddings, session_id=session_id)
        
        if success:
            return DeleteResponse(success=True, message=f"Document {doc_id} successfully deleted.")
        else:
            raise HTTPException(status_code=404, detail="Document not found or delete failed.")
            
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
def get_history(session_id: str = "default_session"):
    """
    Returns messages history for a session.
    """
    return rag_service.session_memory.get_messages(session_id)


@router.post("/clear")
def clear_sessions(session_id: Optional[str] = None):
    """
    Clears conversation history.
    """
    rag_service.session_memory.clear_session(session_id)
    rag_service.summary_memory.clear_summary(session_id)
    return {"status": "history cleared"}


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    App health details, including which LLM provider keys are loaded.
    """
    tracing_status = init_langsmith()
    llm_info = llm_service.get_supported_info()
    
    return HealthResponse(
        status="healthy",
        langsmith_tracing=tracing_status,
        embedding_provider_default=os.getenv("DEFAULT_EMBEDDING_PROVIDER", "huggingface"),
        available_providers=llm_info
    )
