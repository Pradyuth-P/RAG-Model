import os
import logging
from dotenv import load_dotenv
from langchain_core.tracers.context import collect_runs

# Setup logging
logger = logging.getLogger("rag_app.langsmith")

def init_langsmith():
    """
    Validates and initializes LangSmith tracing by checking for both LANGSMITH_* 
    and LANGCHAIN_* environment variables. It maps the active values into 
    os.environ so that the standard LangChain tracing client can pick them up.
    """
    load_dotenv()
    
    # 1. Read Tracing Toggle
    tracing = (
        os.getenv("LANGSMITH_TRACING", "false").lower() == "true" or
        os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    )
    
    # 2. Read API Key
    api_key = (
        os.getenv("LANGSMITH_API_KEY", "").strip() or
        os.getenv("LANGCHAIN_API_KEY", "").strip()
    )
    
    # 3. Read Project Name
    project = (
        os.getenv("LANGSMITH_PROJECT", "").strip() or
        os.getenv("LANGCHAIN_PROJECT", "").strip() or
        "Basic_RAG"
    )
    
    # 4. Read Endpoint
    endpoint = (
        os.getenv("LANGSMITH_ENDPOINT", "").strip() or
        os.getenv("LANGCHAIN_ENDPOINT", "").strip()
    )
    
    if tracing:
        if not api_key:
            logger.warning(
                "LangSmith tracing is toggled on, but no API key was found in "
                "LANGSMITH_API_KEY or LANGCHAIN_API_KEY. Disabling tracing to prevent crash."
            )
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            return False
        
        # Sync variables to standard LANGCHAIN_ environment keys expected by standard tracer
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project.replace('"', '').replace("'", '') # strip outer quotes if any
        if endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = endpoint
            
        logger.info(f"LangSmith Tracing active. Project Name: {os.environ['LANGCHAIN_PROJECT']}")
        return True
    else:
        logger.info("LangSmith Tracing is disabled.")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

def get_tracing_metadata(session_id: str = None) -> dict:
    """
    Generates metadata dictionaries to pass to LangChain run calls
    to label traces clearly in LangSmith.
    """
    metadata = {
        "project": os.getenv("LANGCHAIN_PROJECT", "Basic_RAG")
    }
    if session_id:
        metadata["session_id"] = session_id
    return metadata
