import logging
from typing import Dict, Any, List, TypedDict, Optional
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# Import nodes
from app.services.agents.retrieval_agent import retrieve_node
from app.services.agents.evaluator_agent import evaluate_node
from app.services.agents.citation_agent import citation_node

logger = logging.getLogger("rag_app.agents.coordinator")

class AgentState(TypedDict):
    query: str
    embedding_provider: str
    llm_provider: str
    llm_model: Optional[str]
    session_id: str
    top_k: int
    score_threshold: float
    temperature: float
    
    retrieved_docs: List[Any]
    generation: str
    sources: List[Dict[str, Any]]
    evaluation_result: str
    attempts: int

class RAGCoordinator:
    def __init__(self, rag_service):
        self.rag_service = rag_service
        self.graph = self._build_graph()

    def _build_graph(self):
        # 1. Define graph state
        workflow = StateGraph(AgentState)
        
        # 2. Define nodes
        workflow.add_node("retrieve", lambda state: retrieve_node(state, self.rag_service))
        workflow.add_node("evaluate", lambda state: evaluate_node(state, self.rag_service))
        workflow.add_node("generate", lambda state: self._generate_node(state))
        workflow.add_node("citation", citation_node)
        
        # 3. Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "evaluate")
        
        # Conditional edge: loops back to retrieve if evaluation fails, up to attempts limit
        workflow.add_conditional_edges(
            "evaluate",
            self._decide_next_step,
            {
                "relevant": "generate",
                "irrelevant": "retrieve",
                "fallback": "generate"
            }
        )
        
        workflow.add_edge("generate", "citation")
        workflow.add_edge("citation", END)
        
        return workflow.compile()

    def _decide_next_step(self, state: AgentState) -> str:
        return state["evaluation_result"]

    def _generate_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes standard generator logic using retrieved context chunks and summary history.
        """
        logger.info("Generator Agent: Synthesizing response...")
        query = state["query"]
        retrieved_docs = state.get("retrieved_docs", [])
        
        # Format context parts
        context_parts = []
        for doc, _ in retrieved_docs:
            source_name = doc.metadata.get("source", "Unknown")
            page_num = doc.metadata.get("page", 1)
            context_parts.append(f"[Source: {source_name}, Page: {page_num}]\n{doc.page_content}")
            
        context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."
        
        # Fetch summary if available
        summary = self.rag_service.summary_memory.get_summary(state["session_id"])
        
        system_prompt = (
            "You are an AI assistant answering only from retrieved context.\n"
            "Rules:\n"
            "Do not hallucinate.\n"
            "If information is unavailable say: 'I could not find relevant information.'\n"
            "Use retrieved chunks only.\n"
            "Mention source documents."
        )
        if summary:
            system_prompt += f"\n\nHere is a summary of the previous conversation so far:\n{summary}"

        user_prompt = f"Context:\n{context_str}\n\nQuestion:\n{query}\n\nAnswer:"
        
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            llm = self.rag_service.llm_service.get_llm(
                provider=state["llm_provider"],
                model=state["llm_model"],
                temperature=state["temperature"]
            )
            response = llm.invoke(messages)
            return {"generation": response.content}
        except Exception as e:
            logger.error(f"Generator Agent error: {str(e)}")
            return {"generation": f"⚠️ Agent Generation Error: {str(e)}"}

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the Compiled LangGraph state machine.
        """
        state_input = {
            "query": inputs["query"],
            "embedding_provider": inputs.get("embedding_provider", "huggingface"),
            "llm_provider": inputs.get("llm_provider", "groq"),
            "llm_model": inputs.get("llm_model", None),
            "session_id": inputs.get("session_id", "default_session"),
            "top_k": inputs.get("top_k", 5),
            "score_threshold": inputs.get("score_threshold", 0.0),
            "temperature": inputs.get("temperature", 0.3),
            "retrieved_docs": [],
            "generation": "",
            "sources": [],
            "evaluation_result": "relevant",
            "attempts": 0
        }
        
        return self.graph.invoke(state_input)
