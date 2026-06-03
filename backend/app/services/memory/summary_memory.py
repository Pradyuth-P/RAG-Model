import os
import json
import logging
from typing import Dict, Optional, List
from app.services.llm_service import LLMService

logger = logging.getLogger("rag_app.summary_memory")

class SummaryMemory:
    def __init__(self, session_memory, llm_service: LLMService, summary_threshold: int = 10):
        self.session_memory = session_memory
        self.llm_service = llm_service
        self.summary_threshold = summary_threshold
        self._in_memory_summaries: Dict[str, str] = {}

    def get_summary(self, session_id: str) -> Optional[str]:
        client = self.session_memory._get_client()
        if client:
            try:
                key = f"rag_session:{session_id}:summary"
                return client.get(key)
            except Exception as e:
                logger.error(f"Redis get_summary error: {str(e)}")
        return self._in_memory_summaries.get(session_id)

    def _save_summary(self, session_id: str, summary: str):
        client = self.session_memory._get_client()
        if client:
            try:
                key = f"rag_session:{session_id}:summary"
                client.set(key, summary)
                return
            except Exception as e:
                logger.error(f"Redis save_summary error: {str(e)}")
        self._in_memory_summaries[session_id] = summary

    def clear_summary(self, session_id: str = None):
        client = self.session_memory._get_client()
        if session_id:
            if client:
                try:
                    key = f"rag_session:{session_id}:summary"
                    client.delete(key)
                except Exception as e:
                    logger.error(f"Redis delete summary key error: {str(e)}")
            if session_id in self._in_memory_summaries:
                self._in_memory_summaries[session_id] = ""
        else:
            if client:
                try:
                    keys = client.keys("rag_session:*:summary")
                    if keys:
                        client.delete(*keys)
                except Exception as e:
                    logger.error(f"Redis delete all summaries error: {str(e)}")
            self._in_memory_summaries.clear()

    def update_summary_if_needed(self, session_id: str, llm_provider: str = "groq", llm_model: str = None) -> bool:
        """
        Summarizes past conversations if history meets or exceeds threshold, storing it and pruning old history.
        """
        history = self.session_memory.get_messages(session_id)
        if len(history) < self.summary_threshold:
            return False

        logger.info(
            f"Dialogue history count ({len(history)}) meets or exceeds threshold "
            f"({self.summary_threshold}). Generating summary..."
        )

        try:
            llm = self.llm_service.get_llm(provider=llm_provider, model=llm_model, temperature=0.2)
            
            # Format dialogue
            formatted_dialogue = ""
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                formatted_dialogue += f"{role}: {msg['content']}\n"

            prompt = (
                "You are an AI assistant tasked with summarizing conversation histories. "
                "Produce a concise running summary of the conversation so far, focusing on key facts and user questions. "
                "Keep the summary brief (under 150 words).\n\n"
                f"Conversation History:\n{formatted_dialogue}\n\n"
                "Summary:"
            )
            
            response = llm.invoke(prompt)
            new_summary = response.content.strip()
            logger.info(f"Generated new conversation summary for session {session_id}.")

            # Save summary
            self._save_summary(session_id, new_summary)

            # Prune history: keep only the last 4 messages to preserve immediate conversational context
            pruned_history = history[-4:]
            
            # Save pruned history
            client = self.session_memory._get_client()
            if client:
                key = f"rag_session:{session_id}:history"
                client.set(key, json.dumps(pruned_history))
            else:
                self.session_memory._in_memory_db[session_id] = pruned_history

            return True

        except Exception as e:
            logger.error(f"Failed to generate session summary: {str(e)}")
            return False
