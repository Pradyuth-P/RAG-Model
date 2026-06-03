import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger("rag_app.memory")

try:
    import redis
    redis_available = True
except ImportError:
    redis_available = False

class RedisSessionMemory:
    def __init__(self):
        self._redis_client = None
        self._in_memory_db: Dict[str, List[Dict[str, Any]]] = {}
        
        # Load configurations from environment (sensible defaults match standard local Redis)
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", 6379))
        self.db = int(os.getenv("REDIS_DB", 0))
        self.password = os.getenv("REDIS_PASSWORD", None)

    def _get_client(self):
        if not redis_available:
            return None
        if self._redis_client is None:
            try:
                client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    socket_timeout=2.0,
                    decode_responses=True
                )
                client.ping()
                self._redis_client = client
                logger.info(f"Successfully connected to Redis at {self.host}:{self.port} for session history.")
            except Exception as e:
                logger.warning(
                    f"Could not connect to Redis on {self.host}:{self.port}: {str(e)}. "
                    "Falling back to in-memory session database."
                )
                self._redis_client = False  # Represent tried but failed
        return self._redis_client if self._redis_client is not False else None

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        client = self._get_client()
        if client:
            try:
                key = f"rag_session:{session_id}:history"
                data = client.get(key)
                if data:
                    return json.loads(data)
                return []
            except Exception as e:
                logger.error(f"Redis get_messages error: {str(e)}")
        
        return self._in_memory_db.get(session_id, [])

    def save_message(self, session_id: str, message: Dict[str, Any]):
        client = self._get_client()
        if client:
            try:
                key = f"rag_session:{session_id}:history"
                history = self.get_messages(session_id)
                history.append(message)
                client.set(key, json.dumps(history))
                return
            except Exception as e:
                logger.error(f"Redis save_message error: {str(e)}")

        if session_id not in self._in_memory_db:
            self._in_memory_db[session_id] = []
        self._in_memory_db[session_id].append(message)

    def clear_session(self, session_id: str = None):
        """
        Clears history database.
        """
        client = self._get_client()
        if session_id:
            if client:
                try:
                    key = f"rag_session:{session_id}:history"
                    client.delete(key)
                except Exception as e:
                    logger.error(f"Redis delete key error: {str(e)}")
            if session_id in self._in_memory_db:
                self._in_memory_db[session_id] = []
        else:
            if client:
                try:
                    keys = client.keys("rag_session:*:history")
                    if keys:
                        client.delete(*keys)
                except Exception as e:
                    logger.error(f"Redis delete all keys error: {str(e)}")
            self._in_memory_db.clear()
