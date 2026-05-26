import axios from 'axios';

// Axios instance using proxy configured in vite.config.js
const client = axios.create({
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Get health status (checks active API credentials)
  async getHealth() {
    const response = await client.get('/api/health');
    return response.data;
  },

  // List all uploaded documents for a provider and session
  async getDocuments(embeddingProvider, sessionId = 'default_session') {
    const response = await client.get(`/api/documents?embedding_provider=${embeddingProvider}&session_id=${sessionId}`);
    return response.data;
  },

  // Upload file document associated with a specific session
  async uploadDocument(file, embeddingProvider, chunkSize = 1000, chunkOverlap = 200, sessionId = 'default_session', onUploadProgress = null) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('chunk_size', chunkSize);
    formData.append('chunk_overlap', chunkOverlap);
    formData.append('embedding_provider', embeddingProvider);
    formData.append('session_id', sessionId);

    const response = await client.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress,
    });
    return response.data;
  },

  // Delete document from a session
  async deleteDocument(docId, embeddingProvider, sessionId = 'default_session') {
    const response = await client.delete(`/api/documents/${docId}?embedding_provider=${embeddingProvider}&session_id=${sessionId}`);
    return response.data;
  },

  // Get session history
  async getHistory(sessionId = 'default_session') {
    const response = await client.get(`/api/history?session_id=${sessionId}`);
    return response.data;
  },

  // Clear history
  async clearSessions(sessionId = null) {
    const data = sessionId ? { session_id: sessionId } : {};
    const response = await client.post('/api/clear', data);
    return response.data;
  },
};

/**
 * Streams the chat tokens from the API using fetch and reader chunk decoder.
 * Supports POST requests, which is essential since we need to send query settings.
 */
export async function streamChat(requestData, onEvent, onError) {
  try {
    const sessionId = requestData.sessionId || 'default_session';
    const response = await fetch(`/api/chat?session_id=${sessionId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: requestData.query,
        provider: requestData.provider,
        model: requestData.model,
        temperature: requestData.temperature,
        top_k: requestData.top_k,
        score_threshold: requestData.score_threshold,
      }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({ detail: 'Network error occurred' }));
      throw new Error(errData.detail || 'Failed to initialize chat connection.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');

      // Keep partial last line in buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        const cleaned = line.trim();
        if (cleaned.startsWith('data: ')) {
          try {
            const dataStr = cleaned.slice(6);
            const parsed = JSON.parse(dataStr);
            onEvent(parsed);
          } catch (e) {
            console.error('SSE JSON parsing error', e);
          }
        }
      }
    }
  } catch (err) {
    onError(err);
  }
}
