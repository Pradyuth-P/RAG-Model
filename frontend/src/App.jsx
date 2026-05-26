import React, { useState, useEffect } from 'react';
import { api, streamChat } from './services/api';
import { Menu, HardDrive } from 'lucide-react';

// Components
import HistorySidebar from './components/HistorySidebar';
import ChatWindow from './components/ChatWindow';
import UploadPanel from './components/UploadPanel';
import SourcesPanel from './components/SourcesPanel';
import SettingsPanel from './components/SettingsPanel';

export default function App() {
  // App Configurations & Settings
  const [settings, setSettings] = useState({
    llmProvider: 'groq',
    llmModel: 'llama-3.1-8b-instant',
    temperature: 0.3,
    embeddingProvider: 'huggingface',
    chunkSize: 1000,
    chunkOverlap: 200
  });

  // UI state
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isLeftSidebarOpen, setIsLeftSidebarOpen] = useState(false);
  const [isRightSidebarOpen, setIsRightSidebarOpen] = useState(false);
  const [isSourcesOpen, setIsSourcesOpen] = useState(false);
  const [healthInfo, setHealthInfo] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // Chat & Session state
  const [sessions, setSessions] = useState([
    { id: 'session_1', title: 'Knowledge Session 1' }
  ]);
  const [currentSessionId, setCurrentSessionId] = useState('session_1');
  const [sessionMessages, setSessionMessages] = useState({}); // session_id -> list of messages
  const [activeSources, setActiveSources] = useState([]);

  // 1. Fetch Health and Configuration Status on Startup
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const data = await api.getHealth();
        setHealthInfo(data);
        
        // Auto-configure default provider based on availability
        const providers = Object.entries(data.available_providers);
        const firstAvailable = providers.find(([_, info]) => info.available);
        
        if (firstAvailable) {
          setSettings(prev => ({
            ...prev,
            llmProvider: firstAvailable[0],
            llmModel: firstAvailable[1].default_model,
            embeddingProvider: data.embedding_provider_default
          }));
        } else if (providers.length > 0) {
          // If none are available, fall back to first key in list
          setSettings(prev => ({
            ...prev,
            llmProvider: providers[0][0],
            llmModel: providers[0][1].default_model,
            embeddingProvider: data.embedding_provider_default
          }));
        }
      } catch (err) {
        console.error('Failed to contact backend health API', err);
      }
    };
    fetchHealth();
  }, []);

  // 2. Fetch Documents whenever active embedding provider or session ID changes
  const fetchDocuments = async () => {
    try {
      const docs = await api.getDocuments(settings.embeddingProvider, currentSessionId);
      setDocuments(docs);
    } catch (err) {
      console.error('Failed to load documents list', err);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, [settings.embeddingProvider, currentSessionId]);

  // Load chat history for the active session
  useEffect(() => {
    const fetchSessionHistory = async () => {
      try {
        const history = await api.getHistory(currentSessionId);
        setSessionMessages(prev => ({
          ...prev,
          [currentSessionId]: history
        }));
        
        // Load latest retrieved sources from the last assistant message if available
        const assistantMsgs = history.filter(m => m.role === 'assistant');
        if (assistantMsgs.length > 0) {
          const lastMsg = assistantMsgs[assistantMsgs.length - 1];
          setActiveSources(lastMsg.sources || []);
        } else {
          setActiveSources([]);
        }
      } catch (err) {
        console.error('Failed to load session history', err);
      }
    };
    fetchSessionHistory();
  }, [currentSessionId]);

  // Refresh active documents helper
  const handleRefreshDocs = () => {
    fetchDocuments();
  };

  // Trigger RAG queries and stream tokens back in real-time
  const handleSendMessage = async (query) => {
    setIsLoading(true);
    setIsLeftSidebarOpen(false);
    setIsRightSidebarOpen(false);
    
    // 1. Instantly append user message to local state
    const userMsg = {
      role: 'user',
      content: query,
      timestamp: new Date().toISOString()
    };
    
    const initialMessages = [...(sessionMessages[currentSessionId] || []), userMsg];
    setSessionMessages(prev => ({
      ...prev,
      [currentSessionId]: initialMessages
    }));

    // 2. Setup placeholder bot message
    const botMsgPlaceholder = {
      role: 'assistant',
      content: '',
      sources: [],
      timestamp: new Date().toISOString()
    };

    setSessionMessages(prev => ({
      ...prev,
      [currentSessionId]: [...initialMessages, botMsgPlaceholder]
    }));

    // 3. Initiate SSE Chat Stream
    await streamChat(
      {
        query,
        provider: settings.llmProvider,
        model: settings.llmModel,
        temperature: settings.temperature,
        top_k: 5,
        score_threshold: 0.0,
        sessionId: currentSessionId
      },
      (event) => {
        // SSE Event handler
        if (event.type === 'sources') {
          setActiveSources(event.content);
          setSessionMessages(prev => {
            const list = [...(prev[currentSessionId] || [])];
            if (list.length > 0) {
              list[list.length - 1].sources = event.content;
            }
            return { ...prev, [currentSessionId]: list };
          });
        } 
        else if (event.type === 'token') {
          setSessionMessages(prev => {
            const list = [...(prev[currentSessionId] || [])];
            if (list.length > 0) {
              list[list.length - 1].content += event.content;
            }
            return { ...prev, [currentSessionId]: list };
          });
        }
        else if (event.type === 'error') {
          setSessionMessages(prev => {
            const list = [...(prev[currentSessionId] || [])];
            if (list.length > 0) {
              list[list.length - 1].content = `⚠️ Stream Error: ${event.content}`;
            }
            return { ...prev, [currentSessionId]: list };
          });
        }
        else if (event.type === 'done') {
          setIsLoading(false);
        }
      },
      (err) => {
        // Error handler
        console.error('Chat error', err);
        setSessionMessages(prev => {
          const list = [...(prev[currentSessionId] || [])];
          if (list.length > 0) {
            list[list.length - 1].content = `⚠️ Failed to fetch response. Check if backend is active or if API keys are set.`;
          }
          return { ...prev, [currentSessionId]: list };
        });
        setIsLoading(false);
      }
    );
  };

  // Create a new session
  const handleNewSession = () => {
    const newId = `session_${Date.now()}`;
    const newTitle = `Session ${sessions.length + 1}`;
    setSessions(prev => [...prev, { id: newId, title: newTitle }]);
    setCurrentSessionId(newId);
  };

  // Clear backend and frontend history databases
  const handleClearHistory = async () => {
    if (!confirm('Are you sure you want to clear all conversational histories?')) return;
    try {
      await api.clearSessions();
      setSessionMessages({});
      setActiveSources([]);
      setSessions([{ id: 'session_1', title: 'Knowledge Session 1' }]);
      setCurrentSessionId('session_1');
    } catch (err) {
      alert(`Clear failed: ${err.message}`);
    }
  };

  const getActiveMessages = () => {
    return sessionMessages[currentSessionId] || [];
  };

  const getUniqueDocCount = () => {
    const uniq = new Set(activeSources.map(s => s.source));
    return uniq.size;
  };

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden font-sans relative">
      
      {/* 1. Left Sidebar (History & Settings) - Mobile Drawer + Desktop Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-40 w-64 transform ${
        isLeftSidebarOpen ? 'translate-x-0' : '-translate-x-full'
      } md:translate-x-0 md:static transition-transform duration-200 ease-in-out bg-slate-900`}>
        <HistorySidebar
          sessions={sessions}
          currentSession={currentSessionId}
          onSelectSession={(id) => {
            setCurrentSessionId(id);
            setIsLeftSidebarOpen(false); // Auto close left drawer on mobile
          }}
          onNewSession={handleNewSession}
          onClearHistory={handleClearHistory}
          activeSettings={settings}
          onOpenSettings={() => setIsSettingsOpen(true)}
          healthInfo={healthInfo}
        />
      </div>

      {/* 2. Main Chat Workspace (Center) */}
      <div className="flex-1 flex flex-col min-w-0 h-full border-r border-slate-900">
        {/* Chat top info header */}
        <div className="h-14 border-b border-slate-900 flex items-center justify-between px-4 bg-slate-950">
          <div className="flex items-center gap-2">
            {/* Hamburger button for mobile to open history */}
            <button
              onClick={() => setIsLeftSidebarOpen(true)}
              className="p-1.5 rounded-lg text-slate-400 hover:bg-slate-900 hover:text-slate-200 md:hidden transition"
              title="Open History"
            >
              <Menu className="h-5 w-5" />
            </button>
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span className="text-xs font-semibold text-slate-200 truncate max-w-[140px] sm:max-w-none">
              {sessions.find(s => s.id === currentSessionId)?.title}
            </span>
          </div>
          <div className="flex items-center gap-3">
            {/* Document button for mobile to open upload panel */}
            <button
              onClick={() => setIsRightSidebarOpen(true)}
              className="p-1.5 rounded-lg text-slate-400 hover:bg-slate-900 hover:text-slate-200 md:hidden transition"
              title="Open Documents"
            >
              <HardDrive className="h-5 w-5" />
            </button>
          </div>
        </div>

        <ChatWindow
          messages={getActiveMessages()}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          onSelectSources={(sources) => {
            setActiveSources(sources);
            setIsSourcesOpen(true); // Open sources list as modal overlay
          }}
          hasDocuments={documents.length > 0}
        />
      </div>

      {/* 3. Right Sidebar (Document Manager Only) - Mobile Drawer + Desktop Sidebar */}
      <div className={`fixed inset-y-0 right-0 z-40 w-80 transform ${
        isRightSidebarOpen ? 'translate-x-0' : 'translate-x-full'
      } md:translate-x-0 md:static border-l border-slate-900 h-full bg-slate-950 transition-transform duration-200 ease-in-out`}>
        <div className="h-full p-4 overflow-hidden">
          <UploadPanel
            documents={documents}
            onRefreshDocs={handleRefreshDocs}
            embeddingProvider={settings.embeddingProvider}
            chunkSize={settings.chunkSize}
            chunkOverlap={settings.chunkOverlap}
            sessionId={currentSessionId}
          />
        </div>
      </div>

      {/* 4. Mobile Backdrop Overlay */}
      {(isLeftSidebarOpen || isRightSidebarOpen) && (
        <div 
          className="fixed inset-0 bg-black/60 backdrop-blur-xs z-30 md:hidden transition-all duration-200"
          onClick={() => {
            setIsLeftSidebarOpen(false);
            setIsRightSidebarOpen(false);
          }}
        />
      )}

      {/* 5. Retrieved Sources Modal Overlay */}
      <SourcesPanel
        isOpen={isSourcesOpen}
        onClose={() => setIsSourcesOpen(false)}
        sources={activeSources}
        activeDocCount={getUniqueDocCount()}
      />

      {/* 6. Configuration settings overlay */}
      <SettingsPanel
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        setSettings={setSettings}
        healthInfo={healthInfo}
      />
    </div>
  );
}
