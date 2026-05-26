import React from 'react';
import { Plus, MessageSquare, Trash2, BrainCircuit, Settings, RefreshCw } from 'lucide-react';

export default function HistorySidebar({ 
  sessions, 
  currentSession, 
  onSelectSession, 
  onNewSession, 
  onClearHistory, 
  activeSettings,
  onOpenSettings,
  healthInfo
}) {
  const getProviderName = (prov) => {
    return healthInfo?.available_providers?.[prov]?.name || prov.toUpperCase();
  };

  return (
    <div className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col h-full overflow-hidden">
      {/* Brand Header */}
      <div className="px-5 py-5 flex items-center gap-2 border-b border-slate-850">
        <BrainCircuit className="h-6 w-6 text-indigo-500 flex-shrink-0" />
        <div>
          <h1 className="text-sm font-bold text-slate-100 font-display tracking-tight leading-normal">AetherRAG</h1>
          <span className="text-[10px] text-slate-500 font-medium mt-0.5 block">Modular Retrieval Engine</span>
        </div>
      </div>

      {/* Primary Actions */}
      <div className="p-3.5 space-y-2">
        <button
          onClick={onNewSession}
          className="w-full flex items-center justify-center gap-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-slate-100 px-4 py-2.5 text-xs font-semibold shadow-lg shadow-indigo-600/10 hover:shadow-indigo-600/20 active:scale-[0.98] transition-all"
        >
          <Plus className="h-4 w-4" />
          New Chat Session
        </button>
      </div>

      {/* Sessions Scroll List */}
      <div className="flex-1 overflow-y-auto px-2 space-y-1">
        <div className="px-3 py-1.5 text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
          Recent Conversations
        </div>
        {sessions.length === 0 ? (
          <div className="p-3 text-[11px] text-slate-500 text-center italic">
            No active sessions
          </div>
        ) : (
          sessions.map((session) => (
            <button
              key={session.id}
              onClick={() => onSelectSession(session.id)}
              className={`w-full flex items-center gap-2 px-3 py-2 rounded-xl text-left text-xs transition-colors duration-150 ${
                currentSession === session.id 
                  ? 'bg-slate-800/70 text-indigo-400 font-medium' 
                  : 'text-slate-400 hover:bg-slate-850 hover:text-slate-200'
              }`}
            >
              <MessageSquare className="h-3.5 w-3.5 flex-shrink-0" />
              <span className="truncate flex-1">{session.title}</span>
            </button>
          ))
        )}
      </div>

      {/* Active Configurations Settings Option */}
      <div className="p-4 bg-slate-950/40 border-t border-slate-850 space-y-3">
        <button
          onClick={onOpenSettings}
          className="w-full flex items-center justify-center gap-2 rounded-xl border border-slate-800 bg-slate-900/50 hover:bg-slate-900 text-slate-350 hover:text-slate-100 py-2.5 text-xs font-semibold transition"
        >
          <Settings className="h-4 w-4 text-indigo-400" />
          Settings Configuration
        </button>

        {/* Clear database */}
        <button
          onClick={onClearHistory}
          className="w-full flex items-center justify-center gap-1.5 rounded-xl border border-slate-850 text-slate-450 hover:border-slate-800 hover:bg-rose-950/20 hover:text-rose-455 py-2 text-[11px] font-semibold transition"
        >
          <Trash2 className="h-3.5 w-3.5 text-slate-500" />
          Clear Conversation DB
        </button>
      </div>
    </div>
  );
}
