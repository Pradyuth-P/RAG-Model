import React from 'react';
import { X, BookOpen, Percent, HelpCircle } from 'lucide-react';

export default function SourcesPanel({ isOpen, onClose, sources, activeDocCount }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in p-4">
      <div className="w-full max-w-2xl overflow-hidden rounded-2xl bg-slate-900 border border-slate-800 shadow-2xl flex flex-col max-h-[80vh] animate-scale-up">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-850 px-6 py-4 bg-slate-900">
          <div className="flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-indigo-400" />
            <div>
              <h3 className="text-sm font-semibold text-slate-100">Retrieved Context</h3>
              <span className="text-[10px] text-slate-500 font-medium mt-0.5 block">
                {sources.length} chunk{sources.length !== 1 ? 's' : ''} from {activeDocCount} document{activeDocCount !== 1 ? 's' : ''}
              </span>
            </div>
          </div>
          <button 
            onClick={onClose} 
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-850 hover:text-slate-200 transition"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Chunks List */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-slate-950/20">
          {sources.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-6">
              <HelpCircle className="h-8 w-8 text-slate-700 mb-2 stroke-[1.5]" />
              <p className="text-xs text-slate-400 font-medium">No sources retrieved yet</p>
              <p className="text-[10px] text-slate-500 max-w-[185px] mt-0.5 leading-relaxed">
                Ask a question to see the matching context retrieved from your document store.
              </p>
            </div>
          ) : (
            sources.map((src, idx) => (
              <div 
                key={idx}
                className="rounded-xl bg-slate-900/60 border border-slate-850 overflow-hidden shadow-sm hover:border-slate-800 transition"
              >
                {/* Header metadata */}
                <div className="bg-slate-950/50 px-3.5 py-2 border-b border-slate-850 flex items-center justify-between text-[10px]">
                  <div className="flex items-center gap-1.5 min-w-0 pr-2">
                    <span className="font-semibold text-slate-300 truncate" title={src.source}>
                      {src.source}
                    </span>
                    <span className="text-slate-500 bg-slate-850/60 px-1 py-0.2 rounded font-mono">
                      P. {src.page}
                    </span>
                  </div>
                  <div 
                    className={`flex items-center gap-0.5 px-2 py-0.5 rounded-full font-mono text-[9px] border ${
                      src.score >= 0.7 
                        ? 'bg-emerald-950/30 text-emerald-400 border-emerald-900/50' 
                        : 'bg-slate-850/30 text-slate-400 border-slate-800'
                    }`}
                    title={`Relevance score: ${src.score}`}
                  >
                    <span>Match: {Math.round(src.score * 100)}%</span>
                  </div>
                </div>

                {/* Text content */}
                <div className="p-3.5 text-slate-300 text-xs font-normal leading-relaxed whitespace-pre-wrap font-sans">
                  {src.content}
                </div>
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end border-t border-slate-850 px-6 py-4 bg-slate-900">
          <button
            onClick={onClose}
            className="rounded-xl bg-slate-800 hover:bg-slate-750 text-slate-300 px-4 py-2 text-xs font-semibold transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
