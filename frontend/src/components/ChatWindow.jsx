import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, Sparkles, AlertTriangle, BookOpen } from 'lucide-react';

export default function ChatWindow({ 
  messages, 
  onSendMessage, 
  isLoading, 
  onSelectSources,
  hasDocuments
}) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    onSendMessage(input.trim());
    setInput('');
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 overflow-hidden relative">
      {/* Messages List Area */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center max-w-xl mx-auto space-y-6">
            <div className="rounded-2xl bg-indigo-950/20 border border-indigo-900/30 p-4">
              <Sparkles className="h-10 w-10 text-indigo-400 mx-auto animate-pulse" />
            </div>
            
            <div className="space-y-2">
              <h2 className="text-xl font-bold text-slate-100 font-display">AetherRAG Assistant</h2>
              <p className="text-xs text-slate-400 leading-relaxed">
                Welcome to your modular RAG interface. Ask questions grounded specifically in the documents uploaded in your knowledge base.
              </p>
            </div>

            {!hasDocuments && (
              <div className="flex items-center gap-2.5 rounded-xl border border-amber-900/50 bg-amber-950/20 p-4 text-left">
                <AlertTriangle className="h-5 w-5 text-amber-400 flex-shrink-0" />
                <div>
                  <h5 className="text-xs font-semibold text-slate-200">No Documents Uploaded</h5>
                  <p className="text-[10px] text-slate-400 mt-0.5 leading-relaxed">
                    Please upload one or more documents (PDF, DOCX, TXT, or MD) on the right panel first to load knowledge vectors.
                  </p>
                </div>
              </div>
            )}

            {hasDocuments && (
              <div className="grid grid-cols-2 gap-3 w-full">
                {[
                  "What is the main topic discussed in this document?",
                  "Provide a comprehensive summary of key takeaways.",
                  "Are there any specific dates or figures mentioned?",
                  "Extract any action items or recommendations."
                ].map((prompt, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setInput(prompt);
                      onSendMessage(prompt);
                    }}
                    className="p-3 text-left rounded-xl bg-slate-900/50 border border-slate-850 hover:bg-slate-900 hover:border-slate-800 text-xs text-slate-350 hover:text-slate-250 transition duration-150"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          messages.map((msg, idx) => {
            const isUser = msg.role === 'user';
            return (
              <div 
                key={idx} 
                className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} animate-fade-in`}
              >
                <div 
                  className={`max-w-[80%] rounded-2xl p-4 text-sm leading-relaxed ${
                    isUser 
                      ? 'bg-indigo-600 text-white rounded-br-none shadow-md shadow-indigo-600/10' 
                      : 'bg-slate-900 border border-slate-850 text-slate-250 rounded-bl-none'
                  }`}
                >
                  {isUser ? (
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  ) : (
                    <div className="space-y-3">
                      {/* Markdown Renders Content */}
                      <div className="prose prose-invert max-w-none text-slate-250 text-sm space-y-2">
                        <ReactMarkdown
                          components={{
                            p: ({node, ...props}) => <p className="mb-2 last:mb-0 leading-relaxed" {...props} />,
                            ul: ({node, ...props}) => <ul className="list-disc pl-5 mb-2 space-y-1" {...props} />,
                            ol: ({node, ...props}) => <ol className="list-decimal pl-5 mb-2 space-y-1" {...props} />,
                            li: ({node, ...props}) => <li className="mb-0.5" {...props} />,
                            strong: ({node, ...props}) => <strong className="font-bold text-slate-100" {...props} />,
                            code: ({node, inline, className, children, ...props}) => (
                              <code 
                                className={`${
                                  inline 
                                    ? 'bg-slate-800 px-1.5 py-0.5 rounded font-mono text-xs text-indigo-300' 
                                    : 'block bg-slate-950 p-3 rounded-lg font-mono text-xs overflow-x-auto text-indigo-400 border border-slate-850'
                                }`} 
                                {...props}
                              >
                                {children}
                              </code>
                            )
                          }}
                        >
                          {msg.content}
                        </ReactMarkdown>
                      </div>

                      {/* Display referenced sources if available */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="flex items-center gap-1.5 pt-2 border-t border-slate-850">
                          <button
                            onClick={() => onSelectSources(msg.sources)}
                            className="flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300 font-medium transition"
                          >
                            <BookOpen className="h-3.5 w-3.5" />
                            <span>View {msg.sources.length} matching sources</span>
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}

        {/* Streaming Loading indicator */}
        {isLoading && messages.length > 0 && messages[messages.length - 1].role === 'user' && (
          <div className="flex justify-start">
            <div className="rounded-2xl p-4 bg-slate-900 border border-slate-850 rounded-bl-none flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-indigo-500 typing-dot"></span>
              <span className="h-2 w-2 rounded-full bg-indigo-500 typing-dot"></span>
              <span className="h-2 w-2 rounded-full bg-indigo-500 typing-dot"></span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Form Bar */}
      <div className="p-4 bg-slate-950 border-t border-slate-905">
        <form onSubmit={handleSubmit} className="flex gap-2 max-w-4xl mx-auto">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading || !hasDocuments}
            placeholder={
              !hasDocuments 
                ? "Upload documents first to start chatting..." 
                : "Ask anything about the knowledge base..."
            }
            className="flex-1 rounded-xl bg-slate-900 border border-slate-850 px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-slate-700 focus:ring-1 focus:ring-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim() || !hasDocuments}
            className="rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-3 text-sm font-semibold transition active:scale-[0.97] disabled:opacity-55 disabled:cursor-not-allowed shadow-lg shadow-indigo-600/10 hover:shadow-indigo-600/20"
          >
            <Send className="h-4 w-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
