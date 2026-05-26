import React, { useRef, useState } from 'react';
import { UploadCloud, FileText, Trash2, ShieldAlert, Loader2, HardDrive } from 'lucide-react';
import { api } from '../services/api';

export default function UploadPanel({ 
  documents, 
  onRefreshDocs, 
  embeddingProvider,
  chunkSize,
  chunkOverlap,
  sessionId
}) {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadState, setUploadState] = useState({
    status: 'idle', // idle, uploading, indexing, success, error
    progress: 0,
    errorMsg: ''
  });

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      triggerUpload(files[0]);
    }
  };

  const handleFileSelect = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      triggerUpload(files[0]);
    }
  };

  const triggerUpload = async (file) => {
    // Validate extensions
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'docx', 'txt', 'md'].includes(ext)) {
      setUploadState({
        status: 'error',
        progress: 0,
        errorMsg: `Unsupported extension: .${ext}. Please upload a PDF, DOCX, TXT, or MD file.`
      });
      return;
    }

    setUploadState({ status: 'uploading', progress: 0, errorMsg: '' });

    try {
      await api.uploadDocument(
        file, 
        embeddingProvider, 
        chunkSize, 
        chunkOverlap,
        sessionId,
        (progressEvent) => {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadState(prev => ({ 
            ...prev, 
            progress: percent,
            status: percent === 100 ? 'indexing' : 'uploading' 
          }));
        }
      );

      setUploadState({ status: 'success', progress: 100, errorMsg: '' });
      onRefreshDocs();
      
      // Reset upload state after 3s
      setTimeout(() => {
        setUploadState(prev => prev.status === 'success' ? { status: 'idle', progress: 0, errorMsg: '' } : prev);
      }, 3000);

    } catch (err) {
      console.error(err);
      setUploadState({
        status: 'error',
        progress: 0,
        errorMsg: err.response?.data?.detail || err.message || 'File upload and chunking failed.'
      });
    }
  };

  const handleDelete = async (docId) => {
    if (!confirm('Are you sure you want to delete this document from the vector store?')) return;
    try {
      await api.deleteDocument(docId, embeddingProvider, sessionId);
      onRefreshDocs();
    } catch (err) {
      alert(`Delete failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getProviderTagClass = (provider) => {
    switch (provider) {
      case 'openai': return 'bg-emerald-950/40 text-emerald-400 border-emerald-850';
      case 'gemini': return 'bg-cyan-950/40 text-cyan-400 border-cyan-850';
      default: return 'bg-indigo-950/40 text-indigo-400 border-indigo-850';
    }
  };

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Upload Box */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative flex flex-col items-center justify-center border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all duration-200 ${
          isDragOver 
            ? 'border-indigo-500 bg-indigo-950/20' 
            : 'border-slate-800 hover:border-slate-700 bg-slate-900/40 hover:bg-slate-900/60'
        }`}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          className="hidden"
          accept=".pdf,.docx,.txt,.md"
        />

        <UploadCloud className="h-10 w-10 text-slate-400 mb-2.5 transition-transform duration-200 group-hover:scale-105" />
        
        <h4 className="text-sm font-semibold text-slate-200">
          Upload Knowledge Documents
        </h4>
        <p className="text-xs text-slate-500 mt-1 max-w-[200px] leading-relaxed">
          Drag & drop PDF, DOCX, TXT, or MD here or browse files.
        </p>

        {uploadState.status !== 'idle' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center rounded-2xl bg-slate-950/90 p-4 z-10 transition-all duration-200">
            {uploadState.status === 'uploading' && (
              <div className="w-full px-6 text-center space-y-2">
                <Loader2 className="h-7 w-7 text-indigo-500 animate-spin mx-auto" />
                <p className="text-xs text-slate-300">Uploading File... {uploadState.progress}%</p>
                <div className="w-full bg-slate-800 rounded-full h-1">
                  <div className="bg-indigo-500 h-1 rounded-full transition-all duration-150" style={{ width: `${uploadState.progress}%` }}></div>
                </div>
              </div>
            )}
            
            {uploadState.status === 'indexing' && (
              <div className="text-center space-y-2 animate-pulse-load">
                <Loader2 className="h-7 w-7 text-indigo-400 animate-spin mx-auto" />
                <p className="text-xs text-slate-200 font-medium">Extracting & Chunking Document...</p>
                <p className="text-[10px] text-slate-400">Embedding vectors & updating FAISS index</p>
              </div>
            )}

            {uploadState.status === 'success' && (
              <div className="text-center space-y-1.5">
                <div className="mx-auto flex h-8 w-8 items-center justify-center rounded-full bg-emerald-950 text-emerald-400 border border-emerald-900">
                  ✓
                </div>
                <p className="text-xs font-semibold text-slate-200">Ingestion Complete</p>
                <p className="text-[10px] text-slate-400">Document split and indexed successfully.</p>
              </div>
            )}

            {uploadState.status === 'error' && (
              <div className="text-center p-3 space-y-2">
                <ShieldAlert className="h-7 w-7 text-rose-500 mx-auto" />
                <p className="text-xs font-semibold text-rose-400 leading-snug">Upload Failed</p>
                <p className="text-[10px] text-slate-400 max-h-[60px] overflow-y-auto leading-relaxed">
                  {uploadState.errorMsg}
                </p>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setUploadState({ status: 'idle', progress: 0, errorMsg: '' });
                  }}
                  className="rounded-lg bg-slate-800 px-3 py-1.5 text-[10px] font-semibold text-slate-250 hover:bg-slate-700 transition"
                >
                  Dismiss
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Documents List */}
      <div className="flex-1 flex flex-col min-h-0 bg-slate-900/25 border border-slate-900 rounded-2xl p-4 overflow-hidden">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-3">
          <HardDrive className="h-3.5 w-3.5" />
          <span>Knowledge Base Documents ({documents.length})</span>
        </div>

        <div className="flex-1 overflow-y-auto pr-1 space-y-2">
          {documents.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center p-4">
              <FileText className="h-8 w-8 text-slate-650 stroke-[1.5] mb-2" />
              <p className="text-xs text-slate-400 font-medium">No documents indexed</p>
              <p className="text-[10px] text-slate-500 max-w-[170px] mt-0.5 leading-relaxed">
                Add PDF/TXT files to query them. Linked with embedding provider: <span className="font-mono text-indigo-400">{embeddingProvider}</span>.
              </p>
            </div>
          ) : (
            documents.map((doc) => (
              <div 
                key={doc.id} 
                className="group flex items-center justify-between p-2.5 rounded-xl bg-slate-900/60 border border-slate-850 hover:border-slate-800 hover:bg-slate-900 transition duration-150"
              >
                <div className="flex items-start gap-2.5 min-w-0 pr-2">
                  <div className="flex-shrink-0 mt-0.5 rounded-lg bg-indigo-950/50 p-1.5 text-indigo-400 border border-indigo-900/40">
                    <FileText className="h-4 w-4" />
                  </div>
                  <div className="min-w-0">
                    <h5 className="text-xs font-medium text-slate-200 truncate group-hover:text-slate-100" title={doc.filename}>
                      {doc.filename}
                    </h5>
                    <div className="flex items-center gap-1.5 text-[10px] text-slate-500 mt-0.5 flex-wrap">
                      <span>{formatBytes(doc.file_size)}</span>
                      <span>•</span>
                      <span>{doc.chunk_count} chunks</span>
                      <span>•</span>
                      <span className={`px-1 py-0.2 rounded border font-mono text-[9px] ${getProviderTagClass(doc.embedding_provider)}`}>
                        {doc.embedding_provider}
                      </span>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => handleDelete(doc.id)}
                  className="p-1.5 rounded-lg text-slate-500 hover:bg-rose-950/30 hover:text-rose-400 transition"
                  title="Delete Document"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
