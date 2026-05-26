import React from 'react';
import { X, Shield, Settings, Sliders, Layers } from 'lucide-react';

export default function SettingsPanel({ 
  isOpen, 
  onClose, 
  settings, 
  setSettings, 
  healthInfo 
}) {
  if (!isOpen) return null;

  const handleProviderChange = (e) => {
    const prov = e.target.value;
    const defaultModel = healthInfo?.available_providers?.[prov]?.default_model || '';
    setSettings(prev => ({
      ...prev,
      llmProvider: prov,
      llmModel: defaultModel
    }));
  };

  const handleModelChange = (e) => {
    setSettings(prev => ({ ...prev, llmModel: e.target.value }));
  };

  const activeProviderInfo = healthInfo?.available_providers?.[settings.llmProvider];
  const modelsMap = activeProviderInfo?.models || {};

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in">
      <div className="w-full max-w-lg overflow-hidden rounded-2xl bg-slate-900 border border-slate-800 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-850 px-6 py-4">
          <div className="flex items-center gap-2">
            <Settings className="h-5 w-5 text-indigo-400" />
            <h3 className="text-lg font-semibold text-slate-100">Configuration Panel</h3>
          </div>
          <button 
            onClick={onClose} 
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-850 hover:text-slate-200 transition"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6 max-h-[80vh] overflow-y-auto">
          {/* Provider Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-300 flex items-center gap-1.5">
              <Shield className="h-4 w-4 text-indigo-400" />
              LLM Provider
            </label>
            <div className="w-full rounded-xl bg-slate-850 border border-slate-800 px-3.5 py-2.5 text-sm text-slate-300 select-none">
              Groq (Active)
            </div>
            {activeProviderInfo && !activeProviderInfo.available && (
              <p className="text-xs text-amber-400 mt-1">
                Warning: The GROQ_API_KEY is not set in the backend .env!
              </p>
            )}
          </div>

          {/* Model Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-300">Model Choice</label>
            <select
              value={settings.llmModel}
              onChange={handleModelChange}
              className="w-full rounded-xl bg-slate-800 border border-slate-700 px-3.5 py-2.5 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {Object.entries(modelsMap).map(([id, label]) => (
                <option key={id} value={id}>{label}</option>
              ))}
            </select>
          </div>

          {/* Temperature */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <label className="font-medium text-slate-300 flex items-center gap-1.5">
                <Sliders className="h-4 w-4 text-indigo-400" />
                Temperature
              </label>
              <span className="text-slate-400 font-mono text-xs">{settings.temperature}</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.2"
              step="0.1"
              value={settings.temperature}
              onChange={(e) => setSettings(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>Precise / Grounded (0.0)</span>
              <span>Creative / Loose (1.2)</span>
            </div>
          </div>

          <hr className="border-slate-850" />

          {/* Ingestion & Embeddings */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-indigo-400 flex items-center gap-1.5">
              <Layers className="h-4 w-4" />
              Embedding & Indexing Settings
            </h4>

            {/* Embedding Provider Selection */}
            <div className="space-y-2">
              <label className="text-xs font-medium text-slate-400">Embedding Vector Model</label>
              <div className="w-full rounded-xl bg-slate-855 border border-slate-800 px-3.5 py-2.5 text-sm text-slate-400 select-none">
                HuggingFace Local Embeddings (Free, Local CPU)
              </div>
            </div>

            {/* Chunk Configuration */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <label className="text-xs font-medium text-slate-400">Chunk Size</label>
                <input
                  type="number"
                  value={settings.chunkSize}
                  onChange={(e) => setSettings(prev => ({ ...prev, chunkSize: parseInt(e.target.value) || 1000 }))}
                  className="w-full rounded-xl bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium text-slate-400">Chunk Overlap</label>
                <input
                  type="number"
                  value={settings.chunkOverlap}
                  onChange={(e) => setSettings(prev => ({ ...prev, chunkOverlap: parseInt(e.target.value) || 200 }))}
                  className="w-full rounded-xl bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            </div>
            <p className="text-[11px] text-slate-450 italic leading-relaxed">
              * Note: Chunk settings apply to documents uploaded *after* modification. Already indexed documents will remain unaffected.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 border-t border-slate-850 px-6 py-4 bg-slate-900">
          <button
            onClick={onClose}
            className="w-full rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-indigo-500 transition shadow-lg shadow-indigo-600/20"
          >
            Apply Changes
          </button>
        </div>
      </div>
    </div>
  );
}
