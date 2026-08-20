import React, { useState, useEffect, useRef } from 'react';
import { 
  Search, AlertTriangle, ShieldAlert, 
  Copy, Code2, Layers, Check, Sparkles,
  Activity, Award, Database, FileCheck, Bookmark, ExternalLink, 
  PanelRightClose, PanelRightOpen, XCircle, Loader2, WifiOff
} from 'lucide-react';
import { checkHealth, askQuestion, normalizeAskResponse } from '../services/api';

export default function ClinicalDashboard() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('chunks'); // 'chunks' | 'json'
  const [selectedChunkId, setSelectedChunkId] = useState(null);
  const [copied, setCopied] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false); // Controls right sidebar visibility
  const [isHealthy, setIsHealthy] = useState(null); // true | false | null
  const [healthDetails, setHealthDetails] = useState(null);
  const [error, setError] = useState(null);
  const [validationError, setValidationError] = useState(null);
  const [response, setResponse] = useState(null);
  const isMountedRef = useRef(true);

  // Poll backend health on mount and periodically every 20 seconds
  const pollHealth = async () => {
    try {
      const data = await checkHealth();
      if (isMountedRef.current) {
        setIsHealthy(true);
        setHealthDetails(data);
      }
    } catch {
      if (isMountedRef.current) {
        setIsHealthy(false);
      }
    }
  };

  useEffect(() => {
    isMountedRef.current = true;
    pollHealth();
    const interval = setInterval(pollHealth, 20000);
    return () => {
      isMountedRef.current = false;
      clearInterval(interval);
    };
  }, []);

  const quickPrompts = [
    "What are the recommendations for lung cancer screening?",
    "What are the treatment options for stage III non-small cell lung cancer?",
    "What is the recommended treatment for diabetes?",
    "My patient has these symptoms. What diagnosis should I give?"
  ];

  const handleCopy = () => {
    if (!response) return;
    const evidenceText = response.supporting_evidence?.length > 0
      ? `\n\nSUPPORTING EVIDENCE:\n${response.supporting_evidence.map(e => `- ${e.text}`).join('\n')}`
      : '';
    const textToCopy = `STATUS: ${response.status}\nCONFIDENCE: ${response.confidence}\n\nRECOMMENDATION:\n${response.recommendation}${evidenceText}\n\nSAFETY NOTE:\n${response.safety_note}`;
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSearch = async (searchQuery) => {
    const trimmed = (searchQuery || '').trim();
    if (!trimmed) {
      setValidationError('Please enter a clinical question before running search.');
      return;
    }
    setValidationError(null);
    setError(null);
    setLoading(true);
    setQuery(trimmed);

    try {
      const rawData = await askQuestion(trimmed, 5);
      const normalized = normalizeAskResponse(rawData);
      setResponse(normalized);
      // Auto-open sidebar if evidence chunks are retrieved
      if (normalized?.evidence_chunks?.length > 0) {
        setShowSidebar(true);
      }
    } catch (err) {
      setError(err.message || 'An error occurred while communicating with the Clinical RAG backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 font-sans text-slate-100 overflow-hidden">
      
      {/* LEFT MAIN PANEL */}
      <div className="flex-1 flex flex-col h-full border-r border-slate-800 bg-slate-900/50 backdrop-blur-md">
        
        {/* Top Header */}
        <header className="px-6 py-4 border-b border-slate-800/80 bg-gradient-to-r from-slate-950 via-slate-900 to-indigo-950/60 flex justify-between items-center shadow-lg">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-gradient-to-tr from-blue-600 to-indigo-500 rounded-xl shadow-lg shadow-blue-500/20">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-white via-slate-200 to-indigo-200">
                Lung Cancer Clinical Decision Support RAG System
              </h1>
              <p className="text-xs text-indigo-300/70 font-medium">Evidence-Grounded Non-Small Cell Lung Cancer Guidelines (NCI PDQ®)</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            {/* Sidebar Toggle Button */}
            <button
              onClick={() => setShowSidebar(!showSidebar)}
              className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-2 border transition shadow-sm ${
                showSidebar 
                  ? 'bg-indigo-600 text-white border-indigo-500 shadow-indigo-600/30' 
                  : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700 hover:text-white'
              }`}
            >
              {showSidebar ? <PanelRightClose className="w-4 h-4" /> : <PanelRightOpen className="w-4 h-4" />}
              <span>{showSidebar ? 'Hide Evidence Chunks' : 'View Evidence Chunks'}</span>
            </button>

            {/* Health Status Indicator */}
            {isHealthy === true && (
              <span className="px-3.5 py-1.5 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold rounded-full flex items-center gap-2 shadow-inner" title={healthDetails ? `Indexed Chunks: ${healthDetails.indexed_chunks} | ${healthDetails.llm}` : 'Pipeline Ready'}>
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Pipeline Live
              </span>
            )}
            {isHealthy === false && (
              <span className="px-3.5 py-1.5 bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs font-semibold rounded-full flex items-center gap-2 shadow-inner" title="FastAPI backend is offline or unreachable at port 8000">
                <WifiOff className="w-3.5 h-3.5 text-rose-400" /> Backend Offline
              </span>
            )}
            {isHealthy === null && (
              <span className="px-3.5 py-1.5 bg-slate-800 border border-slate-700 text-slate-400 text-xs font-semibold rounded-full flex items-center gap-2 shadow-inner">
                <span className="w-2 h-2 rounded-full bg-slate-500"></span> Connecting...
              </span>
            )}
          </div>
        </header>

        {/* Scrollable Query & Answer Workspace */}
        <main className="flex-1 overflow-y-auto p-6 space-y-6">
          
          {/* Query Bar */}
          <div className="p-5 bg-slate-900/80 border border-slate-800/80 rounded-2xl shadow-xl space-y-3">
            <label className="text-xs font-bold uppercase tracking-wider text-indigo-400 flex items-center gap-2">
              <Search className="w-3.5 h-3.5" /> Clinical Query Input
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  if (validationError) setValidationError(null);
                }}
                onKeyDown={(e) => e.key === 'Enter' && !loading && handleSearch(query)}
                placeholder="Ask about lung cancer screening criteria, staging treatments, guidelines..."
                className="flex-1 px-4 py-3 bg-slate-950 border border-slate-700/80 rounded-xl text-white placeholder-slate-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition text-sm shadow-inner"
              />
              <button
                onClick={() => handleSearch(query)}
                disabled={loading}
                className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold px-6 py-3 rounded-xl shadow-lg shadow-blue-600/25 transition disabled:opacity-50 flex items-center gap-2 text-sm cursor-pointer disabled:cursor-not-allowed"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin text-white" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  'Run Query'
                )}
              </button>
            </div>

            {/* Validation Message */}
            {validationError && (
              <p className="text-xs text-amber-400 flex items-center gap-1.5 font-medium">
                <AlertTriangle className="w-3.5 h-3.5" /> {validationError}
              </p>
            )}

            {/* Quick Prompts */}
            <div className="flex flex-wrap gap-2 pt-1">
              {quickPrompts.map((prompt, i) => (
                <button
                  key={i}
                  onClick={() => handleSearch(prompt)}
                  disabled={loading}
                  className="text-xs bg-slate-800/60 hover:bg-slate-800 text-slate-300 hover:text-white px-3 py-1.5 rounded-lg transition border border-slate-700/50 flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                >
                  <Sparkles className="w-3 h-3 text-indigo-400" />
                  {prompt}
                </button>
              ))}
            </div>
          </div>

          {/* Backend Error Banner */}
          {error && (
            <div className="p-4 bg-rose-950/40 border border-rose-800/60 rounded-2xl text-rose-200 flex items-start gap-3 shadow-xl">
              <XCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
              <div className="space-y-1">
                <h4 className="text-sm font-semibold text-rose-300">Pipeline Request Error</h4>
                <p className="text-xs text-rose-200/90 leading-relaxed">{error}</p>
              </div>
            </div>
          )}

          {/* Response Output */}
          {response && !loading && (
            <div className="space-y-5">
              
              {/* Badges Bar */}
              <div className="grid grid-cols-4 gap-3">
                {/* Status Badge */}
                <div className={`border p-3.5 rounded-2xl text-center relative overflow-hidden transition ${
                  response.status === 'safety_refusal' 
                    ? 'bg-rose-950/40 border-rose-800/80 text-rose-300'
                    : response.status === 'insufficient_evidence'
                    ? 'bg-amber-950/40 border-amber-800/80 text-amber-300'
                    : 'bg-slate-900/80 border-slate-800 text-blue-400'
                }`}>
                  <div className={`absolute top-0 left-0 w-full h-1 ${
                    response.status === 'safety_refusal' ? 'bg-rose-500' : response.status === 'insufficient_evidence' ? 'bg-amber-500' : 'bg-blue-500'
                  }`}></div>
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Status</span>
                  <span className="text-xs font-bold uppercase tracking-wide">{response.status}</span>
                </div>

                {/* Confidence Badge */}
                <div className="bg-slate-900/80 border border-slate-800 p-3.5 rounded-2xl text-center relative overflow-hidden group hover:border-emerald-500/50 transition">
                  <div className={`absolute top-0 left-0 w-full h-1 ${
                    response.confidence === 'High' ? 'bg-emerald-500' : response.confidence === 'Medium' ? 'bg-blue-500' : 'bg-slate-500'
                  }`}></div>
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Confidence</span>
                  <span className={`text-xs font-bold ${
                    response.confidence === 'High' ? 'text-emerald-400' : response.confidence === 'Medium' ? 'text-blue-400' : 'text-slate-400'
                  }`}>{response.confidence}</span>
                </div>

                {/* Schema Valid Badge */}
                <div className="bg-slate-900/80 border border-slate-800 p-3.5 rounded-2xl text-center relative overflow-hidden group hover:border-indigo-500/50 transition">
                  <div className="absolute top-0 left-0 w-full h-1 bg-indigo-500"></div>
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Schema Valid</span>
                  <span className="text-xs font-bold text-indigo-300">{response.schema_valid ? 'True' : 'False'}</span>
                </div>

                {/* Citation Validation Badge */}
                <div className="bg-slate-900/80 border border-slate-800 p-3.5 rounded-2xl text-center relative overflow-hidden group hover:border-purple-500/50 transition">
                  <div className="absolute top-0 left-0 w-full h-1 bg-purple-500"></div>
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">Citation Audit</span>
                  <span className="text-[11px] font-mono text-purple-300 font-medium">
                    valid={response.citation_validation?.valid ? 'True' : 'False'}
                  </span>
                </div>
              </div>

              {/* Status 1: Safety Refusal Banner */}
              {response.status === 'safety_refusal' && (
                <div className="bg-gradient-to-r from-rose-950/80 via-slate-950 to-rose-950/80 border border-rose-600/70 rounded-2xl p-6 shadow-2xl space-y-4">
                  <div className="flex items-center gap-3 border-b border-rose-800/60 pb-3">
                    <ShieldAlert className="w-6 h-6 text-rose-400 flex-shrink-0 animate-pulse" />
                    <div>
                      <h3 className="text-sm font-bold uppercase tracking-wider text-rose-300">
                        Clinical Safety Guardrail Refusal
                      </h3>
                      <p className="text-xs text-rose-200/70">Patient-specific requests are blocked per clinical protocol</p>
                    </div>
                  </div>
                  <div className="p-4 bg-rose-950/40 rounded-xl border border-rose-900/50 text-sm text-rose-100 leading-relaxed font-medium">
                    {response.recommendation}
                  </div>
                  {response.missing_information?.length > 0 && (
                    <div className="space-y-1.5 text-xs text-rose-300/80">
                      <span className="font-semibold text-rose-300">Refusal Details:</span>
                      <ul className="list-disc list-inside space-y-1">
                        {response.missing_information.map((item, i) => (
                          <li key={i}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* Status 2: Insufficient Evidence Banner */}
              {response.status === 'insufficient_evidence' && (
                <div className="bg-gradient-to-r from-amber-950/70 via-slate-950 to-amber-950/70 border border-amber-600/60 rounded-2xl p-6 shadow-2xl space-y-4">
                  <div className="flex items-center gap-3 border-b border-amber-800/60 pb-3">
                    <AlertTriangle className="w-6 h-6 text-amber-400 flex-shrink-0" />
                    <div>
                      <h3 className="text-sm font-bold uppercase tracking-wider text-amber-300">
                        Insufficient Guideline Evidence
                      </h3>
                      <p className="text-xs text-amber-200/70">Query is out of scope or evidence threshold was not met</p>
                    </div>
                  </div>
                  <div className="p-4 bg-amber-950/30 rounded-xl border border-amber-900/50 text-sm text-amber-100 leading-relaxed font-medium">
                    {response.recommendation}
                  </div>
                  {response.missing_information?.length > 0 && (
                    <div className="space-y-1.5 text-xs text-amber-300/80">
                      <span className="font-semibold text-amber-300">Missing Information & Guidelines Context:</span>
                      <ul className="list-disc list-inside space-y-1">
                        {response.missing_information.map((item, i) => (
                          <li key={i}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* Status 3: Answered (Standard Card) */}
              {response.status === 'answered' && (
                <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-2xl space-y-6 relative">
                  
                  {/* Header Toolbar */}
                  <div className="flex justify-between items-center border-b border-slate-800 pb-4">
                    <span className="text-xs font-bold tracking-widest text-indigo-400 uppercase flex items-center gap-2">
                      <Award className="w-4 h-4 text-indigo-400" /> Grounded Clinical Output
                    </span>
                    <div className="flex items-center gap-2">
                      <button 
                        onClick={handleCopy}
                        className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 px-3 py-1.5 rounded-xl flex items-center gap-1.5 transition font-medium cursor-pointer"
                      >
                        {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                        {copied ? 'Copied' : 'Copy Response'}
                      </button>
                    </div>
                  </div>

                  {/* Recommendation */}
                  <div>
                    <h3 className="text-xs font-bold tracking-wider text-slate-400 uppercase mb-2.5 flex items-center gap-1.5">
                      <FileCheck className="w-4 h-4 text-blue-400" /> Clinical Recommendation
                    </h3>
                    <div className="bg-gradient-to-r from-blue-950/40 to-slate-950 p-4 rounded-xl border border-blue-900/40 text-sm font-medium text-slate-100 leading-relaxed shadow-inner">
                      {response.recommendation}
                    </div>
                  </div>

                  {/* Supporting Evidence */}
                  {response.supporting_evidence?.length > 0 && (
                    <div>
                      <h3 className="text-xs font-bold tracking-wider text-slate-400 uppercase mb-2.5 flex items-center gap-1.5">
                        <Bookmark className="w-4 h-4 text-indigo-400" /> Supporting Evidence Claims
                      </h3>
                      <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 text-sm text-slate-300 space-y-3.5">
                        {response.supporting_evidence.map((evidence, idx) => (
                          <div key={idx} className="flex gap-3 items-start">
                            <span className="font-bold text-indigo-400 bg-indigo-950/80 px-2 py-0.5 rounded-md text-xs border border-indigo-800/50">
                              {idx + 1}
                            </span>
                            <div className="space-y-2 flex-1">
                              <p className="leading-relaxed text-slate-200">{evidence.text}</p>
                              {evidence.citations?.length > 0 && (
                                <div className="flex flex-wrap gap-2 pt-1">
                                  {evidence.citations.map((citeTag, cIdx) => {
                                    // Extract chunk id or match tag
                                    const isSelected = selectedChunkId && citeTag.includes(selectedChunkId);
                                    return (
                                      <button
                                        key={cIdx}
                                        onClick={() => {
                                          const chunkMatch = response.evidence_chunks?.find(c => citeTag.includes(c.chunk_id));
                                          if (chunkMatch) {
                                            setSelectedChunkId(chunkMatch.chunk_id);
                                          }
                                          setShowSidebar(true);
                                          setActiveTab('chunks');
                                        }}
                                        className={`text-[11px] font-mono px-2.5 py-1 rounded-lg border transition flex items-center gap-1.5 cursor-pointer ${
                                          isSelected 
                                            ? 'bg-blue-600 text-white border-blue-500 shadow-md shadow-blue-600/30' 
                                            : 'bg-blue-950/50 text-blue-300 border-blue-800/60 hover:bg-blue-900/50'
                                        }`}
                                      >
                                        <ExternalLink className="w-3 h-3" />
                                        <span>cited:</span>
                                        <span className="font-semibold underline">{citeTag}</span>
                                      </button>
                                    );
                                  })}
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                </div>
              )}

            </div>
          )}

          {/* Initial State / Placeholder when no query has been run */}
          {!response && !loading && !error && (
            <div className="p-12 text-center border border-dashed border-slate-800 rounded-2xl bg-slate-900/20 text-slate-400 space-y-3">
              <Database className="w-8 h-8 text-indigo-400 mx-auto opacity-70" />
              <h3 className="text-sm font-semibold text-slate-300">Ready for Clinical Queries</h3>
              <p className="text-xs text-slate-500 max-w-md mx-auto">
                Submit a question or select a quick prompt above to query the evidence-grounded non-small cell lung cancer clinical decision support system.
              </p>
            </div>
          )}
        </main>

        {/* Safety Banner */}
        <footer className="p-3.5 bg-gradient-to-r from-amber-950/60 via-slate-950 to-amber-950/60 border-t border-amber-900/40 text-amber-200 text-xs flex items-center justify-center gap-2 font-medium">
          <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
          <span>{response?.safety_note || "Educational information only; not a diagnosis or medical advice."}</span>
        </footer>
      </div>

      {/* RIGHT SIDEBAR: Evidence Chunks & Raw JSON */}
      {showSidebar && (
        <div className="w-96 bg-slate-950 border-l border-slate-800 flex flex-col h-full transition-all duration-300 ease-in-out">
          
          {/* Navigation Tabs Header */}
          <div className="p-3 border-b border-slate-800 bg-slate-900/80 flex gap-2">
            <button
              onClick={() => setActiveTab('chunks')}
              className={`flex-1 py-2 px-3 rounded-xl text-xs font-semibold flex items-center justify-center gap-2 transition cursor-pointer ${
                activeTab === 'chunks' 
                  ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-600/20' 
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
            >
              <Layers className="w-3.5 h-3.5" /> Chunks ({response?.evidence_chunks?.length || 0})
            </button>
            <button
              onClick={() => setActiveTab('json')}
              className={`flex-1 py-2 px-3 rounded-xl text-xs font-semibold flex items-center justify-center gap-2 transition cursor-pointer ${
                activeTab === 'json' 
                  ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-600/20' 
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
            >
              <Code2 className="w-3.5 h-3.5" /> Raw JSON
            </button>
          </div>

          {/* Tab 1: Top-5 Retrieved Chunks List */}
          {activeTab === 'chunks' && (
            <div className="flex-1 overflow-y-auto p-4 space-y-3.5">
              {(!response?.evidence_chunks || response.evidence_chunks.length === 0) && (
                <div className="text-center py-10 text-slate-500 text-xs">
                  No evidence chunks retrieved for this query.
                </div>
              )}
              {response?.evidence_chunks?.map((chunk, index) => {
                const isSelected = selectedChunkId === chunk.chunk_id;
                return (
                  <div
                    key={chunk.chunk_id || index}
                    onClick={() => setSelectedChunkId(chunk.chunk_id)}
                    className={`p-4 rounded-2xl border text-xs cursor-pointer transition shadow-lg ${
                      isSelected
                        ? 'border-blue-500 bg-blue-950/40 ring-1 ring-blue-500/50'
                        : 'border-slate-800/80 bg-slate-900/60 hover:border-slate-700'
                    }`}
                  >
                    {/* Rank Badge & Score */}
                    <div className="flex justify-between items-center mb-2.5">
                      <span className="font-semibold text-blue-300 bg-blue-950/80 border border-blue-800/60 px-2.5 py-0.5 rounded-md">
                        Rank #{chunk.rank}
                      </span>
                      <span className="font-mono font-medium text-emerald-400 bg-emerald-950/60 border border-emerald-900/60 px-2 py-0.5 rounded-md">
                        Score: {typeof chunk.score === 'number' ? chunk.score.toFixed(4) : chunk.score}
                      </span>
                    </div>

                    {/* Metadata Stack */}
                    <div className="text-slate-400 font-mono text-[11px] mb-2.5 space-y-1 bg-slate-950/80 p-2.5 rounded-xl border border-slate-800/80">
                      <div className="font-semibold text-slate-200 truncate">{chunk.doc_id}</div>
                      <div className="flex justify-between text-slate-400">
                        <span>Page {chunk.page_number}</span>
                        <span className="text-indigo-300 font-sans font-medium">{chunk.section}</span>
                      </div>
                      <div className="text-[10px] text-slate-500 truncate">{chunk.chunk_id}</div>
                    </div>

                    {/* Content Snippet */}
                    <p className="text-slate-300 leading-relaxed italic border-l-2 border-indigo-500/50 pl-2.5">
                      "{chunk.content}"
                    </p>
                  </div>
                );
              })}
            </div>
          )}

          {/* Tab 2: Raw JSON Viewer */}
          {activeTab === 'json' && (
            <div className="flex-1 overflow-y-auto p-4 bg-slate-950 text-indigo-300 font-mono text-xs">
              <pre className="whitespace-pre-wrap leading-relaxed">
                {JSON.stringify(response?.raw || response, null, 2)}
              </pre>
            </div>
          )}

        </div>
      )}

    </div>
  );
}