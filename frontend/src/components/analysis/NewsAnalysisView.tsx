import React, { useState } from 'react';
import {
  FileSearch,
  Send,
  Loader2,
  ExternalLink,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  Sparkles,
  RefreshCw,
  Globe,
  Info,
} from 'lucide-react';
import { apiService } from '../../services/api';
import { AnalyzeApiResponse, VeracityLabel } from '../../types';

const LABEL_CONFIG: Record<
  VeracityLabel,
  { name: string; bg: string; text: string; border: string; barColor: string }
> = {
  'pants-fire': {
    name: 'Pants-fire',
    bg: 'bg-rose-950/60',
    text: 'text-rose-400',
    border: 'border-rose-800/80',
    barColor: 'bg-rose-500',
  },
  false: {
    name: 'False',
    bg: 'bg-red-950/60',
    text: 'text-red-400',
    border: 'border-red-800/80',
    barColor: 'bg-red-500',
  },
  'barely-true': {
    name: 'Barely-true',
    bg: 'bg-orange-950/60',
    text: 'text-orange-400',
    border: 'border-orange-800/80',
    barColor: 'bg-orange-500',
  },
  'half-true': {
    name: 'Half-true',
    bg: 'bg-amber-950/60',
    text: 'text-amber-400',
    border: 'border-amber-800/80',
    barColor: 'bg-amber-500',
  },
  'mostly-true': {
    name: 'Mostly-true',
    bg: 'bg-emerald-950/60',
    text: 'text-emerald-400',
    border: 'border-emerald-800/80',
    barColor: 'bg-emerald-500',
  },
  true: {
    name: 'True',
    bg: 'bg-green-950/60',
    text: 'text-green-400',
    border: 'border-green-800/80',
    barColor: 'bg-green-500',
  },
};

const ORDERED_LABELS: VeracityLabel[] = [
  'pants-fire',
  'false',
  'barely-true',
  'half-true',
  'mostly-true',
  'true',
];

export const NewsAnalysisView: React.FC = () => {
  const [claimText, setClaimText] = useState('');
  const [speaker, setSpeaker] = useState('');
  const [context, setContext] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeApiResponse | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = claimText.trim();
    if (!trimmed) {
      setError('Please enter a statement or news claim to analyze.');
      return;
    }
    if (trimmed.length < 3) {
      setError('Statement text must be at least 3 characters.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const metadata =
        speaker.trim() || context.trim()
          ? {
              speaker: speaker.trim() || 'unknown',
              context: context.trim() || 'unknown',
            }
          : null;

      const response = await apiService.analyzeClaim(trimmed, metadata, true);
      setResult(response);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'An unexpected error occurred.';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setClaimText('');
    setSpeaker('');
    setContext('');
    setResult(null);
    setError(null);
  };

  const predConfig = result ? LABEL_CONFIG[result.prediction.label] : null;

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950/30 to-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-xl">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-medium mb-3">
          <FileSearch className="w-3.5 h-3.5 text-indigo-400" />
          <span>Interactive Neural Veracity Analysis &bull; RoBERTa + Tavily</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight mb-2">
          Claim &amp; News Analysis
        </h1>
        <p className="text-sm text-slate-300 leading-relaxed max-w-3xl">
          Submit any public claim or news statement. The fine-tuned RoBERTa model processes statement semantics paired with optional metadata, while Tavily simultaneously retrieves real-time web evidence to provide comprehensive context.
        </p>
      </div>

      {/* Main Grid: Input Form + Results */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Form (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-[#111827] border border-slate-800 rounded-xl p-6 shadow-sm">
            <h2 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              Submit Statement for Verification
            </h2>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label htmlFor="claim-text" className="block text-xs font-medium text-slate-300 mb-1.5">
                  Statement Text <span className="text-rose-400">*</span>
                </label>
                <textarea
                  id="claim-text"
                  rows={4}
                  value={claimText}
                  onChange={(e) => setClaimText(e.target.value)}
                  placeholder="e.g., Inflation in the United States reached a 40-year high last month..."
                  className="w-full rounded-lg bg-slate-900/80 border border-slate-700/80 px-3.5 py-2.5 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
                  disabled={isLoading}
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label htmlFor="speaker" className="block text-xs font-medium text-slate-400 mb-1">
                    Speaker (Optional)
                  </label>
                  <input
                    id="speaker"
                    type="text"
                    value={speaker}
                    onChange={(e) => setSpeaker(e.target.value)}
                    placeholder="e.g., Joe Biden, Donald Trump"
                    className="w-full rounded-lg bg-slate-900/80 border border-slate-700/80 px-3 py-2 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition"
                    disabled={isLoading}
                  />
                </div>
                <div>
                  <label htmlFor="context" className="block text-xs font-medium text-slate-400 mb-1">
                    Context / Venue (Optional)
                  </label>
                  <input
                    id="context"
                    type="text"
                    value={context}
                    onChange={(e) => setContext(e.target.value)}
                    placeholder="e.g., Press conference, Twitter"
                    className="w-full rounded-lg bg-slate-900/80 border border-slate-700/80 px-3 py-2 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition"
                    disabled={isLoading}
                  />
                </div>
              </div>

              {error && (
                <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                  <span>{error}</span>
                </div>
              )}

              <div className="flex items-center gap-3 pt-2">
                <button
                  type="submit"
                  disabled={isLoading}
                  className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-900/50 disabled:cursor-not-allowed text-white text-xs font-semibold shadow-md shadow-indigo-600/20 transition"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Analyzing Claim &amp; Searching Web...</span>
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4" />
                      <span>Run TruthLens Analysis</span>
                    </>
                  )}
                </button>

                {result && (
                  <button
                    type="button"
                    onClick={handleReset}
                    className="px-3 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition"
                  >
                    <RefreshCw className="w-4 h-4" />
                  </button>
                )}
              </div>
            </form>
          </div>

          {/* Quick Guidance Box */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 space-y-2 text-xs text-slate-400">
            <div className="flex items-center gap-2 text-slate-300 font-medium">
              <Info className="w-4 h-4 text-sky-400" />
              <span>Academic Analysis Architecture</span>
            </div>
            <p className="leading-relaxed text-[11px]">
              Predictions are generated by the fine-tuned <code className="text-indigo-300 font-mono">soah-13/truthlens-roberta-liar-meta</code> model evaluated against the six LIAR benchmark categories. Evidence is gathered via Tavily Search API.
            </p>
          </div>
        </div>

        {/* Right Column: Analysis Results (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {!result && !isLoading && (
            <div className="bg-[#111827] border border-dashed border-slate-800 rounded-xl p-12 text-center flex flex-col items-center justify-center min-h-[380px]">
              <div className="w-12 h-12 rounded-xl bg-slate-900 border border-slate-800 text-slate-500 flex items-center justify-center mb-4">
                <FileSearch className="w-6 h-6" />
              </div>
              <h3 className="text-sm font-semibold text-slate-200 mb-1">
                Awaiting Statement Submission
              </h3>
              <p className="text-xs text-slate-400 max-w-sm">
                Enter a claim on the left to view the 6-class neural truthfulness probability distribution and external web evidence sources.
              </p>
            </div>
          )}

          {isLoading && (
            <div className="bg-[#111827] border border-slate-800 rounded-xl p-12 text-center flex flex-col items-center justify-center min-h-[380px] space-y-4">
              <Loader2 className="w-8 h-8 text-indigo-400 animate-spin" />
              <div>
                <h3 className="text-sm font-semibold text-white mb-1">
                  Evaluating Neural Credibility...
                </h3>
                <p className="text-xs text-slate-400 max-w-sm">
                  Computing RoBERTa sentence-pair embeddings and querying Tavily search indexes for corroborating articles.
                </p>
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-6 animate-fadeIn">
              {/* Prediction Summary Card */}
              <div className={`rounded-xl border p-6 ${predConfig?.bg} ${predConfig?.border}`}>
                <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
                  <div>
                    <span className="text-[11px] uppercase tracking-wider font-semibold text-slate-400 block mb-0.5">
                      Predicted Classification
                    </span>
                    <div className="flex items-center gap-2">
                      <span className={`text-2xl font-extrabold ${predConfig?.text} uppercase tracking-tight`}>
                        {result.prediction.label}
                      </span>
                    </div>
                  </div>

                  <div className="bg-slate-950/70 border border-slate-800 rounded-lg px-4 py-2 text-right">
                    <span className="text-[10px] uppercase tracking-wider text-slate-400 block">
                      Model Confidence
                    </span>
                    <span className="text-lg font-bold font-mono text-white">
                      {Math.round(result.prediction.confidence * 100)}%
                    </span>
                  </div>
                </div>

                {/* Clear distinction: Model Confidence vs Factual Certainty */}
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80 text-xs text-slate-300 flex items-start gap-2.5">
                  <HelpCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  <div className="leading-relaxed">
                    <strong className="text-amber-300">Confidence Interpretation:</strong>{' '}
                    The {Math.round(result.prediction.confidence * 100)}% score reflects statistical neural model certainty on learned linguistic patterns, <span className="underline decoration-amber-500/50">not verified factual certainty</span>.
                  </div>
                </div>
              </div>

              {/* Six-Class Score Distribution */}
              <div className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-4">
                <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                    Six-Class Veracity Distribution
                  </h3>
                  <span className="text-[11px] font-mono text-slate-400">
                    Model: {result.model}
                  </span>
                </div>

                <div className="space-y-3">
                  {ORDERED_LABELS.map((label) => {
                    const score = result.prediction.class_scores[label] || 0;
                    const pct = Math.round(score * 100);
                    const isWinner = label === result.prediction.label;
                    const cfg = LABEL_CONFIG[label];

                    return (
                      <div key={label} className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span
                            className={`font-medium ${
                              isWinner ? `${cfg.text} font-bold flex items-center gap-1.5` : 'text-slate-400'
                            }`}
                          >
                            {cfg.name}
                            {isWinner && <CheckCircle2 className="w-3.5 h-3.5" />}
                          </span>
                          <span className="font-mono text-slate-300">{pct}%</span>
                        </div>
                        <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${cfg.barColor} transition-all duration-500`}
                            style={{ width: `${Math.max(pct, 2)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* External Evidence Section */}
              <div className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-4">
                <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                  <div className="flex items-center gap-2">
                    <Globe className="w-4 h-4 text-sky-400" />
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                      External Web Evidence (Tavily Search)
                    </h3>
                  </div>
                  <span
                    className={`text-[10px] font-semibold px-2 py-0.5 rounded ${
                      result.evidence?.status === 'success'
                        ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                        : 'bg-amber-950 text-amber-400 border border-amber-800'
                    }`}
                  >
                    {result.evidence?.status === 'success' ? 'Evidence Retrieved' : 'Evidence Unavailable'}
                  </span>
                </div>

                {result.evidence?.status === 'success' && result.evidence.results.length > 0 ? (
                  <div className="space-y-3">
                    {result.evidence.results.map((src, idx) => (
                      <div
                        key={idx}
                        className="p-3.5 rounded-lg bg-slate-900/70 border border-slate-800/80 hover:border-slate-700 transition space-y-1.5"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <h4 className="text-xs font-semibold text-slate-200 line-clamp-1">
                            {src.title}
                          </h4>
                          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 shrink-0">
                            Relevance: {src.relevance_score.toFixed(2)}
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                          {src.snippet}
                        </p>
                        <div className="flex items-center justify-between text-[11px] pt-1">
                          <span className="text-slate-500 font-mono text-[10px]">{src.domain}</span>
                          <a
                            href={src.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 text-sky-400 hover:text-sky-300 text-xs font-medium"
                          >
                            <span>Inspect Source</span>
                            <ExternalLink className="w-3 h-3" />
                          </a>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-4 rounded-lg bg-slate-900/40 border border-slate-800 text-xs text-slate-400 text-center">
                    {result.evidence?.summary || 'No external evidence could be retrieved at this time.'}
                  </div>
                )}
              </div>

              {/* Academic Disclaimer */}
              <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800/80 text-[11px] text-slate-400 leading-relaxed">
                <span className="font-semibold text-slate-300">Academic Disclaimer:</span>{' '}
                {result.disclaimer}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
