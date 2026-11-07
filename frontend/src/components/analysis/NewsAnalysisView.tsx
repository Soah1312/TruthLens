import React, { useState } from 'react';
import {
  FileSearch,
  Send,
  Loader2,
  ExternalLink,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  MinusCircle,
  Scale,
  HelpCircle,
  Sparkles,
  RefreshCw,
  Globe,
  Info,
  Calendar,
  Quote,
  ShieldAlert,
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

const OVERALL_ASSESSMENT_CONFIG: Record<
  string,
  { title: string; bg: string; text: string; border: string; icon: React.ReactNode; description: string }
> = {
  supports: {
    title: 'Evidence Supports Claim',
    bg: 'bg-emerald-950/40',
    text: 'text-emerald-400',
    border: 'border-emerald-800/80',
    icon: <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />,
    description: 'Retrieved external reporting provides direct corroboration for the submitted statement.',
  },
  contradicts: {
    title: 'Evidence Contradicts Claim',
    bg: 'bg-rose-950/40',
    text: 'text-rose-400',
    border: 'border-rose-800/80',
    icon: <XCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />,
    description: 'Retrieved external reporting directly refutes or conflicts with the submitted statement.',
  },
  mixed: {
    title: 'Mixed / Conflicting Evidence',
    bg: 'bg-amber-950/40',
    text: 'text-amber-400',
    border: 'border-amber-800/80',
    icon: <Scale className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />,
    description: 'Retrieved sources contain contradictory findings or differing perspectives on the claim.',
  },
  insufficient_evidence: {
    title: 'Insufficient External Evidence',
    bg: 'bg-slate-900/60',
    text: 'text-slate-300',
    border: 'border-slate-700/80',
    icon: <MinusCircle className="w-5 h-5 text-slate-400 shrink-0 mt-0.5" />,
    description: 'Available external sources do not provide sufficient direct factual reporting to verify or refute the statement.',
  },
};

const RELATIONSHIP_CONFIG: Record<
  string,
  { label: string; bg: string; text: string; border: string }
> = {
  supports: {
    label: 'Supports',
    bg: 'bg-emerald-950/70',
    text: 'text-emerald-300',
    border: 'border-emerald-700/80',
  },
  contradicts: {
    label: 'Contradicts',
    bg: 'bg-rose-950/70',
    text: 'text-rose-300',
    border: 'border-rose-700/80',
  },
  neutral: {
    label: 'Neutral',
    bg: 'bg-slate-800/80',
    text: 'text-slate-300',
    border: 'border-slate-700',
  },
  unclassified: {
    label: 'Unclassified',
    bg: 'bg-slate-800/50',
    text: 'text-slate-400',
    border: 'border-slate-800',
  },
};

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
          <span>Neural Claim Classification &bull; DeBERTa-v3 &bull; Groq Evidence Analysis</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight mb-2">
          Claim &amp; Evidence Analysis
        </h1>
        <p className="text-sm text-slate-300 leading-relaxed max-w-3xl">
          Submit any public claim or news statement. TruthLens executes 6-class neural classification with fine-tuned DeBERTa-v3, retrieves real-time web sources via Tavily, and analyzes evidence relationships using Groq LLM reasoning.
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
                      <span>Analyzing Claim &amp; Evaluating Evidence...</span>
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
              <span>Academic Tripartite Pipeline</span>
            </div>
            <ul className="space-y-1.5 text-[11px] leading-relaxed">
              <li>
                <strong className="text-slate-300">1. DeBERTa-v3:</strong> Neural 6-class claim veracity estimation on LIAR benchmark.
              </li>
              <li>
                <strong className="text-slate-300">2. Tavily Search:</strong> Real-time retrieval of up to 5 external news articles.
              </li>
              <li>
                <strong className="text-slate-300">3. Groq LLM:</strong> Grounded source relationship classification (supports, contradicts, neutral) with verified quotation extraction.
              </li>
            </ul>
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
                Enter a claim on the left to view the 6-class neural truthfulness probability distribution and external web evidence assessment.
              </p>
            </div>
          )}

          {isLoading && (
            <div className="bg-[#111827] border border-slate-800 rounded-xl p-12 text-center flex flex-col items-center justify-center min-h-[380px] space-y-4">
              <Loader2 className="w-8 h-8 text-indigo-400 animate-spin" />
              <div>
                <h3 className="text-sm font-semibold text-white mb-1">
                  Evaluating Neural Credibility &amp; Evidence...
                </h3>
                <p className="text-xs text-slate-400 max-w-sm">
                  Computing DeBERTa-v3 embeddings, retrieving external web sources via Tavily, and executing Groq LLM grounded relationship analysis.
                </p>
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-6 animate-fadeIn">
              {/* SECTION A: MODEL PREDICTION */}
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                  <span className="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-mono">
                    SECTION A
                  </span>
                  <span>Neural Model Prediction</span>
                </div>

                <div className={`rounded-xl border p-6 ${predConfig?.bg} ${predConfig?.border}`}>
                  <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
                    <div>
                      <span className="text-[11px] uppercase tracking-wider font-semibold text-slate-400 block mb-0.5">
                        Predicted Classification
                      </span>
                      <div className="flex items-center gap-2.5 flex-wrap">
                        <span className={`text-2xl font-extrabold ${predConfig?.text} uppercase tracking-tight`}>
                          {result.prediction.label}
                        </span>
                        {result.model && (
                          <span className="text-[10px] font-mono px-2.5 py-0.5 rounded-full bg-slate-900/90 border border-indigo-500/30 text-indigo-300">
                            {result.model}
                          </span>
                        )}
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
                      The {Math.round(result.prediction.confidence * 100)}% score reflects statistical neural certainty across learned linguistic features, <span className="underline decoration-amber-500/50 font-semibold">not independently verified truth</span>.
                    </div>
                  </div>
                </div>

                {/* Six-Class Score Distribution */}
                <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-3">
                  <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                      Six-Class Veracity Distribution
                    </h3>
                    <span className="text-[11px] font-mono text-slate-400">
                      Architecture: {result.model}
                    </span>
                  </div>

                  <div className="space-y-2.5">
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
              </div>

              {/* SECTION B: EXTERNAL EVIDENCE & GROQ LLM ANALYSIS */}
              <div className="space-y-4 pt-2">
                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                  <span className="px-2 py-0.5 rounded bg-sky-500/20 text-sky-300 border border-sky-500/30 font-mono">
                    SECTION B
                  </span>
                  <span>External Evidence &amp; Groq LLM Analysis</span>
                </div>

                {/* Overall Assessment Banner */}
                {result.evidence?.overall_assessment && (
                  <div
                    className={`rounded-xl border p-5 ${
                      OVERALL_ASSESSMENT_CONFIG[result.evidence.overall_assessment]?.bg || 'bg-slate-900/60'
                    } ${
                      OVERALL_ASSESSMENT_CONFIG[result.evidence.overall_assessment]?.border || 'border-slate-800'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      {OVERALL_ASSESSMENT_CONFIG[result.evidence.overall_assessment]?.icon}
                      <div className="space-y-1 flex-1">
                        <div className="flex items-center justify-between gap-2 flex-wrap">
                          <h4
                            className={`text-sm font-bold uppercase tracking-wide ${
                              OVERALL_ASSESSMENT_CONFIG[result.evidence.overall_assessment]?.text || 'text-slate-200'
                            }`}
                          >
                            {OVERALL_ASSESSMENT_CONFIG[result.evidence.overall_assessment]?.title ||
                              result.evidence.overall_assessment.replace('_', ' ').toUpperCase()}
                          </h4>
                          {result.evidence.analysis_status && (
                            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-950/80 border border-slate-800 text-slate-400">
                              Groq Status: {result.evidence.analysis_status}
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-slate-300 leading-relaxed">
                          {result.evidence.analysis_summary ||
                            OVERALL_ASSESSMENT_CONFIG[result.evidence.overall_assessment]?.description}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Retrieved Sources Breakdown */}
                <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
                  <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                    <div className="flex items-center gap-2">
                      <Globe className="w-4 h-4 text-sky-400" />
                      <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                        Retrieved Sources ({result.evidence?.results?.length || 0})
                      </h3>
                    </div>
                    <span
                      className={`text-[10px] font-semibold px-2 py-0.5 rounded ${
                        result.evidence?.status === 'success'
                          ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                          : 'bg-amber-950 text-amber-400 border border-amber-800'
                      }`}
                    >
                      {result.evidence?.status === 'success' ? 'Search Completed' : 'Evidence Unavailable'}
                    </span>
                  </div>

                  {result.evidence?.status === 'success' && result.evidence.results.length > 0 ? (
                    <div className="space-y-4">
                      {result.evidence.results.map((src, idx) => {
                        const rel = src.classification || 'unclassified';
                        const relCfg = RELATIONSHIP_CONFIG[rel] || RELATIONSHIP_CONFIG.unclassified;

                        return (
                          <div
                            key={idx}
                            className="p-4 rounded-lg bg-slate-900/70 border border-slate-800/80 hover:border-slate-700 transition space-y-2.5"
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div className="space-y-1 flex-1">
                                <h4 className="text-xs font-semibold text-slate-200 leading-snug">
                                  {src.title}
                                </h4>
                                <div className="flex items-center gap-3 text-[11px] text-slate-400 flex-wrap">
                                  <span className="font-mono text-slate-400">{src.domain}</span>
                                  {src.published_date && (
                                    <span className="flex items-center gap-1 text-slate-500">
                                      <Calendar className="w-3 h-3" />
                                      {src.published_date}
                                    </span>
                                  )}
                                  <span className="text-[10px] font-mono text-slate-500">
                                    Relevance: {src.relevance_score.toFixed(2)}
                                  </span>
                                </div>
                              </div>

                              {/* Relationship Badge */}
                              <div className="flex flex-col items-end gap-1 shrink-0">
                                <span
                                  className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border ${relCfg.bg} ${relCfg.text} ${relCfg.border}`}
                                >
                                  {relCfg.label}
                                </span>
                                {src.relevance && (
                                  <span className="text-[9px] font-mono uppercase text-slate-400">
                                    {src.relevance} relevance
                                  </span>
                                )}
                              </div>
                            </div>

                            {/* Reasoning */}
                            {src.reasoning && (
                              <div className="text-xs text-slate-300 leading-relaxed bg-slate-950/60 p-2.5 rounded-lg border border-slate-800/60">
                                <span className="font-semibold text-indigo-300 text-[11px]">Analysis: </span>
                                {src.reasoning}
                              </div>
                            )}

                            {/* Verified Evidence Quote */}
                            {src.evidence_quote && (
                              <div className="text-[11px] text-slate-300 leading-relaxed bg-indigo-950/20 border-l-2 border-indigo-500 px-3 py-1.5 rounded-r flex items-start gap-2">
                                <Quote className="w-3.5 h-3.5 text-indigo-400 shrink-0 mt-0.5" />
                                <div>
                                  <span className="text-[10px] uppercase font-bold text-indigo-400 block mb-0.5">
                                    Verified Source Excerpt:
                                  </span>
                                  <span className="italic font-serif">"{src.evidence_quote}"</span>
                                </div>
                              </div>
                            )}

                            {/* Original Snippet if no reasoning/quote */}
                            {!src.reasoning && !src.evidence_quote && (
                              <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                                {src.snippet}
                              </p>
                            )}

                            <div className="flex items-center justify-end text-[11px] pt-1 border-t border-slate-800/40">
                              <a
                                href={src.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 text-sky-400 hover:text-sky-300 text-xs font-medium"
                              >
                                <span>Inspect Source URL</span>
                                <ExternalLink className="w-3 h-3" />
                              </a>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="p-4 rounded-lg bg-slate-900/40 border border-slate-800 text-xs text-slate-400 text-center">
                      {result.evidence?.summary || 'No external evidence could be retrieved at this time.'}
                    </div>
                  )}

                  {/* Methodological Limitations */}
                  {result.evidence?.limitations && result.evidence.limitations.length > 0 && (
                    <div className="pt-2 border-t border-slate-800 space-y-1.5">
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-400">
                        <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
                        <span>Evidence Caveats &amp; Limitations</span>
                      </div>
                      <ul className="space-y-1 text-[11px] text-slate-400 pl-4 list-disc">
                        {result.evidence.limitations.map((lim, lIdx) => (
                          <li key={lIdx}>{lim}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
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
