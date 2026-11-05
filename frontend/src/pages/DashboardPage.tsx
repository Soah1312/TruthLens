import React from 'react';
import { SummaryCards } from '../components/dashboard/SummaryCards';
import { RecentAnalysis } from '../components/dashboard/RecentAnalysis';
import { ModelPerformance } from '../components/dashboard/ModelPerformance';
import { RobustnessEvaluation } from '../components/dashboard/RobustnessEvaluation';
import { Sparkles, Shield, Compass } from 'lucide-react';

export const DashboardPage: React.FC = () => {
  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      {/* 2. Main Dashboard Welcome Section */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950/30 to-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-xl relative overflow-hidden">
        <div className="absolute right-0 top-0 bottom-0 w-1/3 bg-radial from-indigo-500/5 to-transparent pointer-events-none" />
        
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-medium mb-3">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
            <span>Academic Engineering Project &bull; Phase 1 Initialized</span>
          </div>

          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight mb-2">
            TruthLens Dashboard
          </h1>

          <p className="text-sm text-slate-300 leading-relaxed font-normal">
            TruthLens is an NLP-based fake news detection and robustness evaluation platform. It utilizes fine-tuned transformer language models (DistilBERT), real-time web evidence retrieval, and textual perturbation suites to classify online news and assess model resilience.
          </p>

          <div className="mt-5 flex flex-wrap items-center gap-3 text-xs text-slate-400">
            <span className="flex items-center gap-1.5 bg-slate-900/80 px-2.5 py-1 rounded-md border border-slate-800">
              <Shield className="w-3.5 h-3.5 text-indigo-400" />
              Dataset: LIAR Benchmark
            </span>
            <span className="flex items-center gap-1.5 bg-slate-900/80 px-2.5 py-1 rounded-md border border-slate-800">
              <Compass className="w-3.5 h-3.5 text-sky-400" />
              Architecture: DistilBERT + Tavily + Telegram
            </span>
          </div>
        </div>
      </div>

      {/* 3. Summary Cards Section */}
      <section aria-labelledby="summary-cards-heading">
        <h2 id="summary-cards-heading" className="sr-only">
          Platform Summary Cards
        </h2>
        <SummaryCards />
      </section>

      {/* 4. Recent Analysis Section */}
      <section aria-labelledby="recent-analysis-heading">
        <RecentAnalysis />
      </section>

      {/* 5. Model Performance Section */}
      <section aria-labelledby="model-performance-heading">
        <ModelPerformance />
      </section>

      {/* 6. Robustness Evaluation Section */}
      <section aria-labelledby="robustness-evaluation-heading">
        <RobustnessEvaluation />
      </section>
    </div>
  );
};
