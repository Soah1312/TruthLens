import React from 'react';
import { ShieldAlert, Split, RefreshCw, AlertCircle } from 'lucide-react';

interface RobustnessCard {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  tag: string;
}

export const RobustnessEvaluation: React.FC = () => {
  const containers: RobustnessCard[] = [
    {
      id: 'original-performance',
      title: 'Original Model Performance',
      subtitle: 'Baseline Clean Data Evaluation',
      description:
        'Benchmark accuracy and F1-score evaluated strictly on unperturbed, pristine news article test sets.',
      icon: ShieldAlert,
      tag: 'Baseline Metrics',
    },
    {
      id: 'perturbed-performance',
      title: 'Perturbed Model Performance',
      subtitle: 'Adversarial Noise Evaluation',
      description:
        'Accuracy and F1-score after introducing typos, keyboard errors, synonym swaps, and negation insertions.',
      icon: Split,
      tag: 'Stress Test',
    },
    {
      id: 'prediction-consistency',
      title: 'Prediction Consistency',
      subtitle: 'Robustness Stability Index',
      description:
        'Percentage of test predictions where the model classification remains identical before and after text perturbation.',
      icon: RefreshCw,
      tag: 'Invariance Rate',
    },
  ];

  return (
    <div className="bg-[#111827] border border-slate-800/80 rounded-xl p-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800/60 mb-6 gap-2">
        <div>
          <h2 className="text-sm font-semibold text-slate-200 tracking-tight flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-indigo-400" />
            Adversarial Robustness Evaluation
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Evaluating model invariance and resilience against adversarial textual perturbations
          </p>
        </div>
        <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-md bg-amber-950/40 border border-amber-500/30 text-amber-300 text-xs font-medium self-start sm:self-auto">
          <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
          <span>Evaluation Scheduled for Phase 4</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {containers.map((item) => {
          const Icon = item.icon;
          return (
            <div
              key={item.id}
              className="bg-slate-900/50 border border-slate-800 rounded-lg p-5 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="p-2 rounded-lg bg-indigo-950/50 border border-indigo-800/40 text-indigo-400">
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded border border-slate-700/60">
                    {item.tag}
                  </span>
                </div>

                <h3 className="text-xs font-bold text-slate-200 tracking-tight">
                  {item.title}
                </h3>
                <span className="text-[10px] text-indigo-400/80 font-medium block mt-0.5 mb-2">
                  {item.subtitle}
                </span>

                {/* Explicit indicator that evaluation has not yet been implemented */}
                <div className="my-3 py-2 px-3 rounded bg-slate-800/60 border border-slate-700/50 text-center">
                  <span className="text-xs font-mono font-medium text-amber-300 block">
                    Evaluation Not Yet Implemented
                  </span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">
                    Planned for Phase 4
                  </span>
                </div>

                <p className="text-[11px] text-slate-400 leading-relaxed">
                  {item.description}
                </p>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-800/80 text-[10px] text-slate-400 font-mono flex items-center justify-between">
                <span>Phase Target:</span>
                <span className="text-indigo-400 font-semibold">Phase 4</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
