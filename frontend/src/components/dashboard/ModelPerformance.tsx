import React from 'react';
import { Gauge, Info, Cpu } from 'lucide-react';

interface MetricPlaceholder {
  id: string;
  label: string;
  description: string;
  formulaDescription: string;
  category: string;
}

export const ModelPerformance: React.FC = () => {
  const metrics: MetricPlaceholder[] = [
    {
      id: 'accuracy',
      label: 'Accuracy',
      description: 'Overall ratio of correctly classified real vs. fake news samples',
      formulaDescription: '(TP + TN) / (Total Predictions)',
      category: 'Primary Metric',
    },
    {
      id: 'precision',
      label: 'Precision',
      description: 'Proportion of flagged fake articles that are truly fake',
      formulaDescription: 'TP / (TP + FP)',
      category: 'Quality Metric',
    },
    {
      id: 'recall',
      label: 'Recall',
      description: 'Proportion of actual fake news articles correctly detected',
      formulaDescription: 'TP / (TP + FN)',
      category: 'Sensitivity Metric',
    },
    {
      id: 'macro-f1',
      label: 'Macro F1-score',
      description: 'Harmonic mean of precision and recall unweighted across classes',
      formulaDescription: '2 * (P * R) / (P + R)',
      category: 'Balanced Metric',
    },
  ];

  return (
    <div className="bg-[#111827] border border-slate-800/80 rounded-xl p-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800/60 mb-6 gap-2">
        <div>
          <h2 className="text-sm font-semibold text-slate-200 tracking-tight flex items-center gap-2">
            <Gauge className="w-4 h-4 text-indigo-400" />
            DistilBERT Classification Performance
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Validation benchmark on the LIAR dataset (6-class and binary mapped)
          </p>
        </div>
        <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-md bg-amber-950/40 border border-amber-500/30 text-amber-300 text-xs font-medium self-start sm:self-auto">
          <Info className="w-3.5 h-3.5 text-amber-400" />
          <span>Model Training Scheduled for Phase 2</span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((metric) => (
          <div
            key={metric.id}
            className="bg-slate-900/50 border border-slate-800 rounded-lg p-4 flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-slate-300">
                  {metric.label}
                </span>
                <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-1.5 py-0.5 rounded">
                  {metric.category}
                </span>
              </div>

              {/* Explicit placeholder without fabricated values */}
              <div className="my-3 py-2 px-3 rounded bg-slate-800/50 border border-slate-700/40 text-center">
                <span className="text-xs font-mono font-medium text-slate-400 block">
                  Awaiting Phase 2 Evaluation
                </span>
                <span className="text-[10px] text-slate-400 block mt-0.5">
                  No model weights loaded
                </span>
              </div>

              <p className="text-[11px] text-slate-400 leading-relaxed mb-3">
                {metric.description}
              </p>
            </div>

            <div className="pt-2 border-t border-slate-800/80 text-[10px] text-slate-400 font-mono flex items-center justify-between">
              <span>Formula:</span>
              <span className="text-slate-400">{metric.formulaDescription}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-5 p-3.5 rounded-lg bg-slate-900/40 border border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-indigo-400 shrink-0" />
          <span>
            Target Architecture: <strong className="text-slate-300">distilbert-base-uncased</strong> fine-tuned via PyTorch &amp; Hugging Face
          </span>
        </div>
        <span className="text-[11px] font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700/50">
          Status: Uninitialised (Phase 1)
        </span>
      </div>
    </div>
  );
};
