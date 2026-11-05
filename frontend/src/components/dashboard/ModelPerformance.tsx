import React, { useEffect, useState } from 'react';
import { Gauge, Info, Cpu, CheckCircle2, RefreshCw } from 'lucide-react';
import { apiService } from '../../services/api';
import { ModelMetricsResponse } from '../../types';

interface MetricDisplayItem {
  id: string;
  label: string;
  description: string;
  formulaDescription: string;
  category: string;
  getValue: (data: ModelMetricsResponse | null) => string;
}

export const ModelPerformance: React.FC = () => {
  const [metrics, setMetrics] = useState<ModelMetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchMetrics = async () => {
    setLoading(true);
    try {
      const data = await apiService.getModelMetrics();
      setMetrics(data);
    } catch {
      setMetrics({
        status: 'unavailable',
        message: 'Could not connect to metrics API',
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  const metricCards: MetricDisplayItem[] = [
    {
      id: 'accuracy',
      label: 'Accuracy',
      description: 'Overall ratio of correctly classified statements across all 6 classes',
      formulaDescription: '(Correct Predictions) / (Total Predictions)',
      category: 'Primary Metric',
      getValue: (data) =>
        data?.status === 'available' && data.accuracy != null
          ? `${(data.accuracy * 100).toFixed(2)}%`
          : 'Pending Colab Evaluation',
    },
    {
      id: 'macro-precision',
      label: 'Macro Precision',
      description: 'Unweighted mean precision across all six truthfulness labels',
      formulaDescription: 'Mean(TP_c / (TP_c + FP_c))',
      category: 'Quality Metric',
      getValue: (data) =>
        data?.status === 'available' && data.macro_precision != null
          ? `${(data.macro_precision * 100).toFixed(2)}%`
          : 'Pending Colab Evaluation',
    },
    {
      id: 'macro-recall',
      label: 'Macro Recall',
      description: 'Unweighted mean recall across all six truthfulness labels',
      formulaDescription: 'Mean(TP_c / (TP_c + FN_c))',
      category: 'Sensitivity Metric',
      getValue: (data) =>
        data?.status === 'available' && data.macro_recall != null
          ? `${(data.macro_recall * 100).toFixed(2)}%`
          : 'Pending Colab Evaluation',
    },
    {
      id: 'macro-f1',
      label: 'Macro F1-score',
      description: 'Harmonic mean of precision and recall unweighted across classes',
      formulaDescription: '2 * (P * R) / (P + R)',
      category: 'Balanced Metric',
      getValue: (data) =>
        data?.status === 'available' && data.macro_f1 != null
          ? `${(data.macro_f1 * 100).toFixed(2)}%`
          : 'Pending Colab Evaluation',
    },
  ];

  const isAvailable = metrics?.status === 'available';

  return (
    <div className="bg-[#111827] border border-slate-800/80 rounded-xl p-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800/60 mb-6 gap-2">
        <div>
          <h2 className="text-sm font-semibold text-slate-200 tracking-tight flex items-center gap-2">
            <Gauge className="w-4 h-4 text-indigo-400" />
            DistilBERT Classification Performance
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Validation benchmark on the LIAR dataset (6 fine-grained classes)
          </p>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          {isAvailable ? (
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-950/40 border border-emerald-500/30 text-emerald-300 text-xs font-medium">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              <span>Colab Metrics Synced</span>
            </div>
          ) : (
            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-md bg-amber-950/40 border border-amber-500/30 text-amber-300 text-xs font-medium">
              <Info className="w-3.5 h-3.5 text-amber-400" />
              <span>Model Training in Google Colab (Phase 2)</span>
            </div>
          )}

          <button
            onClick={fetchMetrics}
            disabled={loading}
            title="Refresh metrics from backend"
            className="p-1.5 text-slate-400 hover:text-slate-200 rounded-lg hover:bg-slate-800 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {metricCards.map((metric) => {
          const val = metric.getValue(metrics);
          return (
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

                <div className="my-3 py-2 px-3 rounded bg-slate-800/50 border border-slate-700/40 text-center">
                  <span
                    className={`text-sm font-mono font-bold block ${
                      isAvailable ? 'text-indigo-300' : 'text-slate-400'
                    }`}
                  >
                    {val}
                  </span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">
                    {isAvailable ? 'Colab Test Benchmark' : 'No model weights loaded'}
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
          );
        })}
      </div>

      {/* Model Name and Weighted F1 Footer */}
      <div className="mt-5 p-3.5 rounded-lg bg-slate-900/40 border border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between text-xs text-slate-400 gap-2">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-indigo-400 shrink-0" />
          <span>
            Model Name:{' '}
            <strong className="text-slate-200">
              {metrics?.model_name || 'distilbert-base-uncased'}
            </strong>
            {metrics?.weighted_f1 != null && (
              <span className="ml-3 text-indigo-400">
                (Weighted F1: {(metrics.weighted_f1 * 100).toFixed(2)}%)
              </span>
            )}
          </span>
        </div>
        <span className="text-[11px] font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700/50 self-start sm:self-auto">
          Status: {isAvailable ? 'Active Model' : 'Pending Colab Training'}
        </span>
      </div>
    </div>
  );
};
