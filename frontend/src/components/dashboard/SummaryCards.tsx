import React from 'react';
import { Newspaper, CheckCircle, AlertTriangle, Target } from 'lucide-react';

interface MetricItem {
  id: string;
  title: string;
  value: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  accentColor: string;
  badgeText: string;
}

export const SummaryCards: React.FC = () => {
  const cards: MetricItem[] = [
    {
      id: 'total-articles',
      title: 'Total Articles Analysed',
      value: 'Not available yet',
      description: 'Cumulative news articles submitted for verification',
      icon: Newspaper,
      accentColor: 'text-sky-400 bg-sky-950/40 border-sky-500/20',
      badgeText: 'Phase 3 Pipeline',
    },
    {
      id: 'real-news',
      title: 'Real News Detected',
      value: 'Not available yet',
      description: 'Articles classified as authentic and fact-supported',
      icon: CheckCircle,
      accentColor: 'text-emerald-400 bg-emerald-950/40 border-emerald-500/20',
      badgeText: 'Phase 2 Classifier',
    },
    {
      id: 'fake-news',
      title: 'Fake News Detected',
      value: 'Not available yet',
      description: 'Articles flagged with fabricated or unverified claims',
      icon: AlertTriangle,
      accentColor: 'text-amber-400 bg-amber-950/40 border-amber-500/20',
      badgeText: 'Phase 2 Classifier',
    },
    {
      id: 'model-accuracy',
      title: 'Model Accuracy',
      value: 'Not available yet',
      description: 'Current validation accuracy on LIAR test benchmark',
      icon: Target,
      accentColor: 'text-indigo-400 bg-indigo-950/40 border-indigo-500/20',
      badgeText: 'Phase 2 Training',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card) => {
        const Icon = card.icon;
        return (
          <div
            key={card.id}
            className="bg-[#111827] border border-slate-800/80 rounded-xl p-5 hover:border-slate-700/80 transition-all flex flex-col justify-between"
          >
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-medium text-slate-400 tracking-wide">
                  {card.title}
                </span>
                <div className={`p-2 rounded-lg border ${card.accentColor}`}>
                  <Icon className="w-4 h-4" />
                </div>
              </div>

              {/* Exact required value: "Not available yet" */}
              <div className="text-lg font-semibold text-slate-300 tracking-tight mb-1 font-mono">
                {card.value}
              </div>

              <p className="text-[11px] text-slate-400 leading-relaxed mb-4">
                {card.description}
              </p>
            </div>

            <div className="pt-3 border-t border-slate-800/60 flex items-center justify-between text-[10px]">
              <span className="text-slate-400">Status</span>
              <span className="font-mono text-slate-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                {card.badgeText}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
};
