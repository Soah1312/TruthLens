import React from 'react';
import { Send, Clock, Inbox } from 'lucide-react';

export const RecentAnalysis: React.FC = () => {
  return (
    <div className="bg-[#111827] border border-slate-800/80 rounded-xl p-6">
      <div className="flex items-center justify-between pb-4 border-b border-slate-800/60 mb-6">
        <div>
          <h2 className="text-sm font-semibold text-slate-200 tracking-tight flex items-center gap-2">
            <Clock className="w-4 h-4 text-indigo-400" />
            Recent Analysis Activity
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time feed of parsed articles, classification labels, and evidence retrieval
          </p>
        </div>
        <span className="text-[11px] font-mono text-slate-400 bg-slate-900 px-2.5 py-1 rounded-md border border-slate-800">
          Feed: Inactive (Phase 1)
        </span>
      </div>

      {/* Empty State with Exact Required Message */}
      <div className="flex flex-col items-center justify-center py-12 px-4 text-center rounded-lg border border-dashed border-slate-800 bg-slate-900/30">
        <div className="w-12 h-12 rounded-full bg-slate-800/80 border border-slate-700/60 flex items-center justify-center text-slate-400 mb-4">
          <Inbox className="w-6 h-6 text-slate-400" />
        </div>
        
        <h3 className="text-sm font-medium text-slate-200 mb-1">
          No articles analysed yet. Submit an article through the Telegram bot to get started.
        </h3>
        
        <p className="text-xs text-slate-400 max-w-md mt-1 mb-5 leading-relaxed">
          The intake pipeline will listen to user submissions via Telegram (Phase 3), extract web content using Trafilatura, and feed articles to our fine-tuned DistilBERT engine.
        </p>

        <div className="inline-flex items-center gap-2 text-xs font-medium text-slate-400 bg-slate-800/60 border border-slate-700/50 px-3.5 py-1.5 rounded-lg">
          <Send className="w-3.5 h-3.5 text-sky-400" />
          <span>Telegram Integration Scheduled for Phase 3</span>
        </div>
      </div>
    </div>
  );
};
