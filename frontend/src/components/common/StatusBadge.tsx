import React from 'react';
import { ConnectionStatus, HealthResponse } from '../../types';
import { RefreshCw, CheckCircle2, AlertCircle, WifiOff } from 'lucide-react';

interface StatusBadgeProps {
  status: ConnectionStatus;
  data: HealthResponse | null;
  errorMessage: string | null;
  isRetrying: boolean;
  onRefresh: () => void;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  data,
  errorMessage,
  isRetrying,
  onRefresh,
}) => {
  if (status === 'connected') {
    return (
      <div className="flex items-center gap-3">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-950/60 border border-emerald-500/30 text-emerald-300 text-xs font-medium shadow-sm backdrop-blur-sm">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          <span>Backend Connected</span>
          {data?.version && (
            <span className="text-emerald-400/70 border-l border-emerald-500/20 pl-2">
              v{data.version}
            </span>
          )}
        </div>
        <button
          onClick={onRefresh}
          disabled={isRetrying}
          title="Refresh health status"
          className="p-1.5 text-slate-400 hover:text-slate-200 rounded-lg hover:bg-slate-800/80 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRetrying ? 'animate-spin' : ''}`} />
        </button>
      </div>
    );
  }

  if (status === 'disconnected') {
    return (
      <div className="flex items-center gap-2">
        <div
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-rose-950/60 border border-rose-500/30 text-rose-300 text-xs font-medium shadow-sm backdrop-blur-sm"
          title={errorMessage || 'Connection to backend failed'}
        >
          <WifiOff className="w-3.5 h-3.5 text-rose-400" />
          <span>Backend Disconnected</span>
        </div>
        <button
          onClick={onRefresh}
          disabled={isRetrying}
          className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-slate-300 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-md transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3 h-3 ${isRetrying ? 'animate-spin' : ''}`} />
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-950/40 border border-amber-500/30 text-amber-300 text-xs font-medium">
      <AlertCircle className="w-3.5 h-3.5 animate-pulse text-amber-400" />
      <span>Connecting to Backend...</span>
    </div>
  );
};
