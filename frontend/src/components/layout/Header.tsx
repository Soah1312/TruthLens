import React from 'react';
import { ConnectionStatus, HealthResponse } from '../../types';
import { StatusBadge } from '../common/StatusBadge';
import { Database, Server } from 'lucide-react';

interface HeaderProps {
  status: ConnectionStatus;
  data: HealthResponse | null;
  errorMessage: string | null;
  isRetrying: boolean;
  onRefresh: () => void;
  baseUrl: string;
}

export const Header: React.FC<HeaderProps> = ({
  status,
  data,
  errorMessage,
  isRetrying,
  onRefresh,
  baseUrl,
}) => {
  return (
    <header className="h-16 border-b border-slate-800/80 bg-[#0F172A]/80 backdrop-blur-md px-8 flex items-center justify-between sticky top-0 z-20">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span className="font-semibold text-slate-300">TruthLens</span>
          <span className="text-slate-600">/</span>
          <span className="text-indigo-400 font-medium">Dashboard Overview</span>
        </div>
        <span className="hidden sm:inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-slate-800/90 text-slate-300 border border-slate-700/60">
          <Database className="w-2.5 h-2.5 text-indigo-400" />
          SQLite Dev
        </span>
      </div>

      <div className="flex items-center gap-4">
        <div className="hidden md:flex items-center gap-2 text-[11px] font-mono text-slate-400 bg-slate-900/60 px-3 py-1 rounded-md border border-slate-800">
          <Server className="w-3 h-3 text-slate-500" />
          <span>{baseUrl}</span>
        </div>

        <StatusBadge
          status={status}
          data={data}
          errorMessage={errorMessage}
          isRetrying={isRetrying}
          onRefresh={onRefresh}
        />
      </div>
    </header>
  );
};
