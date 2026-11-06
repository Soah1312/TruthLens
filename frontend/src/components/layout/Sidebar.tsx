import React from 'react';
import { NavTabId } from '../../types';
import {
  LayoutDashboard,
  FileSearch,
  ShieldAlert,
  Gauge,
  Settings,
  ShieldCheck,
  Terminal,
} from 'lucide-react';

interface SidebarProps {
  activeTab: NavTabId;
  onSelectTab: (tab: NavTabId) => void;
}

interface NavItemConfig {
  id: NavTabId;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  phaseBadge?: string;
  isPlaceholder?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onSelectTab }) => {
  const navItems: NavItemConfig[] = [
    {
      id: 'overview',
      label: 'Overview',
      icon: LayoutDashboard,
      isPlaceholder: false,
    },
    {
      id: 'news-analysis',
      label: 'News Analysis',
      icon: FileSearch,
      phaseBadge: 'Active',
      isPlaceholder: false,
    },
    {
      id: 'robustness',
      label: 'Robustness Evaluation',
      icon: ShieldAlert,
      phaseBadge: 'Phase 4',
      isPlaceholder: true,
    },
    {
      id: 'performance',
      label: 'Model Performance',
      icon: Gauge,
      phaseBadge: 'Phase 2',
      isPlaceholder: true,
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: Settings,
      phaseBadge: 'Future',
      isPlaceholder: true,
    },
  ];

  return (
    <aside className="w-64 bg-[#0F172A] border-r border-slate-800/80 flex flex-col justify-between shrink-0 min-h-screen">
      {/* Top Branding */}
      <div>
        <div className="h-16 flex items-center gap-3 px-6 border-b border-slate-800/80 bg-slate-900/40">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-sky-400 flex items-center justify-center shadow-lg shadow-indigo-500/20 ring-1 ring-white/20">
            <ShieldCheck className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="text-base font-bold tracking-tight text-white flex items-center gap-1.5">
              TruthLens
            </span>
            <span className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold block">
              NLP Academic System
            </span>
          </div>
        </div>

        {/* Navigation list */}
        <nav className="p-4 space-y-1.5">
          <div className="px-3 py-2 text-[11px] font-semibold tracking-wider text-slate-500 uppercase">
            Platform Navigation
          </div>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;

            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-lg text-xs font-medium transition-all group ${
                  isActive
                    ? 'bg-indigo-600/15 text-indigo-300 border border-indigo-500/30 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon
                    className={`w-4 h-4 transition-colors ${
                      isActive ? 'text-indigo-400' : 'text-slate-500 group-hover:text-slate-300'
                    }`}
                  />
                  <span>{item.label}</span>
                </div>

                {item.phaseBadge && (
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded font-mono font-medium border ${
                      isActive
                        ? 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30'
                        : 'bg-slate-800 text-slate-400 border-slate-700/60'
                    }`}
                  >
                    {item.phaseBadge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Bottom Footer Info */}
      <div className="p-4 border-t border-slate-800/80 bg-slate-900/30">
        <div className="bg-slate-900/80 rounded-lg p-3 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400 flex items-center gap-1.5 font-medium">
              <Terminal className="w-3.5 h-3.5 text-indigo-400" />
              Build Status
            </span>
            <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-indigo-950 text-indigo-300 border border-indigo-800">
              Phase 1
            </span>
          </div>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Architecture and health monitoring active. Ready for Phase 2.
          </p>
        </div>
      </div>
    </aside>
  );
};
