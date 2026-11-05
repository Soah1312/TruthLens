import React, { useState } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { DashboardPage } from './pages/DashboardPage';
import { useHealth } from './hooks/useHealth';
import { NavTabId } from './types';
import { Layers, ArrowLeft } from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavTabId>('overview');
  const { data, status, errorMessage, isRetrying, refresh, baseUrl } = useHealth();

  return (
    <div className="min-h-screen bg-[#0B0F19] text-slate-100 flex">
      {/* 1. Sidebar Navigation */}
      <Sidebar activeTab={activeTab} onSelectTab={setActiveTab} />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <Header
          status={status}
          data={data}
          errorMessage={errorMessage}
          isRetrying={isRetrying}
          onRefresh={refresh}
          baseUrl={baseUrl}
        />

        <main className="flex-1 overflow-y-auto">
          {activeTab === 'overview' ? (
            <DashboardPage />
          ) : (
            <div className="p-8 max-w-4xl mx-auto py-20 text-center">
              <div className="w-14 h-14 rounded-2xl bg-indigo-950/60 border border-indigo-800/40 text-indigo-400 mx-auto flex items-center justify-center mb-4">
                <Layers className="w-7 h-7" />
              </div>
              <h2 className="text-xl font-bold text-white capitalize mb-2">
                {activeTab.replace('-', ' ')}
              </h2>
              <p className="text-sm text-slate-400 max-w-md mx-auto mb-6">
                This module is designated for implementation in subsequent phases of the TruthLens project roadmap. Currently active: Phase 1 (Foundation &amp; System Health).
              </p>
              <button
                onClick={() => setActiveTab('overview')}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                Return to Overview Dashboard
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
