export interface HealthResponse {
  status: string;
  application: string;
  version: string;
}

export type ConnectionStatus = 'connected' | 'disconnected' | 'checking';

export type NavTabId =
  | 'overview'
  | 'news-analysis'
  | 'robustness'
  | 'performance'
  | 'settings';

export interface NavItem {
  id: NavTabId;
  label: string;
  iconName: string;
  phaseBadge?: string;
  isPlaceholder?: boolean;
}

export interface SummaryCardData {
  title: string;
  value: string;
  description: string;
  iconName: string;
  statusBadge: string;
}

export interface PerformanceMetricPlaceholder {
  label: string;
  description: string;
  targetMetric: string;
  phase: string;
}

export interface RobustnessEvaluationPlaceholder {
  title: string;
  description: string;
  expectedOutput: string;
  attackType?: string;
  phase: string;
}
