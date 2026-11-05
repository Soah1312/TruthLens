export interface HealthResponse {
  status: string;
  application: string;
  version: string;
}

export interface ModelMetricsResponse {
  status: 'available' | 'unavailable';
  message?: string;
  model_name?: string;
  accuracy?: number | null;
  macro_precision?: number | null;
  macro_recall?: number | null;
  macro_f1?: number | null;
  weighted_f1?: number | null;
  evaluation_date?: string | null;
  baseline_metrics?: {
    model?: string;
    accuracy?: number;
    macro_f1?: number;
    weighted_f1?: number;
  } | null;
  per_class_metrics?: Record<
    string,
    {
      precision: number;
      recall: number;
      f1: number;
      support: number;
    }
  > | null;
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
