import { HealthResponse, ModelMetricsResponse, AnalyzeApiResponse } from '../types';

// Read API base URL from Vite environment variable with safe fallback
const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string) || 'http://localhost:8000';

/**
 * Normalizes URL path to avoid trailing or missing slashes.
 */
function buildUrl(endpoint: string): string {
  const base = API_BASE_URL.replace(/\/+$/, '');
  const path = endpoint.replace(/^\/+/, '');
  return `${base}/${path}`;
}

export const apiService = {
  /**
   * Fetches backend application health status.
   */
  async getHealth(): Promise<HealthResponse> {
    const url = buildUrl('/api/health');
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Health check failed with status: ${response.status} ${response.statusText}`);
    }

    const data: HealthResponse = await response.json();
    return data;
  },

  /**
   * Fetches trained model evaluation benchmark metrics.
   */
  async getModelMetrics(): Promise<ModelMetricsResponse> {
    const url = buildUrl('/api/model/metrics');
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch model metrics: ${response.status} ${response.statusText}`);
    }

    const data: ModelMetricsResponse = await response.json();
    return data;
  },

  /**
   * Dispatches claim text to the RoBERTa + Tavily analysis endpoint.
   */
  async analyzeClaim(
    text: string,
    metadata?: { speaker?: string; context?: string; subject?: string } | null,
    includeEvidence: boolean = true
  ): Promise<AnalyzeApiResponse> {
    const url = buildUrl('/api/analyze');
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({
        text,
        metadata: metadata || null,
        include_evidence: includeEvidence,
      }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      const detail = errData.detail || `Analysis failed with status ${response.status}`;
      throw new Error(detail);
    }

    const data: AnalyzeApiResponse = await response.json();
    return data;
  },

  /**
   * Returns the configured backend API base URL.
   */
  getBaseUrl(): string {
    return API_BASE_URL;
  },
};
