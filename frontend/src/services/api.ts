import { HealthResponse } from '../types';

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
   * Throws an error if the network request fails or server returns a non-200 status code.
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
   * Returns the configured backend API base URL.
   */
  getBaseUrl(): string {
    return API_BASE_URL;
  },
};
