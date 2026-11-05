import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/api';
import { HealthResponse, ConnectionStatus } from '../types';

export function useHealth(pollIntervalMs = 15000) {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>('checking');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const [isRetrying, setIsRetrying] = useState<boolean>(false);

  const checkHealth = useCallback(async (isManual = false) => {
    if (isManual) {
      setIsRetrying(true);
    }
    try {
      const response = await apiService.getHealth();
      setData(response);
      setStatus('connected');
      setErrorMessage(null);
      setLastChecked(new Date());
    } catch (err: unknown) {
      setStatus('disconnected');
      setData(null);
      const message = err instanceof Error ? err.message : 'Unable to connect to backend server';
      setErrorMessage(message);
      setLastChecked(new Date());
    } finally {
      setIsRetrying(false);
    }
  }, []);

  useEffect(() => {
    // Initial fetch on mount
    checkHealth();

    // Periodic heartbeat check
    const interval = setInterval(() => {
      checkHealth();
    }, pollIntervalMs);

    return () => clearInterval(interval);
  }, [checkHealth, pollIntervalMs]);

  return {
    data,
    status,
    errorMessage,
    lastChecked,
    isRetrying,
    refresh: () => checkHealth(true),
    baseUrl: apiService.getBaseUrl(),
  };
}
