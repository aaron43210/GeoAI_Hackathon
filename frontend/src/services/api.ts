import type { AskResponse, StageEvent } from '../types';

// Same-origin in production (FastAPI serves the built UI); Vite proxies /api in development.
const API_BASE = import.meta.env.VITE_API_URL || '/api/v1';

export interface AskOptions {
  sessionId?: string | null;
  location?: { lat: number; lon: number } | null;
  profiles?: string[];
  onStage?: (s: StageEvent) => void;
  onToken?: (t: string) => void;
}

/** Streams progress over SSE and resolves with the final grounded answer. No canned fallback: errors surface. */
export function ask(question: string, opts: AskOptions = {}): Promise<AskResponse> {
  const params = new URLSearchParams({ q: question });
  if (opts.sessionId) params.set('session_id', opts.sessionId);
  if (opts.profiles?.length) params.set('profiles', opts.profiles.join(','));
  if (opts.location) {
    params.set('lat', String(opts.location.lat));
    params.set('lon', String(opts.location.lon));
  }
  return new Promise((resolve, reject) => {
    const es = new EventSource(`${API_BASE}/ask/stream?${params.toString()}`);
    let settled = false;
    es.addEventListener('stage', (e) => opts.onStage?.(JSON.parse((e as MessageEvent).data)));
    es.addEventListener('token', (e) => opts.onToken?.(JSON.parse((e as MessageEvent).data).t));
    es.addEventListener('final', (e) => {
      settled = true;
      es.close();
      resolve(JSON.parse((e as MessageEvent).data));
    });
    es.addEventListener('error', (e) => {
      if (settled) return;
      es.close();
      const data = (e as MessageEvent).data;
      reject(new Error(data ? JSON.parse(data).message : 'Connection to the TerraPulse API failed.'));
    });
  });
}

export async function health(): Promise<{ status: string; model: { loaded: boolean; path_used: string; repo: string | null } } | null> {
  try {
    const r = await fetch(`${API_BASE}/health?deep=false`);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}
