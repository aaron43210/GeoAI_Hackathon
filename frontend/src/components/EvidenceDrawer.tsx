import React, { useEffect, useRef } from 'react';
import type { Evidence } from '../types';

interface Props {
  evidence: Evidence[];
  highlight: string | null;
  sources: string[];
  limitations: string[];
}

function show(v: unknown): string {
  if (v == null) return 'n/a';
  if (Array.isArray(v)) return v.join(' – ');
  if (typeof v === 'object') return Object.entries(v as Record<string, unknown>).map(([k, x]) => `${k}: ${x}`).join(', ');
  return String(v);
}

/** Every fact the answer may cite: value, unit, source, request URL, method, retrieval time. */
export const EvidenceDrawer: React.FC<Props> = ({ evidence, highlight, sources, limitations }) => {
  const refs = useRef<Record<string, HTMLDivElement | null>>({});
  useEffect(() => {
    if (highlight) refs.current[highlight]?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, [highlight]);
  return (
    <div className="card evidence">
      <div className="card-head">
        <h3>Evidence ({evidence.length})</h3>
      </div>
      <div className="ev-list">
        {evidence.map((e) => (
          <div key={e.id} ref={(el) => { refs.current[e.id] = el; }} className={`ev ${highlight === e.id ? 'hl' : ''}`}>
            <div className="ev-top">
              <span className="ev-id">{e.id}</span>
              <span className="ev-label">{e.label}</span>
              <span className={`ev-kind k-${e.kind}`}>{e.kind}</span>
            </div>
            <div className="ev-val">
              {show(e.value)} {e.unit ?? ''} {e.time ? <span className="muted">· {e.time}</span> : null}
              {e.venue ? <span className="muted"> · {e.venue}</span> : null}
            </div>
            <div className="ev-src">
              {e.url ? <a href={e.url.split(' ')[0]} target="_blank" rel="noreferrer">{e.source}</a> : e.source}
              {e.retrieved_at ? <span className="muted"> · fetched {e.retrieved_at.replace('T', ' ').slice(0, 16)} UTC</span> : null}
            </div>
            {e.method && <div className="ev-method">{e.method}</div>}
          </div>
        ))}
      </div>
      {limitations.length > 0 && (
        <div className="limits">
          <b>Limitations</b>
          <ul>{limitations.map((l) => <li key={l}>{l}</li>)}</ul>
        </div>
      )}
      <div className="sources">{sources.join(' · ')}</div>
    </div>
  );
};
