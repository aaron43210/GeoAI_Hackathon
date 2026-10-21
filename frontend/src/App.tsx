import { useEffect, useRef, useState } from 'react';
import { ask, health } from './services/api';
import type { AskResponse, StageEvent } from './types';
import { Markdown } from './components/Markdown';
import { VenueMap } from './components/VenueMap';
import { HourHeatmap } from './components/HourHeatmap';
import { EvidenceDrawer } from './components/EvidenceDrawer';
import { Timeline } from './components/Timeline';

interface Turn {
  role: 'user' | 'assistant';
  text: string;
  data?: AskResponse;
  stages?: StageEvent[];
  draft?: string;
  error?: string;
}

const EXAMPLES = [
  'Good spot for an evening jog near Indiranagar tomorrow?',
  'My parents want a gentle morning walk in Salt Lake this weekend. When is safest?',
  'Box cricket with friends Saturday or Sunday near Thane?',
  'Golden hour photos near Marine Drive Kochi today?',
];

const PROFILES: { id: string; label: string }[] = [
  { id: 'asthma', label: 'Asthma' }, { id: 'senior', label: 'Senior 60+' }, { id: 'child', label: 'Kids' },
  { id: 'pregnancy', label: 'Pregnant' }, { id: 'cardiac', label: 'Heart condition' }, { id: 'heat_sensitive', label: 'Heat-sensitive' },
];

const FACTOR_LABEL: Record<string, string> = {
  thermal: 'Thermal comfort', rain: 'Rain', wind: 'Wind', uv: 'UV', air: 'Air quality', exposure: 'Traffic/industry air',
  setting: 'Green & water', access: 'Travel', light: 'Light',
};

const STATUS_LABEL: Record<string, string> = {
  ok: 'Recommended', partial: 'Recommended (some data missing)', not_recommended: 'Not recommended',
  clarification_needed: 'Need one detail', out_of_scope: 'Out of scope', not_available: 'Not available', error: 'Error',
};

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [session, setSession] = useState<string | null>(null);
  const [loc, setLoc] = useState<{ lat: number; lon: number } | null>(null);
  const [active, setActive] = useState<AskResponse | null>(null);
  const [hourIndex, setHourIndex] = useState(0);
  const [cite, setCite] = useState<string | null>(null);
  const [model, setModel] = useState<string>('checking…');
  const [profiles, setProfiles] = useState<string[]>([]);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    health().then((h) => setModel(h ? (h.model.loaded ? `${h.model.repo ?? 'SLM'} (${h.model.path_used})` : 'SLM not loaded (template mode)') : 'API offline'));
  }, []);
  useEffect(() => endRef.current?.scrollIntoView({ behavior: 'smooth' }), [turns]);

  const submit = async (q: string) => {
    const question = q.trim();
    if (!question || busy) return;
    setInput('');
    setBusy(true);
    setTurns((t) => [...t, { role: 'user', text: question }, { role: 'assistant', text: '', stages: [], draft: '' }]);
    const patch = (fn: (t: Turn) => Turn) =>
      setTurns((ts) => ts.map((t, i) => (i === ts.length - 1 ? fn(t) : t)));
    try {
      const data = await ask(question, {
        sessionId: session,
        location: loc,
        profiles,
        onStage: (s) => patch((t) => ({ ...t, stages: [...(t.stages ?? []), s] })),
        onToken: (tok) => patch((t) => ({ ...t, draft: (t.draft ?? '') + tok })),
      });
      setSession(data.session_id);
      patch((t) => ({ ...t, text: data.answer_markdown, data, draft: undefined }));
      if (data.heatmap?.hours?.length) {
        setActive(data);
        const best = data.recommendation ?? data.least_bad;
        const idx = best ? data.heatmap.hours.findIndex((h) => h.slice(0, 13) === best.start.slice(0, 13)) : 0;
        setHourIndex(Math.max(0, idx));
      }
    } catch (e) {
      patch((t) => ({ ...t, error: (e as Error).message, draft: undefined }));
    } finally {
      setBusy(false);
    }
  };

  const requestLocation = () =>
    navigator.geolocation?.getCurrentPosition(
      (p) => setLoc({ lat: p.coords.latitude, lon: p.coords.longitude }),
      () => alert('Location permission was denied.'),
    );

  const best = active?.recommendation ?? active?.least_bad ?? null;

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <span className="logo">◎</span> TerraPulse <span className="muted">· where & when to head outdoors</span>
        </div>
        <div className="model-pill" title="Small language model running on the server">{model}</div>
      </header>

      <main className="layout">
        <section className="chat card">
          <div className="thread">
            {turns.length === 0 && (
              <div className="empty">
                <h2>Ask about any outdoor plan</h2>
                <p className="muted">I check live weather, air quality, OpenStreetMap roads/industry/green/water and travel
                  times across several places and hours, then explain the best pick with every fact cited.</p>
                <div className="chips">
                  {EXAMPLES.map((e) => <button key={e} className="chip" onClick={() => submit(e)}>{e}</button>)}
                </div>
              </div>
            )}
            {turns.map((t, i) => (
              <div key={i} className={`turn ${t.role}`}>
                {t.role === 'user' ? (
                  <div className="bubble user">{t.text}</div>
                ) : (
                  <div className="bubble bot">
                    {t.data && <div className={`status s-${t.data.status}`}>{STATUS_LABEL[t.data.status] ?? t.data.status}</div>}
                    {!t.data && !t.error && (
                      <div className="stages">
                        {(t.stages ?? []).map((s, k) => <div key={k} className="stage">✓ {s.message}</div>)}
                        <div className="stage pulse">working…</div>
                        {t.draft && <div className="draft"><Markdown text={t.draft} /></div>}
                      </div>
                    )}
                    {t.error && <div className="err">{t.error}</div>}
                    {t.data && <Markdown text={t.text} onCite={(id) => { setActive(t.data!); setCite(id); }} />}
                    {t.data?.clarification && (
                      <div className="chips">
                        {t.data.clarification.options.slice(0, 8).map((o) => (
                          <button key={o} className="chip" onClick={() => (o === 'Use my location' ? requestLocation() : submit(o))}>{o}</button>
                        ))}
                      </div>
                    )}
                    {t.data?.grounding && (
                      <div className="meta">
                        {t.data.grounding.status === 'passed' && `✔ ${t.data.grounding.facts_checked} facts checked against live data`}
                        {t.data.grounding.status === 'repaired' && '✔ facts corrected and re-checked against live data'}
                        {t.data.grounding.status === 'templated' && 'ⓘ shown as a data summary (model draft failed the fact check)'}
                        {' · '}{(t.data.timings_ms.total / 1000).toFixed(1)} s
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            <div ref={endRef} />
          </div>
          <div className="toggles">
            {PROFILES.map((p) => (
              <button key={p.id} type="button" className={`toggle ${profiles.includes(p.id) ? 'on' : ''}`}
                onClick={() => setProfiles((ps) => (ps.includes(p.id) ? ps.filter((x) => x !== p.id) : [...ps, p.id]))}>
                {p.label}
              </button>
            ))}
          </div>
          <form className="composer" onSubmit={(e) => { e.preventDefault(); submit(input); }}>
            <button type="button" className={`loc ${loc ? 'on' : ''}`} onClick={requestLocation} title="Use my location">⌖</button>
            <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="e.g. Is tomorrow 7 am good for a run near Baner?" disabled={busy} />
            <button type="submit" disabled={busy || !input.trim()}>{busy ? '…' : 'Ask'}</button>
          </form>
        </section>

        <section className="results">
          <VenueMap data={active} hourIndex={hourIndex} />
          {active?.heatmap && (
            <HourHeatmap heatmap={active.heatmap} hourIndex={hourIndex} onHour={setHourIndex} bestVenue={best?.venue.name} />
          )}
          {best && (
            <div className="card pick">
              <div className="card-head">
                <h3>{active?.recommendation ? 'Best pick' : 'Least-bad option'}</h3>
                <span className="score">{best.score}</span>
              </div>
              <div className="pick-name"><a href={best.venue.osm_url} target="_blank" rel="noreferrer">{best.venue.name}</a></div>
              <div className="muted">{best.label}{best.travel ? ` · ${best.travel.minutes} min drive (≈${best.travel_at_start_min} with traffic)` : ''}</div>
              <div className="factors">
                {Object.entries(best.factors).map(([k, v]) => (
                  <div key={k} className="factor">
                    <span>{FACTOR_LABEL[k] ?? k}</span>
                    <div className="bar"><i style={{ width: `${Math.round(v * 100)}%` }} /></div>
                  </div>
                ))}
              </div>
              <div className="cols">
                {active?.advice?.carry?.length ? <div><b>Carry</b><ul>{active.advice.carry.map((a) => <li key={a.id}>{a.text}</li>)}</ul></div> : null}
                {active?.advice?.watch_for?.length ? <div><b>Watch for</b><ul>{active.advice.watch_for.map((a) => <li key={a.id}>{a.text}</li>)}</ul></div> : null}
                {active?.avoid?.length ? <div><b>Avoid</b><ul>{active.avoid.map((a) => <li key={a.start}>{a.start.slice(11, 16)}–{a.end.slice(11, 16)}: {a.reasons[0]}</li>)}</ul></div> : null}
              </div>
            </div>
          )}
          {active?.timeline && best && <Timeline points={active.timeline} venue={best.venue.name} />}
          {active?.evidence && (
            <EvidenceDrawer evidence={active.evidence} highlight={cite} sources={active.sources} limitations={active.limitations ?? []} />
          )}
        </section>
      </main>
      <footer className="foot">Map data © OpenStreetMap contributors · Weather & air quality: Open-Meteo (CAMS) · Routing: OSRM · Model: Qwen3-1.7B (Apache-2.0) + LoRA</footer>
    </div>
  );
}
