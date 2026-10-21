import React from 'react';
import type { TimelinePoint } from '../types';
import { scoreColor } from './scale';

interface Props {
  points: TimelinePoint[];
  venue: string;
}

const ROWS: { key: keyof TimelinePoint; label: string; unit: string; fmt?: (v: number) => string }[] = [
  { key: 'apparent', label: 'Feels like', unit: '°C', fmt: (v) => v.toFixed(0) },
  { key: 'rain_prob', label: 'Rain', unit: '%' },
  { key: 'uv', label: 'UV', unit: '', fmt: (v) => v.toFixed(0) },
  { key: 'us_aqi', label: 'US AQI', unit: '' },
  { key: 'exposure', label: 'Traffic air', unit: '', fmt: (v) => v.toFixed(2) },
];

/** Hour-by-hour strip for the chosen venue; the recommended window is outlined. */
export const Timeline: React.FC<Props> = ({ points, venue }) => {
  if (!points.length) return null;
  return (
    <div className="card timeline">
      <div className="card-head">
        <h3>Hour by hour · {venue}</h3>
      </div>
      <div className="grid-scroll">
        <table>
          <thead>
            <tr>
              <th />
              {points.map((p) => (
                <th key={p.ts} className={p.in_best ? 'win' : ''}>{new Date(p.ts).getHours()}h</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>Score</th>
              {points.map((p) => (
                <td key={p.ts} className={`${p.in_best ? 'win' : ''} ${p.status === 'avoid' ? 'avoid' : ''}`}
                    style={{ background: p.status === 'avoid' ? undefined : scoreColor(p.score), color: '#04130d' }}>
                  {Math.round(p.score)}
                </td>
              ))}
            </tr>
            {ROWS.map((r) => (
              <tr key={r.key}>
                <th>{r.label}</th>
                {points.map((p) => {
                  const v = p[r.key] as number | null;
                  return <td key={p.ts} className={p.in_best ? 'win' : ''}>{v == null ? '–' : (r.fmt ? r.fmt(v) : v)}{v == null ? '' : r.unit}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
