import React from 'react';
import type { Heatmap } from '../types';
import { hourLabel, scoreColor } from './scale';

interface Props {
  heatmap: Heatmap;
  hourIndex: number;
  onHour: (i: number) => void;
  bestVenue?: string;
}

/** Venue × hour grid: the place-and-time reasoning made visible. Hatched cells are gated by a safety rule. */
export const HourHeatmap: React.FC<Props> = ({ heatmap, hourIndex, onHour, bestVenue }) => {
  const { hours, venues, scores, status, reasons } = heatmap;
  if (!hours.length) return null;
  return (
    <div className="heatmap card">
      <div className="card-head">
        <h3>Place × time</h3>
        <span className="muted">{hourLabel(hours[hourIndex])}</span>
      </div>
      <input
        className="scrubber"
        type="range"
        min={0}
        max={hours.length - 1}
        value={hourIndex}
        onChange={(e) => onHour(Number(e.target.value))}
        aria-label="Hour"
      />
      <div className="grid-scroll">
        <table>
          <thead>
            <tr>
              <th />
              {hours.map((h, i) => (
                <th key={h} className={i === hourIndex ? 'sel' : ''} onClick={() => onHour(i)}>
                  {new Date(h).getHours()}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {venues.map((v, vi) => (
              <tr key={v} className={v === bestVenue ? 'best' : ''}>
                <th title={v}>{v.length > 22 ? v.slice(0, 21) + '…' : v}</th>
                {hours.map((h, hi) => {
                  const s = scores[vi][hi];
                  const st = status[vi][hi];
                  const r = reasons[vi][hi];
                  return (
                    <td
                      key={h}
                      className={`${st === 'avoid' ? 'avoid' : ''} ${hi === hourIndex ? 'sel' : ''}`}
                      style={{ background: st === 'avoid' ? undefined : scoreColor(s) }}
                      title={`${v} · ${hourLabel(h)} · ${s ?? 'n/a'}${r.length ? '\n' + r.join('\n') : ''}`}
                      onClick={() => onHour(hi)}
                    />
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
