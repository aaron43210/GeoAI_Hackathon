import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { AskResponse } from '../types';
import { scoreColor } from './scale';

interface Props {
  data: AskResponse | null;
  hourIndex: number;
}

/** Venues coloured by their score at the selected hour, the recommended venue ringed, and the hour's wind. */
export const VenueMap: React.FC<Props> = ({ data, hourIndex }) => {
  const ref = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const layer = useRef<L.LayerGroup | null>(null);

  useEffect(() => {
    if (!ref.current || map.current) return;
    map.current = L.map(ref.current, { zoomControl: false, center: [20.6, 78.9], zoom: 5 });
    L.control.zoom({ position: 'bottomright' }).addTo(map.current);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
      attribution: 'Map data &copy; OpenStreetMap contributors &copy; CARTO',
      maxZoom: 19,
    }).addTo(map.current);
    layer.current = L.layerGroup().addTo(map.current);
  }, []);

  useEffect(() => {
    const m = map.current;
    const g = layer.current;
    if (!m || !g) return;
    g.clearLayers();
    if (!data?.venues?.length) {
      if (data?.place) m.setView([data.place.latitude, data.place.longitude], 12);
      return;
    }
    const hm = data.heatmap;
    const best = data.recommendation ?? data.least_bad;
    const bounds: L.LatLngExpression[] = [];
    data.venues.forEach((v, i) => {
      const score = hm?.scores[i]?.[hourIndex] ?? null;
      const status = hm?.status[i]?.[hourIndex] ?? null;
      const isBest = best?.venue.name === v.name;
      const color = status === 'avoid' ? '#f43f5e' : scoreColor(score);
      const marker = L.circleMarker([v.lat, v.lon], {
        radius: isBest ? 12 : 8,
        color: isBest ? '#ffffff' : color,
        weight: isBest ? 3 : 1.5,
        fillColor: color,
        fillOpacity: 0.9,
      });
      const lines = [
        `<b>${v.name}</b>`,
        score != null ? `Score at this hour: <b>${score}</b>${status === 'avoid' ? ' (avoid)' : ''}` : 'No data for this hour',
        v.travel ? `${v.travel.minutes} min / ${v.travel.km} km by road` : '',
        v.available && v.green_pct != null ? `Green cover (300 m): ${v.green_pct}%` : '',
        v.nearest_major_road ? `Major road: ${v.nearest_major_road} (${v.nearest_major_road_m} m)` : '',
        v.industry?.[0] ? `Industry: ${v.industry[0].name} (${v.industry[0].distance_m} m)` : '',
        ...(hm?.reasons[i]?.[hourIndex] ?? []).map((r) => `⚠ ${r}`),
      ].filter(Boolean);
      marker.bindPopup(lines.join('<br/>'));
      marker.addTo(g);
      bounds.push([v.lat, v.lon]);
    });
    const bestV = data.venues.find((v) => v.name === best?.venue.name);
    if (bestV) {
      L.circle([bestV.lat, bestV.lon], { radius: 300, color: '#10b981', weight: 1.5, fillOpacity: 0.06 })
        .bindTooltip('300 m green-cover buffer').addTo(g);
      L.circle([bestV.lat, bestV.lon], { radius: 250, color: '#06b6d4', weight: 1.5, dashArray: '6 6', fill: false })
        .bindTooltip('250 m road-exposure buffer').addTo(g);
      bestV.industry?.forEach((ind) => {
        if (ind.lat == null || ind.lon == null) return;
        L.marker([ind.lat, ind.lon], {
          icon: L.divIcon({ className: 'ind-icon', html: '🏭', iconSize: [18, 18], iconAnchor: [9, 9] }),
        }).bindTooltip(`${ind.name} · ${ind.distance_m} m`).addTo(g);
      });
    }
    if (data.origin) {
      L.circleMarker([data.origin.lat, data.origin.lon], { radius: 6, color: '#38bdf8', fillColor: '#38bdf8', fillOpacity: 1 })
        .bindTooltip('Start / search centre')
        .addTo(g);
      bounds.push([data.origin.lat, data.origin.lon]);
    }
    m.fitBounds(L.latLngBounds(bounds).pad(0.25), { maxZoom: 14 });
  }, [data, hourIndex]);

  const wind = data?.heatmap?.wind?.[hourIndex];
  return (
    <div className="map-wrap">
      <div ref={ref} className="map" />
      {wind?.from_deg != null && (
        <div className="wind-badge" title="Wind at the selected hour (arrow points where the air goes)">
          <span className="wind-arrow" style={{ transform: `rotate(${(wind.from_deg + 180) % 360}deg)` }}>↑</span>
          <span>{Math.round(wind.kmh ?? 0)} km/h from {Math.round(wind.from_deg)}°</span>
        </div>
      )}
      <div className="map-legend">
        <span><i style={{ background: scoreColor(85) }} /> good</span>
        <span><i style={{ background: scoreColor(60) }} /> fair</span>
        <span><i style={{ background: scoreColor(35) }} /> poor</span>
        <span><i style={{ background: '#f43f5e' }} /> avoid</span>
      </div>
    </div>
  );
};
