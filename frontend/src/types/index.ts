// Mirrors the backend AskResponse (see docs/ARCHITECTURE.md §4).

export interface Venue {
  id: string;
  name: string;
  lat: number;
  lon: number;
  kind: string;
  osm_url: string;
  type: string;
}

export interface Travel {
  minutes: number;
  km: number;
  url: string;
  source: string;
}

export interface Pick {
  venue: Venue;
  start: string;
  end: string;
  label: string;
  score: number;
  status: 'ok' | 'caution' | 'avoid';
  concerns: string[];
  unknowns: string[];
  travel: Travel | null;
  travel_at_start_min: number | null;
  factors: Record<string, number>;
}

export interface Evidence {
  id: string;
  label: string;
  value: unknown;
  unit: string | null;
  source: string;
  url: string | null;
  kind: string;
  time?: string | null;
  venue?: string | null;
  method?: string | null;
  retrieved_at?: string | null;
}

export interface Heatmap {
  hours: string[];
  venues: string[];
  scores: (number | null)[][];
  status: (string | null)[][];
  reasons: string[][][];
  wind: ({ from_deg: number | null; kmh: number | null } | null)[];
}

export interface VenueFeatures {
  name: string;
  lat: number;
  lon: number;
  available: boolean;
  green_pct: number | null;
  nearest_major_road_m: number | null;
  nearest_major_road: string | null;
  nearest_water_m: number | null;
  nearest_water: string | null;
  industry: { name: string; distance_m: number; bearing_deg: number; lat?: number; lon?: number }[];
  travel: Travel | null;
}

export interface TimelinePoint {
  ts: string;
  apparent: number | null;
  rain_prob: number | null;
  uv: number | null;
  us_aqi: number | null;
  wind_kmh: number | null;
  wind_from: number | null;
  exposure: number | null;
  score: number;
  status: string;
  in_best: boolean;
}

export interface AdviceItem {
  id: string;
  text: string;
}

export interface AskResponse {
  status: 'ok' | 'partial' | 'not_recommended' | 'clarification_needed' | 'out_of_scope' | 'not_available' | 'error'
    | 'chat' | 'explanation';
  session_id: string;
  question: string;
  answer_markdown: string;
  answer_path?: string;
  clarification?: { question: string; field: string; options: string[] };
  intent: Record<string, unknown> & { path: string; activity?: string; profiles?: string[] };
  model: { repo: string | null; file: string | null; path_used: string; loaded: boolean };
  grounding?: { status: string; facts_checked?: number; violations?: string[]; first_violations?: string[] };
  recommendation?: Pick | null;
  least_bad?: Pick | null;
  alternatives?: Pick[];
  alt_time?: Pick | null;
  compare?: { day: string; best: Pick | null; gated_windows: number }[];
  avoid?: { start: string; end: string; reasons: string[]; evidence: string }[];
  advice?: { carry: AdviceItem[]; watch_for: AdviceItem[] };
  heatmap?: Heatmap;
  venues?: VenueFeatures[];
  timeline?: TimelinePoint[];
  evidence?: Evidence[];
  notes?: string[];
  limitations?: string[];
  sources: string[];
  place?: { name: string; display_name: string; latitude: number; longitude: number };
  origin?: { lat: number; lon: number };
  resolved_windows?: { day: string; start: string; end: string; label: string }[];
  timings_ms: Record<string, number>;
  errors?: string[];
}

export interface StageEvent {
  name: string;
  message: string;
}
