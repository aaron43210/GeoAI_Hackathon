// Single sequential scale for scores (0 poor → 100 good), shared by map and heatmap.
export function scoreColor(score: number | null | undefined): string {
  if (score == null) return '#334155';
  const s = Math.max(0, Math.min(100, score)) / 100;
  // amber (poor) → teal (good), perceptually ordered
  const stops = [
    [0.0, [180, 83, 9]],
    [0.5, [234, 179, 8]],
    [0.75, [132, 204, 22]],
    [1.0, [16, 185, 129]],
  ] as const;
  for (let i = 1; i < stops.length; i++) {
    const [p1, c1] = stops[i - 1];
    const [p2, c2] = stops[i];
    if (s <= p2) {
      const t = (s - p1) / (p2 - p1);
      const c = c1.map((v, k) => Math.round(v + (c2[k] - v) * t));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return 'rgb(16,185,129)';
}

export function hourLabel(ts: string): string {
  const d = new Date(ts);
  return d.toLocaleString('en-IN', { weekday: 'short', hour: '2-digit', minute: '2-digit', hour12: false });
}
