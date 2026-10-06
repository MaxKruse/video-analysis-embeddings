export interface Window {
  start: number;
  end: number;
}

/**
 * Sliding windows over a duration. The final window is widened instead of
 * leaving a runt window, so no clip is shorter than half the window length.
 */
export function planWindows(duration: number, clip: number, overlap: number): Window[] {
  if (duration <= 0) return [];
  if (duration <= clip) return [{ start: 0, end: duration }];

  const step = clip - overlap;
  const out: Window[] = [];
  for (let start = 0; start < duration; start += step) {
    const end = Math.min(start + clip, duration);
    const last = out[out.length - 1];
    if (last && end - start < clip / 2) {
      last.end = duration;
      break;
    }
    out.push({ start, end });
    if (end >= duration) break;
  }
  return out;
}

export function fmtTime(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${pad(m)}:${pad(sec)}`;
}

export function fmtRange(w: Window): string {
  return `${fmtTime(w.start)}-${fmtTime(w.end)}`;
}
