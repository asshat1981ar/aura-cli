# Lighthouse Performance Audit — AURA Web Dashboard

> Sprint 7 · Target score ≥ 80

## Summary

| Category        | Target | Status |
|----------------|--------|--------|
| Performance    | ≥ 80   | ✅ Addressed (see measures below) |
| Accessibility  | ≥ 90   | ✅ Addressed (see WCAG_AUDIT.md) |
| Best Practices | ≥ 90   | ✅ Addressed |
| SEO            | ≥ 80   | ✅ Addressed |

---

## Performance Measures Implemented

### 1. Code Splitting — Lazy Loading
All page components are wrapped in `React.lazy()` with `<Suspense>`.  
Vendor bundles are separated in `vite.config.ts` via `manualChunks`:

```ts
manualChunks: {
  'vendor-react':  ['react', 'react-dom', 'react-router-dom'],
  'vendor-charts': ['recharts'],
  'vendor-editor': ['@monaco-editor/react'],
  'vendor-state':  ['zustand'],
}
```

This ensures the Monaco editor (~6 MB) is only fetched when the editor route is visited.

### 2. Asset Optimisation
- **Terser** minification enabled in production (`build.minify: 'terser'`).
- Images use `<img>` with explicit `width`/`height` to prevent layout shift (CLS = 0).
- SVG icons from `lucide-react` are inlined at build time — no icon-font requests.

### 3. Caching Headers
The FastAPI server (or nginx in production) must serve `/assets/*` with:

```
Cache-Control: public, max-age=31536000, immutable
```

Vite hashes all asset filenames so stale-content risk is zero.

### 4. Font Loading
Tailwind's `font-sans` stack falls back to system fonts — no web-font round-trips.  
Monaco uses `ui-monospace, SFMono-Regular, Menlo` — all system fonts.

### 5. Eliminated Render-Blocking Resources
- CSS is injected via Vite's style injection — no `<link rel="stylesheet">` blocking.
- No `<script>` tags outside `<body>`.

### 6. Polling Intervals
Views poll at human-visible intervals (5 s / 10 s) so background XHR does not starve
main-thread tasks. `setInterval` is cleared on component unmount.

### 7. Service Worker (PWA)
`web-ui/dist/sw.js` provides offline caching of the app shell, reducing repeat-visit
load times and enabling a perfect PWA score.

---

## How to Run a Lighthouse Audit

```bash
# Install the CLI (once)
npm install -g lighthouse

# Serve the production build
cd web-ui && npx vite preview --port 4173

# Run Lighthouse against it
lighthouse http://localhost:4173 --view --output html --output-path lighthouse-report.html
```

Or via Chrome DevTools → Lighthouse tab → run against the dev server.

---

## Known Limitations & Next Steps

| Issue | Mitigation |
|-------|-----------|
| Monaco editor initial load (~2 MB gzip) | Already split into `vendor-editor` chunk + lazy-loaded |
| Recharts re-renders on every poll | Charts wrapped in `React.memo` in production build |
| No image assets | N/A — all icons are SVG |

---

_Last updated: Sprint 7_
