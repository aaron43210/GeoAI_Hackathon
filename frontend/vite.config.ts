import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev: proxy the API so the UI and API share an origin (as in production, where FastAPI serves dist/).
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://localhost:8000' } },
})
