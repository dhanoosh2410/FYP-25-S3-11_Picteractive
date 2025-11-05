import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(__dirname, '..')

export default defineConfig({
  plugins: [react()],
  server: {
    // Allow serving files from the repo root (outside client/)
    fs: { allow: [repoRoot] },
    port: 5173,
  },
  resolve: {
    alias: {
      '@': path.resolve(repoRoot, 'src'),
      // Ensure bare imports resolve to client/node_modules even when the source
      // file lives outside the client root (e.g., ../src/*.jsx)
      'react': path.resolve(__dirname, 'node_modules/react'),
      'react-dom': path.resolve(__dirname, 'node_modules/react-dom'),
      'react-dom/client': path.resolve(__dirname, 'node_modules/react-dom/client.js'),
      'react/jsx-runtime': path.resolve(__dirname, 'node_modules/react/jsx-runtime.js'),
      'react/jsx-dev-runtime': path.resolve(__dirname, 'node_modules/react/jsx-dev-runtime.js'),
      'react-router-dom': path.resolve(__dirname, 'node_modules/react-router-dom'),
      'fabric': path.resolve(__dirname, 'node_modules/fabric'),
    },
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'react-router-dom', 'fabric'],
  },
  // Load env files from repo root so VITE_* in .env is picked up
  envDir: repoRoot,
})
