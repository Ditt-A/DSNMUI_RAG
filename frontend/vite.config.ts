import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import {fileURLToPath} from 'node:url';
import {defineConfig, loadEnv} from 'vite';

export default defineConfig(({mode}) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = process.env.RAG_API_URL || env.RAG_API_URL || 'http://127.0.0.1:8300';
  const apiKey = process.env.RAG_API_KEY || env.RAG_API_KEY;
  const proxy = {
    '/rag': {
      target,
      changeOrigin: true,
      headers: apiKey ? {'x-api-key': apiKey} : {},
      proxyTimeout: 300_000,
    },
  };

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url)),
      },
    },
    server: {
      port: 3000,
      strictPort: true,
      proxy,
    },
    preview: {port: 3000, strictPort: true, proxy},
  };
});
