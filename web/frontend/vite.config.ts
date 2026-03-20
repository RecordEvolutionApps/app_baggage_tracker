import { defineConfig } from 'vite';
import { nodePolyfills } from 'vite-plugin-node-polyfills';

export default defineConfig({
  base: '/',
  plugins: [
    nodePolyfills({
      include: ['events', 'util', 'stream', 'buffer', 'process', 'assert'],
    }),
  ],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    strictPort: true,
    host: true,
    proxy: {
      '/streams': 'http://localhost:1100',
      '/cameras': 'http://localhost:1100',
      '/api': 'http://localhost:1100',
    },
  },
});
