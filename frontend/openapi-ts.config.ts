import { defineConfig } from '@hey-api/openapi-ts';

export default defineConfig({
  input: './openapi.json',
  output: 'src/lib/client',
  plugins: [
    { name: '@hey-api/client-fetch', runtimeConfigPath: './src/hey-api.ts' },
    '@tanstack/react-query',
  ],
});