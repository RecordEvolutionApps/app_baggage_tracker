import nodeResolve from '@rollup/plugin-node-resolve';
import babel from '@rollup/plugin-babel';
import { importMetaAssets } from '@web/rollup-plugin-import-meta-assets';
import esbuild from 'rollup-plugin-esbuild';
import typescript from '@rollup/plugin-typescript';

export default {
  input: 'src/camera-shell.ts',
  output: {
    entryFileNames: '[hash].js',
    chunkFileNames: '[hash].js',
    assetFileNames: '[hash][extname]',
    format: 'es',
    dir: 'dist',
  },
  watch: {
    include: 'src/**/*'
  },
  preserveEntrySignatures: false,

  plugins: [
    /** Resolve bare module imports */
    typescript({
      tsconfig: './tsconfig.json',
    }),
    nodeResolve(),
    /** Minify JS, compile JS to a lower language target */
    esbuild({
      minify: true,
      target: ['chrome64', 'firefox67', 'safari11.1'],
    }),    
    /** Bundle assets references via import.meta.url */
    importMetaAssets(),
    /** Minify html and css tagged template literals */
    babel({babelHelpers: 'bundled'}),
  ],
};
