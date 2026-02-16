import { css } from 'lit';

export const mainStyles = css`
  md-outlined-select,
  md-outlined-text-field,
  md-list-item,
  md-list,
  md-dialog,
  md-text-button,
  md-elevated-button {
    --my-brand-font: sans-serif;
    --md-ref-typeface-brand: var(--my-brand-font);
    --md-ref-typeface-plain: var(--my-brand-font);
    --md-outlined-text-field-input-text-font: sans-serif;
    --md-sys-typescale-body-font: var(--my-brand-font);
    --md-sys-typescale-display-font: var(--my-brand-font);
    --md-sys-typescale-headline-font: var(--my-brand-font);
    --my-sys-typescale-label-font: var(--my-brand-font);
    --md-sys-typescale-title-font: var(--my-brand-font);
  }

  md-outlined-select::part(menu) {
    --md-menu-container-color: #dde5eb;
  }

  md-outlined-select {
    --md-sys-color-secondary: #78889496;
    --md-sys-color-on-secondary: #78889496;
    --md-sys-color-secondary-container: #78889496;
    --md-sys-color-on-secondary-container: #78889496;
    --md-ripple-hover-color: #788894;
    --md-menu-item-hover-state-layer-color: #788894;
    --md-sys-color-on-surface: #788894;
    --md-ripple-pressed-color: #788894;
    --md-menu-item-pressed-state-layer-color: #788894;
    --md-outlined-select-text-field-focus-trailing-icon-color: #788894;
    --md-outlined-field-focus-label-text-color: #334d5c;
    --md-outlined-field-focus-outline-color: #788894;
    --md-outlined-select-text-field-input-text-placeholder-color: #788894;
    --md-outlined-select-text-field-outline-color: #788894;
    --md-outlined-select-text-field-supporting-text-color: #788894;
    --md-outlined-select-text-field-label-text-color: #334d5c;
    --md-outlined-select-text-field-input-text-color: #334d5c;
    --md-outlined-select-text-field-hover-supporting-text-color: #334d5c;
    --md-outlined-select-text-field-hover-outline-color: #788894;
    --md-outlined-select-text-field-hover-label-text-color: #334d5c;
    --md-outlined-select-text-field-hover-input-text-color: #334d5c;
    --md-outlined-select-text-field-focus-supporting-text-color: #334d5c;
    --md-outlined-select-text-field-focus-outline-color: #788894;
    --md-outlined-select-text-field-focus-label-text-color: #334d5c;
    --md-outlined-select-text-field-focus-input-text-color: #334d5c;
    --md-outlined-select-text-field-error-supporting-text-color: #334d5c;
    --md-outlined-select-text-field-error-outline-color: #788894;
    --md-outlined-select-text-field-error-label-text-color: #334d5c;
    --md-outlined-select-text-field-error-input-text-color: #334d5c;
    --md-outlined-select-text-field-error-hover-supporting-text-color: #334d5c;
    --md-outlined-select-text-field-error-hover-outline-color: #788894;
    --md-outlined-select-text-field-error-hover-label-text-color: #334d5c;
    --md-outlined-select-text-field-error-hover-input-text-color: #334d5c;
    --md-outlined-select-text-field-error-focus-supporting-text-color: #334d5c;
    --md-outlined-select-text-field-error-focus-outline-color: #788894;
    --md-outlined-select-text-field-error-focus-input-text-color: #334d5c;
    --md-outlined-select-text-field-error-focus-label-text-color: #334d5c;
    --md-outlined-select-text-field-disabled-supporting-text-color: #334d5c;
    --md-outlined-select-text-field-disabled-outline-color: #788894;
    --md-outlined-select-text-field-disabled-leading-icon-color: #788894;
    --md-outlined-select-text-field-disabled-label-text-color: #334d5c;
    --md-outlined-select-text-field-disabled-input-text-color: #334d5c;
    --md-menu-item-selected-label-text-color: #334d5c;
  }

  md-outlined-text-field {
    --md-sys-color-primary: rgb(149, 149, 149);
    --md-outlined-text-field-label-text-color: #788894;
    --md-outlined-text-field-input-text-placeholder-color: #788894;
    --md-outlined-text-field-input-text-color: #243542;
    --md-outlined-text-field-error-input-text-color: #b3261e;
    --md-outlined-text-field-hover-label-text-color: #788894;
    --md-outlined-text-field-hover-supporting-text-color: #334d5c;
    --md-outlined-text-field-hover-outline-color: #788894;
    --md-outlined-text-field-hover-input-text-color: #334d5c;
    --md-outlined-text-field-focus-supporting-text-color: #334d5c;
    --md-outlined-text-field-focus-outline-color: #788894;
    --md-outlined-text-field-focus-label-text-color: #334d5c;
    --md-outlined-text-field-focus-input-text-color: #334d5c;
  }
`;

export function getRandomColor() {
  // Generate random values for red, green, and blue channels
  const red = Math.floor(Math.random() * 256);
  const green = Math.floor(Math.random() * 256);
  const blue = Math.floor(Math.random() * 256);

  // Construct the color string in hexadecimal format
  const color =
    '#' +
    red.toString(16).padStart(2, '0') +
    green.toString(16).padStart(2, '0') +
    blue.toString(16).padStart(2, '0');

  return color;
}

export function hexToTransparent(hex: string, opacity: number) {
  // Remove the hash if it exists
  hex = hex.replace('#', '');

  // Parse the hex value into RGB components
  var r = parseInt(hex.substring(0, 2), 16);
  var g = parseInt(hex.substring(2, 4), 16);
  var b = parseInt(hex.substring(4, 6), 16);

  // Convert opacity to a value between 0 and 1
  opacity = Math.min(Math.max(opacity, 0), 1);

  // Convert the RGB values into a transparent RGBA string
  var rgba = 'rgba(' + r + ', ' + g + ', ' + b + ', ' + opacity + ')';

  return rgba;
}

export type PolygonType = 'ZONE' | 'LINE'

export type PolygonState = {
  selectedPolygonId: number | undefined;
  polygons: {
    id: number;
    label: string;
    lineColor: string;
    fillColor: string;
    committed: boolean;
    type: PolygonType;
    points: { x: number; y: number }[];
  }[];
}

export type Camera = {
  type: 'USB' | 'IP'
  name?: string
  id?: string
  camStream: string
  path?: string
  username?: string
  password?: string
  model?: string
  useSahi?: boolean
  useSmoothing?: boolean
  frameBuffer?: number
  confidence?: number
  nmsIou?: number
  sahiIou?: number
  overlapRatio?: number
  classList?: number[]
  classNames?: string[]
}

export type ModelOption = {
  id: string
  label: string
  arch?: string
  dataset?: string
  architecture?: string
  task?: string
  paper?: string
  summary?: string
  description?: string
  openVocab?: boolean
  fileSize?: number
}

export type ClassOption = {
  id: number
  name: string
}

export type CamSetup = {
  camera: Camera
  width: number
  height: number
}