import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import { repeat } from 'lit/directives/repeat.js';
import { mainStyles, CamSetup, ModelOption, ClassOption } from './utils.js';
import prettyBytes from 'pretty-bytes';
import '@material/web/chips/filter-chip.js';
import '@material/web/dialog/dialog.js';
import { MdDialog } from '@material/web/dialog/dialog.js';
import '@material/web/button/text-button.js';
import '@material/web/button/elevated-button.js';
import '@material/web/button/outlined-button.js';

@customElement('inference-setup')
export class InferenceSetup extends LitElement {
  @property({ type: String })
  declare camStream: string;

  @property({ type: Object })
  declare camSetup?: CamSetup;

  @state()
  declare models: ModelOption[];

  @state()
  declare selectedModel: string;

  @state()
  declare useSahi: boolean;

  @state()
  declare useSmoothing: boolean;

  @state()
  declare frameBuffer: number;

  @state()
  declare confidence: number;

  @state()
  declare nmsIou: number;

  @state()
  declare sahiIou: number;

  @state()
  declare overlapRatio: number;

  @state()
  declare modelFilter: string;

  @state()
  declare selectedDataset: string;

  @state()
  declare selectedArch: string;

  @state()
  declare pendingModelId: string;

  @state()
  declare activeTags: Set<string>;

  @state()
  declare availableTags: Record<string, string[]>;

  @state()
  declare availableClasses: ClassOption[];

  @state()
  declare selectedClassIds: Set<number>;

  @state()
  declare classFilter: string;

  @state()
  declare pendingClassIds: Set<number>;

  @state()
  declare classesLoading: boolean;

  @state()
  declare classesError: string;

  /** true once classList has been restored from camSetup (even if empty) */
  private hasPersistedClassList = false;

  @state()
  declare classNamesText: string;

  @state()
  declare cachedModelIds: Set<string>;

  private classNamesDebounce: ReturnType<typeof setTimeout> | null = null;

  @state()
  declare modelStatus: 'idle' | 'checking' | 'downloading' | 'applying' | 'ready' | 'error';

  @state()
  declare modelProgress: number;

  @state()
  declare modelStatusMessage: string;

  @state()
  declare backendInfo: {
    backend: string;
    model: string;
    precision: string;
    device: string;
    trt_cached: boolean;
    requested_backend: string;
    message: string;
  } | null;

  @state()
  declare trtBuildStatus: 'idle' | 'building' | 'ready' | 'error';

  @state()
  declare trtBuildProgress: number;

  @state()
  declare trtBuildMessage: string;

  private backendPollTimer: ReturnType<typeof setInterval> | null = null;
  private fastPollTimer: ReturnType<typeof setInterval> | null = null;
  private modelSizeTimer: ReturnType<typeof setTimeout> | null = null;
  private modelSizeAttempts = 0;

  classDialog?: MdDialog;
  modelDialog?: MdDialog;
  private basepath = window.location.protocol + '//' + window.location.host;

  constructor() {
    super();
    this.camStream = 'frontCam';
    this.models = [];
    this.selectedModel = '';
    this.useSahi = false;
    this.useSmoothing = false;
    this.frameBuffer = 64;
    this.confidence = 0.1;
    this.nmsIou = 0.5;
    this.sahiIou = 0.5;
    this.overlapRatio = 0.2;
    this.modelFilter = '';
    this.selectedDataset = '';
    this.selectedArch = '';
    this.pendingModelId = this.selectedModel;
    this.activeTags = new Set();
    this.availableTags = {};
    this.availableClasses = [];
    this.selectedClassIds = new Set();
    this.classFilter = '';
    this.pendingClassIds = new Set();
    this.classesLoading = false;
    this.classesError = '';
    this.classNamesText = '';
    this.cachedModelIds = new Set();
    this.modelStatus = 'idle';
    this.modelProgress = 0;
    this.modelStatusMessage = '';
    this.backendInfo = null;
    this.trtBuildStatus = 'idle';
    this.trtBuildProgress = 0;
    this.trtBuildMessage = '';
  }

  static styles = [
    mainStyles,
    css`
      :host {
        --md-sys-color-primary: #002e6a;
        --md-sys-color-on-primary: #ffffff;
      }

      .control {
        display: flex;
        flex-direction: column;
        gap: 8px;
        padding: 8px;
        border-radius: 4px;
        border: 1px solid #aaa;
      }

      .model-picker {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }

      .model-summary-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #5e5f61;
        font-family: sans-serif;
        font-size: 0.875rem;
        padding: 2px 0;
      }

      .model-button {
        width: 100%;
        --md-elevated-button-container-color: #eceff1;
        --md-elevated-button-label-text-color: #5e5f61;
      }

      .model-filter {
        width: 100%;
        box-sizing: border-box;
        padding: 8px 12px;
        border: 1px solid #aaa;
        border-radius: 4px;
        font-size: 0.875rem;
        font-family: sans-serif;
        color: #334d5c;
        outline: none;
      }

      .model-filter:focus {
        border-color: #002e6a;
      }

      .model-filter::placeholder {
        color: #999;
      }

      .model-select {
        width: 100%;
        box-sizing: border-box;
        padding: 8px 12px;
        border: 1px solid #aaa;
        border-radius: 4px;
        font-size: 0.875rem;
        font-family: sans-serif;
        color: #334d5c;
        background: #fff;
        outline: none;
        cursor: pointer;
      }

      .model-browser {
        display: flex;
        gap: 8px;
        flex: 1;
        min-height: 0;
        overflow: hidden;
      }

      .model-browser .tag-filter-sidebar {
        display: flex;
        flex-direction: column;
        gap: 3px;
        min-width: 180px;
        max-width: 240px;
        overflow-y: auto;
        overflow-x: hidden;
        padding-right: 4px;
      }

      .model-browser .tag-filter-sidebar .tag-dim-label {
        padding: 6px 2px 2px;
      }

      .model-panel {
        border: 1px solid #d0d0d0;
        border-radius: 6px;
        background: #fff;
        display: flex;
        flex-direction: column;
        flex: 1;
        overflow: hidden;
      }

      .model-panel-header {
        padding: 6px 8px;
        font-family: sans-serif;
        font-size: 0.75rem;
        color: #5e5f61;
        background: #f3f5f7;
        border-bottom: 1px solid #d0d0d0;
        flex-shrink: 0;
      }

      .scroller {
        flex: 1;
        overflow-y: auto;
      }

      .model-details {
        border: 1px solid #d0d0d0;
        border-radius: 6px;
        background: #fff;
        padding: 8px 10px;
        font-family: sans-serif;
        font-size: 0.85rem;
        color: #334d5c;
        margin-top: 8px;
      }

      .model-details h4 {
        margin: 0 0 6px;
        font-size: 0.85rem;
        color: #5e5f61;
      }

      .model-details a {
        color: #002e6a;
        text-decoration: none;
      }

      .model-details a:hover {
        text-decoration: underline;
      }

      .model-item {
        width: 100%;
        text-align: left;
        border: none;
        background: transparent;
        padding: 6px 8px;
        font-family: sans-serif;
        font-size: 0.85rem;
        color: #334d5c;
        cursor: pointer;
      }

      .model-item:hover {
        background: #eef2f6;
      }

      .model-item.active {
        background: #e7f0fb;
        color: #002e6a;
        font-weight: 600;
      }

      .model-item-count {
        float: right;
        color: #999;
        font-size: 0.75rem;
      }

      .model-dropdown {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }

      .model-group {
        border: 1px solid #d0d0d0;
        border-radius: 6px;
        background: #f7f7f7;
        overflow: hidden;
      }

      .model-group summary {
        list-style: none;
        cursor: pointer;
        padding: 8px 10px;
        font-family: sans-serif;
        font-size: 0.85rem;
        color: #334d5c;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }

      .model-group summary::-webkit-details-marker {
        display: none;
      }

      .model-group[open] summary {
        background: #eef2f6;
      }

      .model-options {
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding: 6px 8px 10px;
        background: #fff;
      }

      .model-option {
        text-align: left;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        padding: 6px 8px;
        background: #fff;
        font-family: sans-serif;
        font-size: 0.85rem;
        color: #334d5c;
        cursor: pointer;
      }

      .model-option:hover {
        border-color: #002e6a;
      }

      .model-option.selected {
        background: #e7f0fb;
        border-color: #002e6a;
        color: #002e6a;
      }

      .model-select:focus {
        border-color: #002e6a;
      }

      .model-count {
        font-size: 0.75rem;
        color: #999;
        font-family: sans-serif;
      }

      .sahi-toggle {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #5e5f61;
        font-family: sans-serif;
        font-size: 0.875rem;
        cursor: pointer;
        padding: 4px 0;
      }

      .sahi-toggle input[type='checkbox'] {
        width: 18px;
        height: 18px;
        accent-color: #002e6a;
        cursor: pointer;
      }

      .frame-buffer-row {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 0 0 4px;
      }

      .fb-label {
        color: #5e5f61;
        font-family: sans-serif;
        font-size: 0.875rem;
        white-space: nowrap;
        display: inline-block;
        min-width: 130px;
      }

      .sahi-subsettings {
        margin-left: 26px;
        padding-left: 10px;
        border-left: 2px solid #d0d0d0;
        display: flex;
        flex-direction: column;
        gap: 2px;
      }

      .fb-input {
        width: 72px;
        padding: 4px 8px;
        border: 1px solid #aaa;
        border-radius: 4px;
        font-size: 0.875rem;
        font-family: sans-serif;
        color: #334d5c;
      }

      .class-filter-section {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }

      .class-summary-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #5e5f61;
        font-family: sans-serif;
        font-size: 0.875rem;
        padding: 2px 0;
      }

      .class-summary-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
      }

      .class-summary-chips md-filter-chip {
        --md-filter-chip-container-color: #f5f5f5;
        --md-filter-chip-selected-container-color: #f5f5f5;
        --md-filter-chip-label-text-color: #5e5f61;
        --md-filter-chip-selected-label-text-color: #5e5f61;
        --md-filter-chip-outline-color: #d0d0d0;
        --md-filter-chip-label-text-font: sans-serif;
        pointer-events: none;
      }

      .class-summary-empty {
        font-size: 0.75rem;
        color: #999;
        font-family: sans-serif;
      }

      .class-filter-search {
        width: 100%;
        box-sizing: border-box;
        padding: 6px 10px;
        border: 1px solid #aaa;
        border-radius: 4px;
        font-size: 0.8rem;
        font-family: sans-serif;
        color: #334d5c;
        outline: none;
      }

      .class-filter-search:focus {
        border-color: #002e6a;
      }

      .class-filter-search::placeholder {
        color: #999;
      }

      #class-dialog::part(container) {
        display: flex;
        flex-direction: column;
      }

      #class-dialog::part(content) {
        display: flex;
        flex-direction: column;
        flex: 1;
        overflow: hidden;
      }

      form {
        position: absolute;
        display: flex;
        flex-direction: column;
        height: 100%;
        width: 100%;
        gap: 12px;
        overflow: hidden;
        padding: 12px;
      }

      .class-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        flex: 1;
        align-content: flex-start;
        overflow-y: auto;
      }

      .class-chips md-filter-chip {
        --md-filter-chip-label-text-size: 0.75rem;
        --md-filter-chip-container-height: 28px;
        --md-filter-chip-container-shape: 14px;
        --md-filter-chip-selected-container-color: #dbe7f5;
        --md-filter-chip-selected-label-text-color: #002e6a;
        --md-filter-chip-label-text-color: #5e5f61;
        --md-filter-chip-outline-color: #ccc;
        --md-filter-chip-label-text-font: sans-serif;
      }

      .class-summary-chips md-filter-chip {
        --md-filter-chip-label-text-font: sans-serif;
      }

      .class-bulk-actions {
        display: flex;
        gap: 8px;
        font-size: 0.75rem;
        font-family: sans-serif;
      }

      .class-bulk-actions span {
        cursor: pointer;
        color: #002e6a;
        text-decoration: underline;
      }

      .class-bulk-actions span:hover {
        color: #004aad;
      }

      .class-count-label {
        font-size: 0.75rem;
        color: #999;
        font-family: sans-serif;
      }

      .dialog {
        --md-dialog-container-color: #fff;
        --md-dialog-headline-color: #5e5f61;
        --md-dialog-supporting-text-color: #5e5f61;
        min-width: 80vw !important;
        max-width: 90vw !important;
        min-height: 80vh !important;
        max-height: 90vh !important;
      }

      .dialog form[slot="content"] {
        display: flex;
        flex-direction: column;
        min-height: 60vh;
      }

      .model-status-bar {
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding: 8px;
        border-radius: 4px;
        background: #f5f7fa;
        border: 1px solid #ddd;
        font-family: sans-serif;
        font-size: 0.8rem;
        color: #5e5f61;
      }

      .model-status-bar.error {
        background: #fff0f0;
        border-color: #e88;
        color: #b33;
      }

      .model-status-bar.ready {
        background: #f0fff4;
        border-color: #8c8;
        color: #363;
      }

      .progress-track {
        width: 100%;
        height: 6px;
        background: #ddd;
        border-radius: 3px;
        overflow: hidden;
      }

      .progress-fill {
        height: 100%;
        background: #002e6a;
        border-radius: 3px;
        transition: width 0.3s ease;
      }

      .model-select:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }

      .model-filter:disabled {
        opacity: 0.5;
      }

      .ov-badge {
        display: inline-block;
        font-size: 0.6rem;
        font-weight: 600;
        color: #7c3aed;
        background: #ede9fe;
        border: 1px solid #c4b5fd;
        border-radius: 3px;
        padding: 1px 4px;
        margin-left: 4px;
        vertical-align: middle;
        line-height: 1.2;
      }

      .cached-badge {
        display: inline-block;
        font-size: 0.6rem;
        font-weight: 600;
        color: #0d6939;
        background: #d1fae5;
        border: 1px solid #6ee7b7;
        border-radius: 3px;
        padding: 1px 4px;
        margin-left: 4px;
        vertical-align: middle;
        line-height: 1.2;
      }

      .delete-cache-btn {
        background: none;
        border: none;
        cursor: pointer;
        padding: 2px 4px;
        font-size: 0.75rem;
        color: #999;
        border-radius: 3px;
        line-height: 1;
        flex-shrink: 0;
      }
      .delete-cache-btn:hover {
        color: #c53030;
        background: #fee2e2;
      }

      .clear-cache-btn {
        font-family: sans-serif;
        font-size: 0.7rem;
        color: #b33;
        background: #fff5f5;
        border: 1px solid #fcc;
        border-radius: 4px;
        padding: 3px 8px;
        cursor: pointer;
        margin-left: auto;
      }
      .clear-cache-btn:hover {
        background: #fee2e2;
        border-color: #f99;
      }

      /* ── Tag filter chips ── */
      .tag-filter-bar {
        display: flex;
        gap: 4px;
        padding: 0 0 6px;
      }

      .tag-dim-label {
        font-family: sans-serif;
        font-size: 0.65rem;
        font-weight: 600;
        color: #788894;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 4px 2px 2px;
        width: 100%;
      }

      .tag-chip {
        display: inline-flex;
        align-items: center;
        gap: 3px;
        border: 1px solid #c0cad4;
        border-radius: 12px;
        padding: 2px 8px;
        font-family: sans-serif;
        font-size: 0.72rem;
        color: #455a64;
        background: #f5f7fa;
        cursor: pointer;
        user-select: none;
        transition: all 0.15s ease;
        line-height: 1.4;
        max-width: 100%;
        min-height: 24px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .tag-chip:hover {
        border-color: #002e6a;
        background: #eef2f6;
      }

      .tag-chip.active {
        background: #e7f0fb;
        border-color: #002e6a;
        color: #002e6a;
        font-weight: 600;
      }

      .tag-chip .tag-count {
        font-size: 0.62rem;
        color: #999;
        font-weight: 400;
      }

      .tag-chip.active .tag-count {
        color: #002e6a;
      }

      /* ── Output type badges on model items ── */
      .output-badge {
        display: inline-block;
        font-size: 0.55rem;
        font-weight: 600;
        border-radius: 3px;
        padding: 1px 4px;
        margin-left: 3px;
        vertical-align: middle;
        line-height: 1.2;
      }

      .output-badge.bbox {
        color: #1565c0;
        background: #e3f2fd;
        border: 1px solid #90caf9;
      }

      .output-badge.mask {
        color: #6a1b9a;
        background: #f3e5f5;
        border: 1px solid #ce93d8;
      }

      .output-badge.keypoints {
        color: #e65100;
        background: #fff3e0;
        border: 1px solid #ffcc80;
      }

      .speed-indicator {
        display: inline-flex;
        gap: 1px;
        margin-left: 4px;
        vertical-align: middle;
      }

      .speed-indicator .speed-bar {
        width: 3px;
        background: #c0cad4;
        border-radius: 1px;
      }

      .speed-indicator .speed-bar.filled {
        background: #002e6a;
      }

      .speed-indicator .speed-bar:nth-child(1) { height: 5px; }
      .speed-indicator .speed-bar:nth-child(2) { height: 8px; }
      .speed-indicator .speed-bar:nth-child(3) { height: 11px; }

      .tag-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 3px;
        margin-top: 4px;
      }

      .tag-badge-sm {
        display: inline-block;
        font-size: 0.6rem;
        border-radius: 3px;
        padding: 1px 4px;
        line-height: 1.3;
        border: 1px solid #d0d7de;
        color: #57606a;
        background: #f6f8fa;
      }

      .backend-badge {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 4px 6px;
        padding: 6px 10px;
        border-radius: 4px;
        font-family: sans-serif;
        font-size: 0.8rem;
        border: 1px solid #d0d0d0;
        background: #f5f7fa;
        color: #5e5f61;
      }

      .backend-badge .badge-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
      }

      .backend-badge.tensorrt {
        background: #eef6ee;
        border-color: #8bc48b;
        color: #2d5a2d;
      }

      .backend-badge.tensorrt .badge-dot {
        background: #4caf50;
      }

      .backend-badge.mmdet {
        background: #f5f7fa;
        border-color: #b0bec5;
        color: #455a64;
      }

      .backend-badge.mmdet .badge-dot {
        background: #78909c;
      }

      .backend-badge.fallback {
        background: #fff8e1;
        border-color: #ffcc02;
        color: #7a6400;
      }

      .backend-badge.fallback .badge-dot {
        background: #ff9800;
      }

      .backend-badge.unknown {
        background: #fafafa;
        border-color: #e0e0e0;
        color: #999;
      }

      .backend-badge.unknown .badge-dot {
        background: #bbb;
      }

      .backend-badge .badge-detail {
        font-size: 0.7rem;
        color: inherit;
        opacity: 0.75;
      }

      .backend-badge .running-model {
        flex-basis: 100%;
        font-size: 0.75rem;
        font-weight: 500;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        padding-left: 14px;
      }

      .backend-badge.model-mismatch {
        border-color: #f0ad4e;
        background: #fff9ed;
      }

      .backend-badge.model-mismatch .badge-dot {
        background: #f0ad4e;
        animation: pulse-dot 1.2s ease-in-out infinite;
      }

      @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
      }

      .backend-badge .model-switching {
        flex-basis: 100%;
        font-size: 0.7rem;
        color: #b37a00;
        font-style: italic;
        padding-left: 14px;
      }

      .trt-build-btn {
        margin-left: auto;
        padding: 2px 8px;
        border: 1px solid #8bc48b;
        border-radius: 4px;
        background: #eef6ee;
        color: #2d5a2d;
        font-size: 0.7rem;
        font-family: sans-serif;
        cursor: pointer;
        white-space: nowrap;
      }

      .trt-build-btn:hover {
        background: #d4eed4;
        border-color: #4caf50;
      }

      .trt-build-btn:disabled {
        opacity: 0.6;
        cursor: not-allowed;
      }

      .trt-build-status {
        flex-basis: 100%;
        font-size: 0.7rem;
        padding-left: 14px;
        color: #5e5f61;
      }

      .trt-build-status.error {
        color: #b33;
      }

      .trt-build-status.ready {
        color: #2d5a2d;
      }

      .trt-progress-track {
        flex-basis: 100%;
        height: 3px;
        background: #e0e0e0;
        border-radius: 2px;
        margin-left: 14px;
        overflow: hidden;
      }

      .trt-progress-fill {
        height: 100%;
        background: #4caf50;
        transition: width 0.3s;
      }
    `,
  ];

  protected firstUpdated(
    _changedProperties: PropertyValueMap<any> | Map<PropertyKey, unknown>,
  ): void {
    this.classDialog = this.shadowRoot?.getElementById('class-dialog') as MdDialog;
    this.modelDialog = this.shadowRoot?.getElementById('model-dialog') as MdDialog;
    this.fetchModels();
    this.fetchBackendStatus();
    this.backendPollTimer = setInterval(() => this.fetchBackendStatus(), 10000);
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    if (this.backendPollTimer) {
      clearInterval(this.backendPollTimer);
      this.backendPollTimer = null;
    }
    if (this.fastPollTimer) {
      clearInterval(this.fastPollTimer);
      this.fastPollTimer = null;
    }
    if (this.modelSizeTimer) {
      clearTimeout(this.modelSizeTimer);
      this.modelSizeTimer = null;
    }
  }

  private async fetchBackendStatus() {
    if (!this.camStream) return;
    try {
      const res = await fetch(
        `${this.basepath}/cameras/streams/${encodeURIComponent(this.camStream)}/backend`,
        { signal: AbortSignal.timeout(5000) },
      );
      if (res.ok) {
        this.backendInfo = await res.json();
        // Stop fast polling once the running model matches the selected model
        if (this.fastPollTimer && !this._isModelMismatch) {
          clearInterval(this.fastPollTimer);
          this.fastPollTimer = null;
        }
      }
    } catch {
      // Silently ignore — badge stays as-is
    }
  }

  /** Poll backend status every 3s until the running model catches up to the selected one. */
  private _startFastBackendPoll() {
    if (this.fastPollTimer) clearInterval(this.fastPollTimer);
    this.fastPollTimer = setInterval(() => this.fetchBackendStatus(), 3000);
    // Safety: stop after 2 minutes regardless
    setTimeout(() => {
      if (this.fastPollTimer) {
        clearInterval(this.fastPollTimer);
        this.fastPollTimer = null;
      }
    }, 120_000);
  }

  private get _isModelMismatch(): boolean {
    if (!this.backendInfo?.model || !this.selectedModel) return false;
    return this.backendInfo.model !== this.selectedModel && this.selectedModel !== 'none';
  }

  private get _backendBadgeClass(): string {
    if (!this.backendInfo) return 'unknown';
    const { backend, requested_backend } = this.backendInfo;
    let cls = 'unknown';
    if (backend === 'tensorrt') cls = 'tensorrt';
    else if (backend === 'mmdet' && requested_backend === 'tensorrt') cls = 'fallback';
    else if (backend === 'mmdet') cls = 'mmdet';
    if (this._isModelMismatch) cls += ' model-mismatch';
    return cls;
  }

  private get _backendLabel(): string {
    if (!this.backendInfo) return 'Loading…';
    const { backend, precision, trt_cached, requested_backend } = this.backendInfo;
    if (backend === 'tensorrt') {
      const cached = trt_cached ? ' (cached)' : '';
      return `TensorRT ${precision.toUpperCase()}${cached}`;
    }
    if (backend === 'mmdet' && requested_backend === 'tensorrt') {
      return `PyTorch FP32 (TRT fallback)`;
    }
    if (backend === 'mmdet') return `PyTorch FP32`;
    return this.backendInfo.message || 'Unknown';
  }

  private get _runningModelName(): string {
    if (!this.backendInfo?.model) return '';
    return this.backendInfo.model;
  }

  private get _backendDetail(): string {
    if (!this.backendInfo) return '';
    return this.backendInfo.device || '';
  }

  /** Show the "Build TensorRT" button when running mmdet on a CUDA device (TRT available but not built). */
  private get _canBuildTrt(): boolean {
    if (!this.backendInfo) return false;
    if (this.trtBuildStatus === 'ready') return false;
    const { backend, device } = this.backendInfo;
    // Only show when running PyTorch (mmdet) on a CUDA device — TRT is possible but not built
    if (backend === 'tensorrt') return false;
    if (!device || !device.includes('cuda')) return false;
    if (this.selectedModel === 'none' || !this.selectedModel) return false;
    return true;
  }

  private async buildTrt() {
    if (this.trtBuildStatus === 'building') return;
    const model = this.selectedModel;
    if (!model || model === 'none') return;

    this.trtBuildStatus = 'building';
    this.trtBuildProgress = 0;
    this.trtBuildMessage = 'Starting TensorRT build...';

    try {
      const res = await fetch(`${this.basepath}/cameras/models/build-trt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model }),
      });

      if (!res.ok || !res.body) {
        this.trtBuildStatus = 'error';
        this.trtBuildMessage = `Failed to start TRT build (HTTP ${res.status})`;
        return;
      }

      // Read SSE stream
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        let updated = false;
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6));
              this.trtBuildStatus = event.status === 'building' ? 'building' : event.status;
              this.trtBuildProgress = event.progress ?? 0;
              this.trtBuildMessage = event.message ?? '';
              updated = true;
            } catch { /* ignore parse errors */ }
          }
        }
        if (updated) {
          await new Promise(r => requestAnimationFrame(r));
        }
      }

      if (this.trtBuildStatus === 'ready') {
        // TRT engine built — restart stream to pick it up
        this.trtBuildMessage = 'TensorRT engine built! Restarting stream...';
        await this.applyModel(model);
        // Refresh backend status
        setTimeout(() => this.fetchBackendStatus(), 3000);
        this._startFastBackendPoll();
        // Auto-clear after a few seconds
        setTimeout(() => {
          if (this.trtBuildStatus === 'ready') {
            this.trtBuildStatus = 'idle';
          }
        }, 6000);
      }
    } catch (err) {
      console.error('TRT build failed', err);
      this.trtBuildStatus = 'error';
      this.trtBuildMessage = `Error: ${err}`;
    }
  }

  update(
    _changedProperties: PropertyValueMap<any> | Map<PropertyKey, unknown>,
  ) {
    if (_changedProperties.has('camSetup') && this.camSetup) {
      if (this.camSetup.camera?.model) {
        this.selectedModel = this.camSetup.camera.model;
      }
      this.useSahi = this.camSetup.camera?.useSahi ?? true;
      this.useSmoothing = this.camSetup.camera?.useSmoothing ?? true;
      this.frameBuffer = this.camSetup.camera?.frameBuffer ?? 64;
      this.confidence = this.camSetup.camera?.confidence ?? 0.1;
      this.nmsIou = this.camSetup.camera?.nmsIou ?? 0.5;
      this.sahiIou = this.camSetup.camera?.sahiIou ?? 0.5;
      this.overlapRatio = this.camSetup.camera?.overlapRatio ?? 0.2;
      // Restore persisted class selection
      if (this.camSetup.camera?.classList && this.camSetup.camera.classList.length > 0) {
        this.selectedClassIds = new Set(this.camSetup.camera.classList);
        this.hasPersistedClassList = true;
      } else if (this.camSetup.camera?.classList !== undefined) {
        // Explicitly clear if classList exists but is empty
        this.selectedClassIds = new Set();
        this.hasPersistedClassList = true;
      }
      // Restore persisted open-vocab class names
      if (this.camSetup.camera?.classNames && this.camSetup.camera.classNames.length > 0) {
        this.classNamesText = this.camSetup.camera.classNames.join(', ');
      }
    }
    super.update(_changedProperties);
  }

  private async fetchModels() {
    try {
      const res = await fetch(`${this.basepath}/cameras/models`);
      if (res.ok) this.models = await res.json();
    } catch (err) {
      console.error('Failed to fetch models', err);
    }
    this.modelSizeAttempts = 0;
    this.scheduleModelSizeRefresh();
    this.fetchTags();
    this.fetchCachedModels();
    // Set current model from camSetup if available
    if (this.camSetup?.camera?.model) {
      this.selectedModel = this.camSetup.camera.model;
    }
    if (this.selectedModel) {
      this.fetchModelClasses(this.selectedModel);
      this.syncModelSelection();
    }
  }

  private async fetchTags() {
    try {
      const res = await fetch(`${this.basepath}/cameras/models/tags`);
      if (res.ok) this.availableTags = await res.json();
    } catch (err) {
      console.error('Failed to fetch model tags', err);
    }
  }

  private async fetchCachedModels() {
    try {
      const res = await fetch(`${this.basepath}/cameras/models/cache`);
      if (res.ok) {
        const data = await res.json();
        this.cachedModelIds = new Set((data as any[]).map((c: any) => c.model));
      }
    } catch (err) {
      console.error('Failed to fetch cached models', err);
    }
  }

  private async deleteCachedModel(modelId: string, ev: Event) {
    ev.stopPropagation();
    try {
      const res = await fetch(`${this.basepath}/cameras/models/${encodeURIComponent(modelId)}/cache`, { method: 'DELETE' });
      if (res.ok) {
        const next = new Set(this.cachedModelIds);
        next.delete(modelId);
        this.cachedModelIds = next;
      }
    } catch (err) {
      console.error('Failed to delete cached model', err);
    }
  }

  private async clearAllCache() {
    if (!confirm('Delete all cached model files? This will free disk space but models will need to be re-downloaded.')) return;
    try {
      const res = await fetch(`${this.basepath}/cameras/models/cache`, { method: 'DELETE' });
      if (res.ok) {
        this.cachedModelIds = new Set();
      }
    } catch (err) {
      console.error('Failed to clear cache', err);
    }
  }

  private scheduleModelSizeRefresh(delayMs: number = 2000) {
    if (this.modelSizeTimer) {
      clearTimeout(this.modelSizeTimer);
    }
    const missingSizes = this.models.some(m => m.id !== 'none' && m.fileSize === undefined);
    if (!missingSizes || this.modelSizeAttempts >= 6) return;
    this.modelSizeTimer = setTimeout(() => this.fetchModelSizes(), delayMs);
  }

  private async fetchModelSizes() {
    this.modelSizeAttempts += 1;
    try {
      const res = await fetch(`${this.basepath}/cameras/models`);
      if (!res.ok) {
        this.scheduleModelSizeRefresh(3000);
        return;
      }
      const fresh = await res.json();
      if (!Array.isArray(fresh)) {
        this.scheduleModelSizeRefresh(3000);
        return;
      }
      const sizeMap = new Map<string, number>();
      for (const m of fresh) {
        if (m?.id && typeof m.fileSize === 'number') {
          sizeMap.set(m.id, m.fileSize);
        }
      }
      if (sizeMap.size) {
        this.models = this.models.map(m =>
          sizeMap.has(m.id) ? { ...m, fileSize: sizeMap.get(m.id) } : m,
        );
      }
    } catch (err) {
      console.error('Failed to refresh model sizes', err);
    }

    this.scheduleModelSizeRefresh(3000 + this.modelSizeAttempts * 500);
  }

  /** Models filtered by the search term and grouped by architecture */
  private get filteredGroupedModels(): {
    arch: string;
    models: ModelOption[];
  }[] {
    const filtered = this.filteredModels;

    // Group by arch (falling back to first segment of the id)
    const groups = new Map<string, ModelOption[]>();
    for (const m of filtered) {
      const arch = (m as any).arch || m.id.split('_')[0];
      if (!groups.has(arch)) groups.set(arch, []);
      groups.get(arch)!.push(m);
    }

    return Array.from(groups.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([arch, models]) => ({ arch, models }));
  }

  private get filteredModels(): ModelOption[] {
    const q = this.modelFilter.toLowerCase();
    const activeTags = this.activeTags;
    return this.models.filter((m) => {
      // Text search
      if (q && !(
        m.id.toLowerCase().includes(q) ||
        m.label.toLowerCase().includes(q) ||
        (m.description || '').toLowerCase().includes(q) ||
        (m.openVocab && 'open vocab'.includes(q)) ||
        (m.tags || []).some(t => t.toLowerCase().includes(q))
      )) return false;
      // Special "cached" filter
      if (activeTags.has('__cached__')) {
        if (!this.cachedModelIds?.has(m.id)) return false;
      }
      // Tag filter: model must have ALL active tags
      const tagFilter = new Set(activeTags);
      tagFilter.delete('__cached__');
      if (tagFilter.size > 0) {
        const modelTags = new Set(m.tags || []);
        for (const tag of tagFilter) {
          if (!modelTags.has(tag)) return false;
        }
      }
      return true;
    });
  }

  /** Toggle a tag in the active filter set */
  private toggleTag(tag: string) {
    const next = new Set(this.activeTags);
    if (next.has(tag)) {
      next.delete(tag);
    } else {
      next.add(tag);
    }
    this.activeTags = next;
    // Reset drilldown selection when tags change
    this.selectedDataset = '';
    this.selectedArch = '';
  }

  /** Clear all active tag filters */
  private clearTags() {
    this.activeTags = new Set();
  }

  /** Get tag counts for currently filtered models (by dimension) */
  private getTagCounts(): Map<string, number> {
    const counts = new Map<string, number>();
    // Count against text-filtered models only (not tag-filtered) to show what's available
    const q = this.modelFilter.toLowerCase();
    const textFiltered = this.models.filter((m) =>
      !q ||
      m.id.toLowerCase().includes(q) ||
      m.label.toLowerCase().includes(q) ||
      (m.description || '').toLowerCase().includes(q) ||
      (m.openVocab && 'open vocab'.includes(q))
    );
    for (const m of textFiltered) {
      for (const tag of (m.tags || [])) {
        counts.set(tag, (counts.get(tag) ?? 0) + 1);
      }
    }
    return counts;
  }

  /** Get display label for a tag (strip dimension prefix, format nicely) */
  private tagLabel(tag: string): string {
    const value = tag.includes(':') ? tag.split(':')[1] : tag;
    return value.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  /** Get the speed level (1-3) from a model's tags */
  private getSpeedLevel(model: ModelOption): number {
    const tags = model.tags || [];
    if (tags.includes('speed:fast')) return 3;
    if (tags.includes('speed:balanced')) return 2;
    if (tags.includes('speed:accurate')) return 1;
    return 0;
  }

  /** Get output type badges for a model */
  private getOutputBadges(model: ModelOption): { label: string; cls: string }[] {
    const tags = model.tags || [];
    const badges: { label: string; cls: string }[] = [];
    if (tags.includes('output:bounding-box')) badges.push({ label: 'BBox', cls: 'bbox' });
    if (tags.includes('output:mask')) badges.push({ label: 'Mask', cls: 'mask' });
    if (tags.includes('output:keypoints')) badges.push({ label: 'KPts', cls: 'keypoints' });
    return badges;
  }

  private normalizeDataset(model: ModelOption): string {
    const raw = String((model as any).dataset ?? '').toLowerCase();
    if (!raw) return 'unknown';
    if (raw.includes('coco')) return 'coco';
    if (raw.includes('voc')) return 'voc';
    if (raw.includes('wider')) return 'wider_face';
    if (raw.includes('crowd')) return 'crowdhuman';
    if (raw.includes('objects365')) return 'objects365';
    if (raw.includes('lvis')) return 'lvis';
    return raw.split(',')[0].split(' ')[0];
  }

  private get datasetOptions(): { id: string; label: string; count: number }[] {
    const counts = new Map<string, number>();
    for (const m of this.filteredModels) {
      const key = this.normalizeDataset(m);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return Array.from(counts.entries())
      .map(([id, count]) => ({ id, label: id.replace(/_/g, ' ').toUpperCase(), count }))
      .sort((a, b) => a.label.localeCompare(b.label));
  }

  private get archOptions(): { id: string; count: number }[] {
    const counts = new Map<string, number>();
    for (const m of this.filteredModels) {
      if (this.normalizeDataset(m) !== this.selectedDataset) continue;
      const arch = (m as any).arch || m.id.split('_')[0];
      counts.set(arch, (counts.get(arch) ?? 0) + 1);
    }
    return Array.from(counts.entries())
      .map(([id, count]) => ({ id, count }))
      .sort((a, b) => a.id.localeCompare(b.id));
  }

  private get modelOptions(): ModelOption[] {
    return this.filteredModels.filter((m) => {
      const datasetMatch = this.normalizeDataset(m) === this.selectedDataset;
      const arch = (m as any).arch || m.id.split('_')[0];
      const archMatch = this.selectedArch ? arch === this.selectedArch : true;
      return datasetMatch && archMatch;
    });
  }

  private get pendingModelInfo(): ModelOption | undefined {
    return this.models.find(m => m.id === this.pendingModelId);
  }

  private get modelSizesPending(): boolean {
    const missing = this.models.some(m => m.id !== 'none' && m.fileSize === undefined);
    return missing && this.modelSizeAttempts < 6;
  }

  /** Whether the currently selected model is an open-vocabulary model */
  private get isOpenVocab(): boolean {
    const model = this.models.find(m => m.id === this.selectedModel);
    return model?.openVocab === true;
  }

  private syncModelSelection(modelId: string = this.selectedModel) {
    const current = this.models.find(m => m.id === modelId);
    if (!current) return;
    this.selectedDataset = this.normalizeDataset(current);
    this.selectedArch = (current as any).arch || current.id.split('_')[0];
  }

  /**
   * Format a status message by replacing raw byte counts
   * (e.g. "123456789/987654321 bytes") with human-readable sizes.
   */
  private formatStatusMessage(msg: string): string {
    return msg.replace(
      /(\d+)\/(\d+)\s*bytes/gi,
      (_match, downloaded, total) =>
        `${prettyBytes(Number(downloaded))} / ${prettyBytes(Number(total))}`,
    );
  }

  private get _statusIcon(): string {
    switch (this.modelStatus) {
      case 'checking': return '🔍';
      case 'downloading': return '⬇️';
      case 'applying': return '⏳';
      case 'ready': return '✅';
      case 'error': return '❌';
      default: return '';
    }
  }

  private onModelFilterInput(ev: Event) {
    const input = ev.target as HTMLInputElement;
    this.modelFilter = input.value;
  }

  private openModelDialog() {
    this.modelFilter = '';
    this.pendingModelId = this.selectedModel;
    this.syncModelSelection(this.pendingModelId);
    this.fetchCachedModels();
    this.modelDialog?.show();
  }

  private async resetModel() {
    this.selectedModel = 'none';
    this.pendingModelId = 'none';
    this.selectedClassIds = new Set();
    this.availableClasses = [];
    this.classNamesText = '';
    this.backendInfo = null;
    await this.applyModel('none');
  }

  private onPendingModelSelect(model: string) {
    if (!model) return;
    this.pendingModelId = model;
  }

  private applyPendingModel() {
    if (!this.pendingModelId || this.pendingModelId === this.selectedModel) {
      this.modelDialog?.close('close');
      return;
    }
    const model = this.pendingModelId;
    this.modelDialog?.close('apply');
    // Defer model selection so the dialog close doesn't block rendering
    setTimeout(() => this.selectModel(model), 0);
  }

  private async onModelChange(ev: Event) {
    const select = ev.target as HTMLSelectElement;
    const model = select.value;
    await this.selectModel(model);
  }

  private async selectModel(model: string) {
    if (!model || model === this.selectedModel) return;
    this.selectedModel = model;
    this.pendingModelId = model;
    this.syncModelSelection();
    // Reset class selection for the new model
    this.selectedClassIds = new Set();
    this.hasPersistedClassList = false;
    this.availableClasses = [];
    this.classNamesText = '';
    this.fetchModelClasses(model);

    // Skip preparation for 'none'
    if (model === 'none') {
      await this.applyModel(model);
      return;
    }

    // Prepare (download if needed) then apply
    this.modelStatus = 'checking';
    this.modelProgress = 0;
    this.modelStatusMessage = 'Checking model cache...';

    try {
      const res = await fetch(`${this.basepath}/cameras/models/prepare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model }),
      });

      if (!res.ok || !res.body) {
        this.modelStatus = 'error';
        this.modelStatusMessage = `Failed to prepare model (HTTP ${res.status})`;
        return;
      }

      // Read SSE stream
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events from buffer
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        let updated = false;
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6));
              this.modelStatus = event.status;
              this.modelProgress = event.progress ?? 0;
              this.modelStatusMessage = event.message ?? '';
              updated = true;
            } catch { /* ignore parse errors */ }
          }
        }
        // Yield to let Lit render the updated progress
        if (updated) {
          await new Promise(r => requestAnimationFrame(r));
        }
      }

      if (this.modelStatus === 'error') {
        // Stay in error state, don't apply
        return;
      }

      // Model is ready — apply it (saves config + restarts stream)
      this.modelStatus = 'applying';
      this.modelStatusMessage = 'Applying model and restarting stream...';
      await this.applyModel(model);

      this.modelStatus = 'ready';
      this.modelStatusMessage = 'Model applied successfully';
      // Update cached model set (model was just downloaded)
      const nextCached = new Set(this.cachedModelIds);
      nextCached.add(model);
      this.cachedModelIds = nextCached;
      // Refresh backend status after stream restart (with delay for process startup)
      setTimeout(() => this.fetchBackendStatus(), 5000);
      // Poll faster until the running model catches up
      this._startFastBackendPoll();
      // Auto-clear the status bar after a short delay
      setTimeout(() => {
        if (this.modelStatus === 'ready') {
          this.modelStatus = 'idle';
        }
      }, 4000);

    } catch (err) {
      console.error('Failed to prepare model', err);
      this.modelStatus = 'error';
      this.modelStatusMessage = `Error: ${err}`;
    }
  }

  private async applyModel(model: string) {
    try {
      await fetch(`${this.basepath}/cameras/model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camStream: this.camStream, model }),
      });
    } catch (err) {
      console.error('Failed to update model', err);
    }
  }

  private async onSahiToggle(ev: Event) {
    const input = ev.target as HTMLInputElement;
    const useSahi = input.checked;
    this.useSahi = useSahi;
    try {
      await fetch(`${this.basepath}/cameras/sahi`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camStream: this.camStream, useSahi }),
      });
    } catch (err) {
      console.error('Failed to update SAHI setting', err);
    }
  }

  private async onSmoothingToggle(ev: Event) {
    const input = ev.target as HTMLInputElement;
    const useSmoothing = input.checked;
    this.useSmoothing = useSmoothing;
    try {
      await fetch(`${this.basepath}/cameras/smoothing`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camStream: this.camStream, useSmoothing }),
      });
    } catch (err) {
      console.error('Failed to update smoothing setting', err);
    }
  }

  private async onFrameBufferChange(ev: Event) {
    const input = ev.target as HTMLInputElement;
    const frameBuffer = Math.max(0, parseInt(input.value, 10) || 0);
    this.frameBuffer = frameBuffer;
    try {
      await fetch(`${this.basepath}/cameras/frameBuffer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camStream: this.camStream, frameBuffer }),
      });
    } catch (err) {
      console.error('Failed to update sahi padding', err);
    }
  }

  private async onConfidenceChange(ev: Event) {
    const input = ev.target as HTMLInputElement;
    const confidence = parseFloat(input.value);
    this.confidence = confidence;
    try {
      await fetch(`${this.basepath}/cameras/confidence`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camStream: this.camStream, confidence }),
      });
    } catch (err) {
      console.error('Failed to update confidence', err);
    }
  }

  private async onNmsIouChange(ev: Event) {
    const input = ev.target as HTMLInputElement;
    const nmsIou = Math.min(1, Math.max(0.1, parseFloat(input.value)));
    this.nmsIou = nmsIou;
    try {
      await fetch(`${this.basepath}/cameras/nmsIou`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camStream: this.camStream, nmsIou }),
      });
    } catch (err) {
      console.error('Failed to update NMS IOU', err);
    }
  }

  private async onSahiIouChange(ev: Event) {
    const input = ev.target as HTMLInputElement;
    const sahiIou = Math.min(1, Math.max(0.1, parseFloat(input.value)));
    this.sahiIou = sahiIou;
    try {
      await fetch(`${this.basepath}/cameras/sahiIou`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camStream: this.camStream, sahiIou }),
      });
    } catch (err) {
      console.error('Failed to update SAHI IOU', err);
    }
  }

  private async onOverlapRatioChange(ev: Event) {
    const input = ev.target as HTMLInputElement;
    const percentValue = Math.min(50, Math.max(5, parseFloat(input.value) || 0));
    const overlapRatio = percentValue / 100;
    this.overlapRatio = overlapRatio;
    try {
      await fetch(`${this.basepath}/cameras/overlapRatio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camStream: this.camStream, overlapRatio }),
      });
    } catch (err) {
      console.error('Failed to update overlap ratio', err);
    }
  }

  private async fetchModelClasses(modelId: string, retries = 4) {
    if (!modelId || modelId === 'none') {
      this.availableClasses = [];
      return;
    }
    this.classesLoading = true;
    this.classesError = '';
    for (let attempt = 0; attempt <= retries; attempt += 1) {
      try {
        const res = await fetch(`${this.basepath}/cameras/models/${encodeURIComponent(modelId)}/classes`);
        if (res.ok) {
          this.availableClasses = await res.json();
          // If no persisted selection, default to all classes selected
          if (this.selectedClassIds.size === 0 && !this.hasPersistedClassList) {
            this.selectedClassIds = new Set(this.availableClasses.map(c => c.id));
          }
          this.classesLoading = false;
          this.classesError = '';
          return;
        }
        if (res.status < 500) {
          this.classesError = `Failed to load classes (HTTP ${res.status})`;
          break;
        }
      } catch (err) {
        if (attempt === retries) {
          console.error('Failed to fetch model classes', err);
        }
      }
      await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
    }

    this.classesLoading = false;
    if (!this.classesError) {
      this.classesError = 'Failed to load classes. Video service may be starting.';
    }
  }

  private get filteredClasses(): ClassOption[] {
    const q = this.classFilter.toLowerCase();
    if (!q) return this.availableClasses;
    return this.availableClasses.filter(c => c.name.toLowerCase().includes(q));
  }

  private onClassFilterInput(ev: Event) {
    const input = ev.target as HTMLInputElement;
    this.classFilter = input.value;
  }

  private openClassDialog() {
    if (this.selectedModel) {
      this.fetchModelClasses(this.selectedModel);
    }
    this.classFilter = '';
    this.pendingClassIds = new Set(this.selectedClassIds);
    this.classDialog?.show();
  }

  private retryFetchClasses() {
    if (this.selectedModel) {
      this.fetchModelClasses(this.selectedModel);
    }
  }

  private async onClassDialogClose() {
    if (this.classDialog?.returnValue !== 'apply') {
      this.pendingClassIds = new Set();
      return;
    }
    this.selectedClassIds = new Set(this.pendingClassIds);
    this.pendingClassIds = new Set();
    await this.persistClassList();
  }

  private onPendingClassToggle(classId: number) {
    const newSet = new Set(this.pendingClassIds);
    if (newSet.has(classId)) {
      newSet.delete(classId);
    } else {
      newSet.add(classId);
    }
    this.pendingClassIds = newSet;
  }

  private selectAllClasses() {
    this.pendingClassIds = new Set(this.availableClasses.map(c => c.id));
  }

  private selectNoneClasses() {
    this.pendingClassIds = new Set();
  }

  private async persistClassList() {
    const classList = Array.from(this.selectedClassIds);
    try {
      await fetch(`${this.basepath}/cameras/classList`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camStream: this.camStream, classList }),
      });
    } catch (err) {
      console.error('Failed to update class list', err);
    }
  }

  private onClassNamesInput(ev: Event) {
    const input = ev.target as HTMLInputElement;
    this.classNamesText = input.value;
    // Debounce persistence
    if (this.classNamesDebounce) clearTimeout(this.classNamesDebounce);
    this.classNamesDebounce = setTimeout(() => this.persistClassNames(), 600);
  }

  private async persistClassNames() {
    const classNames = this.classNamesText
      .split(',')
      .map(s => s.trim())
      .filter(s => s.length > 0);
    try {
      await fetch(`${this.basepath}/cameras/classNames`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camStream: this.camStream, classNames }),
      });
    } catch (err) {
      console.error('Failed to update class names', err);
    }
  }

  private get selectedClassOptions(): ClassOption[] {
    if (!this.availableClasses.length) return [];
    return this.availableClasses.filter(c => this.selectedClassIds.has(c.id));
  }

  render() {
    return html`
      <div class="control">
        <div class="model-picker">
          <div class="model-summary-header">
            <span>Model</span>
                      <div style="display: flex; gap: 8px;">
            ${this.selectedModel !== 'none' ? html`
              <md-text-button @click=${this.resetModel}>Reset</md-text-button>
            ` : ''}
          </div>
            <span class="model-count">
              ${this.models.length} models available
            </span>
          </div>
          <md-elevated-button class="model-button" @click=${this.openModelDialog}>
            ${this.selectedModel}
            <md-icon slot="icon">neurology</md-icon>
          </md-elevated-button>


        </div>

        ${this.selectedModel && this.selectedModel !== 'none' ? html`
        <div class="backend-badge ${this._backendBadgeClass}">
          <span class="badge-dot"></span>
          <span>${this._backendLabel}</span>
          ${this._backendDetail ? html`
            <span class="badge-detail">${this._backendDetail}</span>
          ` : ''}
          ${this._canBuildTrt ? html`
            <button
              class="trt-build-btn"
              @click=${this.buildTrt}
              ?disabled=${this.trtBuildStatus === 'building'}
            >${this.trtBuildStatus === 'building' ? 'Building…' : 'Build TensorRT'}</button>
          ` : ''}
          ${this._runningModelName ? html`
            <span class="running-model" title="${this._runningModelName}">${this._runningModelName}</span>
          ` : ''}
          ${this._isModelMismatch ? html`
            <span class="model-switching">⟶ Switching model…</span>
          ` : ''}
          ${this.trtBuildStatus === 'building' ? html`
            <div class="trt-progress-track">
              <div class="trt-progress-fill" style="width: ${this.trtBuildProgress}%"></div>
            </div>
            <span class="trt-build-status">${this.trtBuildMessage}</span>
          ` : ''}
          ${this.trtBuildStatus === 'error' ? html`
            <span class="trt-build-status error">${this.trtBuildMessage}</span>
          ` : ''}
          ${this.trtBuildStatus === 'ready' ? html`
            <span class="trt-build-status ready">${this.trtBuildMessage}</span>
          ` : ''}
        </div>
        ` : ''}

        ${this.modelStatus !== 'idle' ? html`
          <div class="model-status-bar ${this.modelStatus}">
            <span>${this._statusIcon} ${this.formatStatusMessage(this.modelStatusMessage)}</span>
            ${this.modelStatus === 'downloading' ? html`
              <div class="progress-track">
                <div class="progress-fill" style="width: ${this.modelProgress}%"></div>
              </div>
            ` : ''}
          </div>
        ` : ''}

        <div class="frame-buffer-row">
          <label for="confidence" class="fb-label">Confidence</label>
          <input
            id="confidence"
            type="number"
            min="0.1"
            max="1.0"
            step="0.05"
            .value=${String(this.confidence)}
            @change=${this.onConfidenceChange}
            class="fb-input"
          />
        </div>

        <label class="sahi-toggle">
          <input
            type="checkbox"
            .checked=${this.useSahi}
            @change=${this.onSahiToggle}
          />
          Use SAHI (Slicing Aided Hyper Inference)
        </label>

        ${this.useSahi ? html`
        <div class="sahi-subsettings">
          <div class="frame-buffer-row">
            <label for="frameBuffer" class="fb-label">Padding (px)</label>
            <input
              id="frameBuffer"
              type="number"
              min="0"
              max="500"
              .value=${String(this.frameBuffer)}
              @change=${this.onFrameBufferChange}
              class="fb-input"
            />
          </div>

          <div class="frame-buffer-row">
            <label for="sahiIou" class="fb-label">Merge IoU</label>
            <input
              id="sahiIou"
              type="number"
              min="0.1"
              max="1.0"
              step="0.05"
              .value=${String(this.sahiIou)}
              @change=${this.onSahiIouChange}
              class="fb-input"
            />
          </div>

          <div class="frame-buffer-row">
            <label for="overlapRatio" class="fb-label">Overlap %</label>
            <input
              id="overlapRatio"
              type="number"
              min="0.05"
              max="0.5"
              step="0.05"
              .value=${String((this.overlapRatio * 100).toFixed(0))}
              @change=${this.onOverlapRatioChange}
              class="fb-input"
            />
          </div>
        </div>
        ` : ''}

        <div class="frame-buffer-row">
          <label for="nmsIou" class="fb-label">NMS IoU</label>
          <input
            id="nmsIou"
            type="number"
            min="0.1"
            max="1.0"
            step="0.05"
            .value=${String(this.nmsIou)}
            @change=${this.onNmsIouChange}
            class="fb-input"
          />
        </div>

        <label class="sahi-toggle">
          <input
            type="checkbox"
            .checked=${this.useSmoothing}
            @change=${this.onSmoothingToggle}
          />
          Detection Smoothing
        </label>


        <div class="class-filter-section">
          <div class="class-summary-header">
            <span>Detection Classes</span>
            ${!this.isOpenVocab ? html`
              <md-text-button @click=${this.openClassDialog}>Select</md-text-button>
            ` : ''}
          </div>

          ${this.isOpenVocab ? html`
            <input
              class="model-filter"
              type="text"
              placeholder="Enter class names, e.g. person, dog, car"
              .value=${this.classNamesText}
              @input=${this.onClassNamesInput}
            />
            <span class="class-summary-empty" style="font-size: 0.75rem; color: #999;">
              Comma-separated list of objects to detect
            </span>
          ` : html`
            <div class="class-summary-chips">
              ${repeat(this.selectedClassOptions.slice(0, 8), c => c.id, c => html`
                <md-filter-chip label=${c.name} selected></md-filter-chip>
              `)}
              ${this.selectedClassOptions.length > 8 ? html`
                <span class="class-summary-empty">+${this.selectedClassOptions.length - 8} more</span>
              ` : ''}
              ${this.selectedClassOptions.length === 0 ? html`
                <span class="class-summary-empty">
                  ${this.classesLoading ? 'Loading classes...' : (this.classesError ? 'Classes unavailable' : (this.availableClasses.length ? 'All classes selected' : 'Classes not loaded'))}
                </span>
              ` : ''}
            </div>
          `}
        </div>
      </div>

      <md-dialog
        id="class-dialog"
        class="dialog"
        @close=${this.onClassDialogClose}
      >
        <div slot="headline">Select Detection Classes</div>
        <form slot="content" method="dialog">
          ${this.classesLoading ? html`
            <div class="class-count-label" style="margin-bottom: 8px;">
              Loading classes...
            </div>
          ` : ''}
          ${this.classesError ? html`
            <div class="class-count-label" style="margin-bottom: 8px; color: #b33;">
              ${this.classesError}
            </div>
            <md-text-button @click=${this.retryFetchClasses}>Retry</md-text-button>
          ` : ''}

          <div class="class-bulk-actions">
            <span @click=${this.selectAllClasses}>Select All</span>
            <span @click=${this.selectNoneClasses}>Select None</span>
            <span class="class-count-label" style="cursor:default;text-decoration:none;color:#999">
              ${this.filteredClasses.length} classes
            </span>
          </div>

          <input
            class="class-filter-search"
            type="text"
            placeholder="Filter classes..."
            .value=${this.classFilter}
            @input=${this.onClassFilterInput}
          />

          <div class="class-chips">
            ${!this.classesLoading && !this.classesError ? repeat(this.filteredClasses, c => c.id, c => html`
              <md-filter-chip
                label=${c.name}
                ?selected=${this.pendingClassIds.has(c.id)}
                @click=${() => this.onPendingClassToggle(c.id)}
              ></md-filter-chip>
            `) : ''}
          </div>

          <div class="class-count-label">
            Leaving the selection empty means "all classes".
          </div>
        </form>
        <div slot="actions">
          <md-text-button @click=${() => this.classDialog?.close('cancel')}>Cancel</md-text-button>
          <md-elevated-button @click=${() => this.classDialog?.close('apply')}>Apply</md-elevated-button>
        </div>
      </md-dialog>

      <md-dialog
        id="model-dialog"
        class="dialog"
      >
        <div slot="headline">Select Model</div>
        <form slot="content" method="dialog">
          <input
            class="model-filter"
            type="text"
            placeholder="Search models..."
            .value=${this.modelFilter}
            @input=${this.onModelFilterInput}
            ?disabled=${this.modelStatus !== 'idle' && this.modelStatus !== 'ready' && this.modelStatus !== 'error'}
          />
          <div class="model-browser">
            ${Object.keys(this.availableTags).length > 0 ? html`
              <div class="tag-filter-sidebar">
                ${this.activeTags.size > 0 ? html`
                  <button class="tag-chip active" type="button" @click=${this.clearTags}
                    style="border-color: #b33; color: #b33; background: #fff0f0;">✕ Clear filters</button>
                ` : ''}
                ${(this.cachedModelIds?.size ?? 0) > 0 ? html`
                  <span class="tag-dim-label">Status</span>
                  <button
                    class="tag-chip ${this.activeTags.has('__cached__') ? 'active' : ''}"
                    type="button"
                    @click=${() => this.toggleTag('__cached__')}
                  >Cached <span class="tag-count">${this.cachedModelIds?.size ?? 0}</span></button>
                ` : ''}
                ${Object.entries(this.availableTags).map(([dim, tags]) => {
                  const counts = this.getTagCounts();
                  const dimLabels: Record<string, string> = {
                    task: 'Task', output: 'Output', speed: 'Speed',
                    capability: 'Capability', domain: 'Domain',
                  };
                  return html`
                    <span class="tag-dim-label">${dimLabels[dim] || dim}</span>
                    ${tags.map(tag => {
                      const count = counts.get(tag) ?? 0;
                      const isActive = this.activeTags.has(tag);
                      return html`
                        <button
                          class="tag-chip ${isActive ? 'active' : ''}"
                          type="button"
                          @click=${() => this.toggleTag(tag)}
                        >${this.tagLabel(tag)} <span class="tag-count">${count}</span></button>
                      `;
                    })}
                  `;
                })}
              </div>
            ` : ''}
            <div class="model-panel">
              <div class="model-panel-header" style="display:flex;align-items:center;gap:6px;">Models <span class="model-item-count">${this.filteredModels.length}</span>${(this.cachedModelIds?.size ?? 0) > 0 ? html`<button class="clear-cache-btn" type="button" @click=${() => this.clearAllCache()}>🗑 Clear cache</button>` : ''}</div>
              <div class="scroller">
                ${repeat(this.filteredModels, m => m.id, m => {
                    const outputBadges = this.getOutputBadges(m);
                    const speedLevel = this.getSpeedLevel(m);
                    return html`
                    <button
                    class="model-item ${m.id === this.pendingModelId ? 'active' : ''}"
                    type="button"
                    ?disabled=${this.modelStatus !== 'idle' && this.modelStatus !== 'ready' && this.modelStatus !== 'error'}
                    @click=${() => this.onPendingModelSelect(m.id)}
                    >
                    <span>${m.label}${m.openVocab ? html`<span class="ov-badge">Open Vocab</span>` : ''}${this.cachedModelIds?.has(m.id) ? html`<span class="cached-badge">Cached</span>` : ''}${outputBadges.map(b => html`<span class="output-badge ${b.cls}">${b.label}</span>`)}${speedLevel > 0 ? html`<span class="speed-indicator" title="${speedLevel === 3 ? 'Fast' : speedLevel === 2 ? 'Balanced' : 'Accurate'}">${[1,2,3].map(i => html`<span class="speed-bar ${i <= speedLevel ? 'filled' : ''}"></span>`)}</span>` : ''}</span>${this.cachedModelIds?.has(m.id) ? html`<button class="delete-cache-btn" type="button" title="Remove from cache" @click=${(ev: Event) => this.deleteCachedModel(m.id, ev)}>✕</button>` : ''}${m.fileSize ? html`<span class="model-item-count">${m.fileSize} MB</span>` : ''}
                    </button>
                `;})}
              </div>
            </div>
          </div>
          ${this.pendingModelInfo ? html`
            <div class="model-details">
              <h4>Model Details</h4>
              ${this.pendingModelInfo.description ? html`
                ${this.pendingModelInfo.description.split('\n').map(line => line.trim()).filter(line => line).map(line => html`<div style="margin-bottom: 4px;">${line}</div>`)}
              ` : (this.pendingModelInfo.summary ? html`
                <div>${this.pendingModelInfo.summary}</div>
              ` : html`<div>No description available.</div>`)}
              ${(this.pendingModelInfo.tags || []).length > 0 ? html`
                <div class="tag-badges">
                  ${(this.pendingModelInfo.tags || []).map(tag => html`<span class="tag-badge-sm">${this.tagLabel(tag)}</span>`)}
                </div>
              ` : ''}
              ${this.pendingModelInfo.task ? html`
                <div style="margin-top: 6px;">Task: ${this.pendingModelInfo.task}</div>
              ` : ''}
              ${this.pendingModelInfo.fileSize ? html`
                <div>Download size: ${this.pendingModelInfo.fileSize} MB</div>
              ` : ''}
              ${this.pendingModelInfo.openVocab ? html`
                <div style="margin-top: 4px;"><span class="ov-badge" style="font-size: 0.7rem;">Open Vocab</span> This model accepts free-text class names.</div>
              ` : ''}
              ${this.pendingModelInfo.paper ? html`
                <div>
                  Paper: <a href="${this.pendingModelInfo.paper}" target="_blank" rel="noopener">${this.pendingModelInfo.paper}</a>
                </div>
              ` : ''}
            </div>
          ` : ''}
        </form>
        <div slot="actions">
          <md-text-button @click=${() => this.modelDialog?.close('close')}>Close</md-text-button>
          <md-elevated-button @click=${this.applyPendingModel}>Apply</md-elevated-button>
        </div>
      </md-dialog>
    `;
  }
}
