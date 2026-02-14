import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import { repeat } from 'lit/directives/repeat.js';
import { mainStyles, CamSetup, ModelOption, ClassOption } from './utils.js';
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
  declare frameBuffer: number;

  @state()
  declare modelFilter: string;

  @state()
  declare selectedDataset: string;

  @state()
  declare selectedArch: string;

  @state()
  declare pendingModelId: string;

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

  private classNamesDebounce: ReturnType<typeof setTimeout> | null = null;

  @state()
  declare modelStatus: 'idle' | 'checking' | 'downloading' | 'applying' | 'ready' | 'error';

  @state()
  declare modelProgress: number;

  @state()
  declare modelStatusMessage: string;

  classDialog?: MdDialog;
  modelDialog?: MdDialog;
  private basepath = window.location.protocol + '//' + window.location.host;

  constructor() {
    super();
    this.camStream = 'frontCam';
    this.models = [];
    this.selectedModel = 'rtmdet_tiny_8xb32-300e_coco';
    this.useSahi = true;
    this.frameBuffer = 64;
    this.modelFilter = '';
    this.selectedDataset = '';
    this.selectedArch = '';
    this.pendingModelId = this.selectedModel;
    this.availableClasses = [];
    this.selectedClassIds = new Set();
    this.classFilter = '';
    this.pendingClassIds = new Set();
    this.classesLoading = false;
    this.classesError = '';
    this.classNamesText = '';
    this.modelStatus = 'idle';
    this.modelProgress = 0;
    this.modelStatusMessage = '';
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
        --md-outlined-button-label-text-font: sans-serif;
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

      .model-drilldown {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
        flex: 1;
        min-height: 0;
        overflow: hidden;
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
        align-items: center;
        gap: 8px;
        padding: 0 0 4px;
      }

      .fb-label {
        color: #5e5f61;
        font-family: sans-serif;
        font-size: 0.875rem;
        white-space: nowrap;
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
    `,
  ];

  protected firstUpdated(
    _changedProperties: PropertyValueMap<any> | Map<PropertyKey, unknown>,
  ): void {
    this.classDialog = this.shadowRoot?.getElementById('class-dialog') as MdDialog;
    this.modelDialog = this.shadowRoot?.getElementById('model-dialog') as MdDialog;
    this.fetchModels();
  }

  update(
    _changedProperties: PropertyValueMap<any> | Map<PropertyKey, unknown>,
  ) {
    if (_changedProperties.has('camSetup') && this.camSetup) {
      if (this.camSetup.camera?.model) {
        this.selectedModel = this.camSetup.camera.model;
      }
      this.useSahi = this.camSetup.camera?.useSahi ?? true;
      this.frameBuffer = this.camSetup.camera?.frameBuffer ?? 64;
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
    // Set current model from camSetup if available
    if (this.camSetup?.camera?.model) {
      this.selectedModel = this.camSetup.camera.model;
    }
    if (this.selectedModel) {
      this.fetchModelClasses(this.selectedModel);
      this.syncModelSelection();
    }
  }

  /** Models filtered by the search term and grouped by architecture */
  private get filteredGroupedModels(): {
    arch: string;
    models: ModelOption[];
  }[] {
    const q = this.modelFilter.toLowerCase();
    const filtered = this.models.filter(
      (m) =>
        !q ||
        m.id.toLowerCase().includes(q) ||
        m.label.toLowerCase().includes(q) ||
        (m.openVocab && 'open vocab'.includes(q)),
    );

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
    return this.models.filter(
      (m) =>
        !q ||
        m.id.toLowerCase().includes(q) ||
        m.label.toLowerCase().includes(q) ||
        (m.openVocab && 'open vocab'.includes(q)),
    );
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
    this.modelDialog?.show();
  }

  private async resetModel() {
    this.selectedModel = 'none';
    this.pendingModelId = 'none';
    this.selectedClassIds = new Set();
    this.availableClasses = [];
    this.classNamesText = '';
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
      console.error('Failed to update frame buffer', err);
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
          </div>
          <md-outlined-button class="model-button" @click=${this.openModelDialog}>
            ${this.selectedModel}
          </md-outlined-button>
          <div style="display: flex; gap: 8px;">
            ${this.selectedModel !== 'none' ? html`
              <md-text-button @click=${this.resetModel}>Reset</md-text-button>
            ` : ''}
          </div>
          <span class="model-count">
            ${this.models.length} models available
          </span>
        </div>

        ${this.modelStatus !== 'idle' ? html`
          <div class="model-status-bar ${this.modelStatus}">
            <span>${this._statusIcon} ${this.modelStatusMessage}</span>
            ${this.modelStatus === 'downloading' ? html`
              <div class="progress-track">
                <div class="progress-fill" style="width: ${this.modelProgress}%"></div>
              </div>
            ` : ''}
          </div>
        ` : ''}

        <label class="sahi-toggle">
          <input
            type="checkbox"
            .checked=${this.useSahi}
            @change=${this.onSahiToggle}
          />
          Use SAHI (Slicing Aided Hyper Inference)
        </label>

        <div
          class="frame-buffer-row"
          style="display:${this.useSahi ? 'flex' : 'none'}"
        >
          <label for="frameBuffer" class="fb-label">Frame Buffer (px)</label>
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
          <div class="model-drilldown">
            <div class="model-panel">
              <div class="model-panel-header">Training Data</div>
              <div class="scroller">
                ${repeat(this.datasetOptions, d => d.id, d => html`
                    <button
                    class="model-item ${d.id === this.selectedDataset ? 'active' : ''}"
                    type="button"
                    @click=${() => { this.selectedDataset = d.id; this.selectedArch = ''; }}
                    >
                    ${d.label}
                    <span class="model-item-count">${d.count}</span>
                    </button>
                `)}
              </div>
            </div>
            <div class="model-panel">
              <div class="model-panel-header">Architecture</div>
              <div class="scroller">
                ${repeat(this.archOptions, a => a.id, a => html`
                    <button
                    class="model-item ${a.id === this.selectedArch ? 'active' : ''}"
                    type="button"
                    @click=${() => { this.selectedArch = a.id; }}
                    >
                    ${a.id.toUpperCase()}
                    <span class="model-item-count">${a.count}</span>
                    </button>
                `)}
              </div>
            </div>
            <div class="model-panel">
              <div class="model-panel-header">Models</div>
              <div class="scroller">
                ${repeat(this.modelOptions, m => m.id, m => html`
                    <button
                    class="model-item ${m.id === this.pendingModelId ? 'active' : ''}"
                    type="button"
                    ?disabled=${this.modelStatus !== 'idle' && this.modelStatus !== 'ready' && this.modelStatus !== 'error'}
                    @click=${() => this.onPendingModelSelect(m.id)}
                    >
                    ${m.label}${m.openVocab ? html`<span class="ov-badge">Open Vocab</span>` : ''}${m.fileSize ? html`<span class="model-item-count">${m.fileSize} MB</span>` : ''}
                    </button>
                `)}
              </div>
            </div>
          </div>
          ${this.pendingModelInfo ? html`
            <div class="model-details">
              <h4>Model Details</h4>
              ${this.pendingModelInfo.summary ? html`
                <div>${this.pendingModelInfo.summary}</div>
              ` : html`<div>No description available.</div>`}
              ${this.pendingModelInfo.task ? html`
                <div style="margin-top: 6px;">Task: ${this.pendingModelInfo.task}</div>
              ` : ''}
              ${this.pendingModelInfo.architecture ? html`
                <div>Architecture: ${this.pendingModelInfo.architecture}</div>
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
