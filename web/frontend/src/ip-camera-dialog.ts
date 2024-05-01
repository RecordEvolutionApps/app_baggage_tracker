import { LitElement, html, css, PropertyValueMap } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import { mainStyles, Camera, CamSetup } from './utils.js';
import './camera-selector.js';

import '@material/web/button/elevated-button.js';
import '@material/web/button/text-button.js';
import '@material/web/textfield/outlined-text-field.js';
import '@material/web/radio/radio.js';
import '@material/web/dialog/dialog.js';
import { MdDialog } from '@material/web/dialog/dialog.js';
import { MdOutlinedTextField } from '@material/web/textfield/outlined-text-field.js';

@customElement('ip-camera-dialog')
export class IpCameraDialog extends LitElement {

  @property({ type: String })
  camStream: string = 'frontCam'

  @property({ type: Object })
  camSetup?: CamSetup

  @state()
  private camList: Camera[] = [];

  dialog?: MdDialog;
  basepath: string;
  pathEl?: MdOutlinedTextField;
  usernameEl?: MdOutlinedTextField;
  passwordEl?: MdOutlinedTextField;

  constructor() {
    super();
    this.basepath = window.location.protocol + '//' + window.location.host;
  }

  static styles = [
    mainStyles,
    css`
      :host {
        min-width: 64px;
        --md-sys-color-primary: #002e6a;
        --md-sys-color-on-primary: #FFFFFF;
        --md-sys-color-primary-container: #2986cc;
        --md-sys-color-on-primary-container: #002020;
      }
      
      .primary {
        background: var(--md-sys-color-primary);
        color: var(--md-sys-color-on-primary);
      }

      .paging:not([active]) { display: none !important; }

      md-elevated-button, md-text-button {
        width: 100%;
        --md-elevated-button-container-color: #eceff1;
        --md-elevated-button-label-text-color: #5e5f61;
      }

      #dialog {
        --md-dialog-container-color: #fff;
        --md-dialog-headline-color: #5e5f61;
        --md-dialog-supporting-text-color: #5e5f61;
      }

      .column {
        display: flex;
        flex-direction: column;
        gap: 12px;
      }

      form {
        color: #5e5f61;
      }

    `,
  ];

  async firstUpdated(
    _changedProperties: PropertyValueMap<any> | Map<PropertyKey, unknown>,
  ) {

    this.dialog = this.shadowRoot?.getElementById('dialog') as MdDialog;
    this.pathEl = this.shadowRoot?.getElementById('path') as MdOutlinedTextField
    this.usernameEl = this.shadowRoot?.getElementById('username') as MdOutlinedTextField
    this.passwordEl = this.shadowRoot?.getElementById('password') as MdOutlinedTextField
  }

  update(changedProps: any) {
    if (changedProps.has('camSetup')) {
      
    }
    super.update(changedProps)
  }

  async getCameras() {
    this.camList = await fetch(`${this.basepath}/cameras`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
    }).then(res => res.json());

    console.log('CAMLIST', this.camList);

    await this.updateComplete;
  }
  
  async submitIPCamera() {
    const path = this.pathEl?.value
    const username = this.usernameEl?.value
    const password = this.passwordEl?.value
    console.log('selected', path, this.camStream);
    const payload: Camera = {
      type: 'IP',
      path,
      username,
      password,
      camStream: this.camStream,
    };

    await fetch(`${this.basepath}/cameras/select`, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
      },
      body: JSON.stringify(payload),
    });
  }

  show() {
    this.dialog?.show()
  }

  handleNameInputKeypress(ev: any) {
    if (ev.key === 'Enter' && this.passwordEl?.value.length) {
      this.dialog?.close('create');
    }
  }

  setupIPCam() {
    console.log(this.dialog?.returnValue);
    if (this.dialog?.returnValue === 'setup') {
      this.submitIPCamera()
    }
  }

  onCreateClick() {
    this.dialog?.show();
  }

  onDialogCancel() {
    if (this.dialog) {
      this.dialog.returnValue = 'cancel';
    }
  }

  render() {
    return html`
      <md-dialog
        @cancel=${this.onDialogCancel}
        @close=${this.setupIPCam}
        id="dialog"
        style="width: 480px;"
      >
        <div slot="headline">IP Camera Setup</div>
        <form slot="content" id="create-ip-form" method="dialog">
          <div class="column">
            <p>Enter your IP camera connection information</p>
            <md-outlined-text-field
                label="Camera IP address (127.0.0.1:4000/streampath)"
                value="${ this.camSetup?.camera?.path ?? '' }"
                type="text"
                field="ipaddress"
                >
            </md-outlined-text-field>
            <md-outlined-text-field
                label="Username"
                .value="${ this.camSetup?.camera?.username ?? '' }"
                type="text"
                field="username"
                >
            </md-outlined-text-field>
            <md-outlined-text-field
                label="Password"
                value=""
                type="password"
                field="password"
                >
            </md-outlined-text-field>
          </div>          
        </form>
        <div slot="actions">
          <md-text-button @click=${() => this.dialog?.close('cancel')}>Cancel</md-text-button>
          <md-text-button form="create-ip-form" value="setup">Set Camera</md-text-button>
        </div>
      </md-dialog>
      `;
  }
}
