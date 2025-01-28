import { LitElement, html, css } from 'lit';
import { property, customElement, state } from 'lit/decorators.js';
import './video-canvas.js';
import { mainStyles, CamSetup } from './utils.js';

const WEBRTC_PORT = '1200'
@customElement('camera-player')
export class CameraPlayer extends LitElement {
  @property({ type: String }) label = 'Front';
  @property({ type: String }) id = 'frontCam';
  basepath: string;

  @state()
  camSetup?: CamSetup;

  animation_handle: any;
  videoElement?: HTMLVideoElement;
  canvasElement?: HTMLCanvasElement;

  constructor() {
    super();
    this.basepath = window.location.protocol + '//' + window.location.host;
  }

  async getCameraMetadata() {
    try {
      this.camSetup = await fetch(
        `${this.basepath}/cameras/setup?camStream=${this.id}`,
        {
          method: 'GET',
          headers: {
            Accept: 'application/json',
          },
        },
      ).then(res => res.json());
      console.log('got Camera Setup', this.camSetup)
    } catch (error) {
      console.error('Failed to get cameras', error);
    } finally {
      this.requestUpdate();
    }
  }

  protected async firstUpdated() {
    this.videoElement = this.shadowRoot?.getElementById(
      'video',
    ) as HTMLVideoElement;
    this.initializeWebRTC();
    this.getCameraMetadata();

    this.dispatchEvent(new CustomEvent('video-ready'));
  }

  getSignalingServerUrl() {
    let pa = location.host.split('-');
    let jns = pa[2]?.split('.') ?? [];
    jns[0] = WEBRTC_PORT ?? 1111;
    let jjns = jns.join('.');
    pa[2] = jjns;
    let jpa = pa.join('-');
    return 'wss://' + jpa;
  }

  async initializeWebRTC() {
    const ssurl = this.getSignalingServerUrl()
    console.log('ssurl', ssurl)
    // WebSocket-Verbindung zum Signaling-Server
    const ws = new WebSocket(ssurl);

    // WebRTC-Verbindung erstellen
    this.peerConnection = new RTCPeerConnection({
      iceServers: [
          { urls: "stun:stun.l.google.com:19302" },
          { urls: "stun:stun.l.google.com:5349" },
          { urls: "stun:stun1.l.google.com:3478" },
          { urls: "stun:stun1.l.google.com:5349" },
          { urls: "stun:stun2.l.google.com:19302" },
          { urls: "stun:stun2.l.google.com:5349" },
          { urls: "stun:stun3.l.google.com:3478" },
          { urls: "stun:stun3.l.google.com:5349" },
          { urls: "stun:stun4.l.google.com:19302" },
          { urls: "stun:stun4.l.google.com:5349" },
          {
              urls: "turn:relay1.expressturn.com:3478",
              username: "ef8VXO351A31UJVGBY",
              credential: "PD1trsvPrgQ4uWAf",
          },
          {
              urls: "turn:a.relay.metered.ca:80?transport=tcp",
              username: "f63d4fc5ff93197d239f602f",
              credential: "ZaHWKZRVcc1+8sKn",
          },
          {
              urls: "turn:a.relay.metered.ca:443",
              username: "f63d4fc5ff93197d239f602f",
              credential: "ZaHWKZRVcc1+8sKn",
          },
          {
              urls: "turn:a.relay.metered.ca:443?transport=tcp",
              username: "f63d4fc5ff93197d239f602f",
              credential: "ZaHWKZRVcc1+8sKn",
          },
      ],
    });

    // Remote-Stream verarbeiten
    this.peerConnection.ontrack = (event) => {
      if (event.streams && event.streams[0]) {
        this.videoElement.srcObject = event.streams[0];
      }
    };

    // ICE-Candidate an den Signaling-Server senden
    this.peerConnection.onicecandidate = (event) => {
      if (event.candidate) {
        ws.send(JSON.stringify({ type: 'ice_candidate', candidate: event.candidate }));
      }
    };

    // Nachrichten vom Signaling-Server verarbeiten
    ws.onmessage = async (event) => {
      const message = JSON.parse(event.data);

      if (message.type === 'offer') {
        // Remote-Beschreibung setzen
        await this.peerConnection.setRemoteDescription(new RTCSessionDescription(message.offer));

        // Antwort (Answer) erstellen
        const answer = await this.peerConnection.createAnswer();
        await this.peerConnection.setLocalDescription(answer);

        // Answer an den Signaling-Server senden
        ws.send(JSON.stringify({ type: 'answer', answer }));
      } else if (message.type === 'ice_candidate') {
        // ICE-Candidate hinzufügen
        await this.peerConnection.addIceCandidate(new RTCIceCandidate(message.candidate));
      }
    };
  }

  static styles = [
    mainStyles,
    css`
      :host {
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
        position: relative;
        margin: 0 auto;
        padding: 23px 16px;
        box-sizing: border-box;
        background: #fff;
      }

      h3 {
        display: none;
        align-self: baseline;
        margin: 0 0 5px 0;
      }

      @media only screen and (max-width: 600px) {
        h3 {
          display: block;
        }
        :host {
          padding: 5px 16px 0 16px;
        }
      }
    `,
  ];

  render() {
    return html`
      <h3>Baggage Vision</h3>
      <video-canvas
        .video=${this.videoElement}
        .camSetup=${this.camSetup}
        .width=${this.camSetup?.width ?? 1280}
        .height=${this.camSetup?.height ?? 720}
        .camStream=${this.id}
      ></video-canvas>
      <video id="video" autoplay controls muted playsinline hidden></video>
    `;
  }
}
