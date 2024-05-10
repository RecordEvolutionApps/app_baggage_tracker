declare global {
  interface Window {
      Janus: any,
      bootbox: any,
      streamsList: any,
      initializing: boolean
  }
}

let streaming: any
const WEBRTC_PORT = '1200'
const videoPlayer: any = {} 
videoPlayer.frontCam = document.getElementById('frontCam')
videoPlayer.backCam = document.getElementById('backCam')
videoPlayer.leftCam = document.getElementById('leftCam')
videoPlayer.rightCam = document.getElementById('rightCam')

const janusServerUrl = getJanusUrl(); // Replace with your Janus server URL
const iceServers = [
  {
    urls: "stun:stun.l.google.com:19302",
  },
  {
    urls: "turn:relay1.expressturn.com:3478",
    username: "ef8VXO351A31UJVGBY",
    credential: "PD1trsvPrgQ4uWAf",
  },
  // {
  //   urls: "turn:a.relay.metered.ca:80?transport=tcp",
  //   username: "f63d4fc5ff93197d239f602f",
  //   credential: "ZaHWKZRVcc1+8sKn",
  // },
  // {
  //   urls: "turn:a.relay.metered.ca:443",
  //   username: "f63d4fc5ff93197d239f602f",
  //   credential: "ZaHWKZRVcc1+8sKn",
  // },
  // {
  //   urls: "turn:a.relay.metered.ca:443?transport=tcp",
  //   username: "f63d4fc5ff93197d239f602f",
  //   credential: "ZaHWKZRVcc1+8sKn",
  // },
]

// Initialize Janus
window.window.Janus.debug = true
function initJanus(videoPlayer: any) {
  if (window.initializing) return
  window.initializing = true
  window.Janus.init({
    debug: "all",
    callback: function () {
      const janus = new window.Janus({
        server: janusServerUrl,
        iceServers: iceServers,
        withCredentials: true,
        success: function () {
          console.log("Janus connection successful!");

          // Attach to a Janus plugin (e.g., videoroom) for publishing and subscribing
          janus.attach({
            plugin: "janus.plugin.streaming",
            success: function (pluginHandle: any) {
              console.log("Plugin attached!");
              streaming = pluginHandle
              // updateStreamsList()
              startStream(123);
            },
            onmessage: function (msg: any, jsep: any) {
              // Handle msg, if needed, and check jsep
              console.log('<<<<<<<<< Got Message', msg.result, jsep )
              if (jsep) {
                // We have the ANSWER from the plugin
                console.log('>>>>>>>>>> creating answer ...')
                streaming.createAnswer(
                  {
                    jsep: jsep,
                    success: function (jsep: any) {
                      // window.Janus.debug("Got SDP!", jsep);
                      console.log('<<<<<<< answer created, now sending start >>>>>>>')
                      streaming.send({ message: { request: "start" }, jsep: jsep,
                        success: (result: any) => console.log('<<<<<<< start result', result)
                        });
                    },
                    error: function (error: any) {
                      window.Janus.error("WebRTC error:", error);
                    }
                  });
              }
            },

            onremotetrack: function (track: any, mid: any, on: any, metadata: any) {
              console.log('>>>>> Got remote track', mid, {on}, metadata)
              if (!on || metadata?.reason !== 'created') return
              console.log('>>>>>>>>> attaching stream to video-tag')
              const stream = new MediaStream([track]);
              window.Janus.attachMediaStream(videoPlayer[mid], stream);
              try {
                console.log('>>>>>>>>>>playing video now...')
                videoPlayer[mid].play()
              } catch (err) {
                console.warn('Could not Play the video directly', err)
              }
              // Invoked after handleRemoteJsep has got us a PeerConnection
              // This is info on a remote track: when added, we can choose to render
              // You can query metadata to get some more information on why track was added or removed
              // metadata fields:
              //   - reason: 'created' | 'ended' | 'mute' | 'unmute'
            },
            error: function (error: any) {
              window.Janus.error("  -- Error attaching plugin... ", error);
            },
            iceState: function (state: any) {
              window.Janus.log("ICE state changed to " + state);
            },
            webrtcState: function (on: any) {
              window.Janus.log("Janus says our WebRTC PeerConnection is " + (on ? "up" : "down") + " now");
            },
            slowLink: function (uplink: any, lost: any, mid: any) {
              window.Janus.warn("Janus reports problems " + (uplink ? "sending" : "receiving") +
                " packets on mid " + mid + " (" + lost + " lost packets)");
            },
          });
        },

        error: function (error: any) {
          console.error("Janus connection failed:", error);
        },
        destroyed: function () {
          console.log("Janus connection closed!");
        },
      });
    },
  });
}

function getJanusUrl() {
  let pa = location.host.split('-');
  let jns = pa[2]?.split('.') ?? [];
  jns[0] = WEBRTC_PORT ?? 1111;
  let jjns = jns.join('.');
  pa[2] = jjns;
  let jpa = pa.join('-');
  return 'https://' + jpa + '/janusrtc';
}


// function updateStreamsList() {
//   streaming.send({
//     message: { request: "list" }, success: function (result: any) {
//       if (!result) {
//         window.bootbox.alert("Got no response to our query for available streams");
//         return;
//       }
//       if (result["list"]) {
//         let list = result["list"];

//         window.Janus.log("Got a list of available streams:", list);
//         window.streamsList = {};
//         for (let mp in list) {
//           window.Janus.debug("  >> [" + list[mp]["id"] + "] " + list[mp]["description"] + " (" + list[mp]["type"] + ")");
//           // Check the nature of the available streams, and if there are some multistream ones
//           list[mp].legacy = true;
//           if (list[mp].media) {
//             let audios = 0, videos = 0;
//             for (let mi in list[mp].media) {
//               if (!list[mp].media[mi])
//                 continue;
//               if (list[mp].media[mi].type === "audio")
//                 audios++;
//               else if (list[mp].media[mi].type === "video")
//                 videos++;
//               if (audios > 1 || videos > 1) {
//                 list[mp].legacy = false;
//                 break;
//               }
//             }
//           }
//           // Keep track of all the available streams
//           // @ts-ignore
//           window.streamsList[list[mp]["id"]] = list[mp];
//         }
//         console.log('prepared streamslist', window.streamsList)
//         startStream(123);
//       }
//     }
//   });
// }

// function getStreamInfo(selectedStream: any) {
//   if (!selectedStream || !window.streamsList[selectedStream])
//     return;
//   // Send a request for more info on the mountpoint we subscribed to
//   streaming.send({
//     message: { request: "info", id: selectedStream }, success: function (result: any) {
//       console.log('Stream info', result)
//       if (result && result.info && result.info.metadata) {
//       }
//     }
//   });
// }

function startStream(selectedStream: any) {
  window.Janus.log(">>>>>> Starting watch for video id #" + selectedStream);
  
  // Prepare the request to start streaming and send it
  streaming.send({ 
    message: { request: "watch", id: selectedStream },
    success: (result: any) => console.log('<<<<<<< watch result', result)
  });
}

// function stopStream(selectedStream: any) {
//   let body = { request: "stop" };
//   streaming.send({ message: body });
//   streaming.hangup();
// }

export { initJanus }