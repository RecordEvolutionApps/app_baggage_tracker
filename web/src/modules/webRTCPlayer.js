 let streaming
    const videoPlayer = document.getElementById('remoteVideo')
    const WEBRTC_PORT = '1200'
    const janusServerUrl = getJanusUrl(); // Replace with your Janus server URL
    const iceServers = [
          {
            urls: "stun:stun.relay.metered.ca:80",
          },
          {
            urls: "turn:a.relay.metered.ca:80",
            username: "f63d4fc5ff93197d239f602f",
            credential: "ZaHWKZRVcc1+8sKn",
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
      ]
    // Initialize Janus
    Janus.init({
      debug: "all",
      callback: function () {
        const janus = new Janus({
          server: janusServerUrl,
          iceServers: iceServers,
          success: function () {
            console.log("Janus connection successful!");

            // Attach to a Janus plugin (e.g., videoroom) for publishing and subscribing
            janus.attach({
              plugin: "janus.plugin.streaming",
              success: function (pluginHandle) {
                console.log("Plugin attached!");
                streaming = pluginHandle
                streaming.send({message: {request: 'watch', id: 1}})
                // streaming.createOffer(
                // {
                //     tracks: [
                //         { type: 'video', capture: false, recv: true }
                //     ],
                //     success: function(jsep) {
                //         // Got our SDP! Send our OFFER to the plugin
                //         console.log('offer callback success', jsep)
                //         streaming.handleRemoteJsep({jsep: jsep});
                //         //streaming.send({ message: 'lalalu', jsep: jsep });
                //     },
                //     error: function(error) {
                //         // An error occurred...
                //     },
                //     customizeSdp: function(jsep) {
                //         // if you want to modify the original sdp, do as the following
                //         // oldSdp = jsep.sdp;
                //         // jsep.sdp = yourNewSdp;
                //     }
                // });
                // console.log('tracks', streaming.getRemoteTracks())
              },
              onmessage: function(msg, jsep) {
                // Handle msg, if needed, and check jsep
                console.log('Got Message', {msg, jsep})
                if(jsep) {
                    // We have the ANSWER from the plugin
                    // streaming.handleRemoteJsep({jsep: jsep});
                    streaming.createAnswer(
											{
												jsep: jsep,
												success: function(jsep) {
													Janus.debug("Got SDP!", jsep);
													let body = { request: "start" };
													streaming.send({ message: body, jsep: jsep });
												},
												error: function(error) {
													Janus.error("WebRTC error:", error);
												}
											});
                }
              },
              onlocaltrack: function(track, added) {
                  console.log('Got local track', track, added)
                  // Invoked after createOffer
                  // This is info on a local track: when added, we can choose to render
              },
              onremotetrack: function(track, mid, added, metadata) {
                console.log('Got remote track', track, mid, added, metadata)
                  const stream = new MediaStream([track]);
                  Janus.attachMediaStream(videoPlayer, stream);
                  try {
                    videoPlayer.play()
                  } catch(err){
                    console.warn('Could not Play the video directly', err)
                  }
                  // Invoked after handleRemoteJsep has got us a PeerConnection
                  // This is info on a remote track: when added, we can choose to render
                  // You can query metadata to get some more information on why track was added or removed
                  // metadata fields:
                  //   - reason: 'created' | 'ended' | 'mute' | 'unmute'
              },
              error: function(error) {
                Janus.error("  -- Error attaching plugin... ", error);
              },
              iceState: function(state) {
                Janus.log("ICE state changed to " + state);
              },
              webrtcState: function(on) {
                Janus.log("Janus says our WebRTC PeerConnection is " + (on ? "up" : "down") + " now");
              },
              slowLink: function(uplink, lost, mid) {
                Janus.warn("Janus reports problems " + (uplink ? "sending" : "receiving") +
                  " packets on mid " + mid + " (" + lost + " lost packets)");
              },
            });
          },

          error: function (error) {
            console.error("Janus connection failed:", error);
          },
          destroyed: function () {
            console.log("Janus connection closed!");
          },
        });
      },
    });

    function getJanusUrl() {
      let pa = location.host.split('-');
      let jns = pa[2]?.split('.') ?? [];
      jns[0] = WEBRTC_PORT ?? 1111;
      let jjns = jns.join('.');
      pa[2] = jjns;
      let jpa = pa.join('-');
      return 'https://' + jpa + '/janusrtc';
    }