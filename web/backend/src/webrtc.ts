// Typdefinitionen für Nachrichten
type ClientID = string;

const PORT = 1200

interface SignalingMessage {
    type: 'register' | 'offer' | 'answer' | 'ice_candidate';
    client_id?: ClientID;
    sender_id?: ClientID;
    receiver_id?: ClientID;
    offer?: any;
    answer?: any;
    ice_candidate?: any;
}

// Globale Variablen zur Speicherung der Verbindungen
const connections: Map<ClientID, WebSocket> = new Map();

// WebSocket-Server erstellen
const server = Bun.serve<{ clientId: ClientID }>({
    port: PORT,
    fetch(req, server) {
        // Upgrade HTTP-Verbindung zu WebSocket
        const success = server.upgrade(req, {
            data: { clientId: new URL(req.url).searchParams.get("client_id") || "" },
        });
        return success ? undefined : new Response("WebSocket upgrade failed", { status: 400 });
    },
    websocket: {
        open(ws) {
            const clientId = ws.data.clientId;
            if (clientId) {
                connections.set(clientId, ws);
                console.log(`Client ${clientId} connected`);
            }
        },
        message(ws, message: string) {
            try {
                const data: SignalingMessage = JSON.parse(message);
                handleSignalingMessage(ws, data);
            } catch (error) {
                console.error("Error parsing message:", error);
            }
        },
        close(ws) {
            // Entferne den Client aus der Verbindungsliste
            for (const [clientId, connection] of connections.entries()) {
                if (connection === ws) {
                    connections.delete(clientId);
                    console.log(`Client ${clientId} disconnected`);
                    break;
                }
            }
        },
    },
});

console.log(`Signaling server running on ws://localhost:${server.port}`);

// Nachrichtenverarbeitung
function handleSignalingMessage(ws: WebSocket, message: SignalingMessage) {
    switch (message.type) {
        case 'register':
            if (message.client_id) {
                connections.set(message.client_id, ws);
                console.log(`Client ${message.client_id} registered`);
            }
            break;

        case 'offer':
            if (message.sender_id && message.receiver_id && message.offer) {
                const receiverSocket = connections.get(message.receiver_id);
                if (receiverSocket) {
                    receiverSocket.send(JSON.stringify({
                        type: 'offer',
                        offer: message.offer,
                        sender_id: message.sender_id,
                    }));
                    console.log(`Offer forwarded from ${message.sender_id} to ${message.receiver_id}`);
                }
            }
            break;

        case 'answer':
            if (message.sender_id && message.receiver_id && message.answer) {
                const receiverSocket = connections.get(message.receiver_id);
                if (receiverSocket) {
                    receiverSocket.send(JSON.stringify({
                        type: 'answer',
                        answer: message.answer,
                        sender_id: message.sender_id,
                    }));
                    console.log(`Answer forwarded from ${message.sender_id} to ${message.receiver_id}`);
                }
            }
            break;

        case 'ice_candidate':
            if (message.sender_id && message.receiver_id && message.ice_candidate) {
                const receiverSocket = connections.get(message.receiver_id);
                if (receiverSocket) {
                    receiverSocket.send(JSON.stringify({
                        type: 'ice_candidate',
                        ice_candidate: message.ice_candidate,
                        sender_id: message.sender_id,
                    }));
                    console.log(`ICE candidate forwarded from ${message.sender_id} to ${message.receiver_id}`);
                }
            }
            break;

        default:
            console.warn('Unknown message type:', message.type);
            break;
    }
}