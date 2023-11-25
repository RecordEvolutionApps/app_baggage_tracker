import crypto from 'crypto';

const MIRIP = '192.168.178.28:8080' //= MIR-S143 '192.168.178.45' = MIRAGV
// MiR100/MiR200’s internal IP Address is: 192.168.12.20
const password = 'lala'
const username = 'marko'

const hashedPassword = crypto.createHash('sha256').update(password).digest('hex');
const tokenData = `${username}:${hashedPassword}`;
const apiToken = Buffer.from(tokenData).toString('base64');


export async function getStatus() {

    const response = await fetch(`http://${MIRIP}/status`, {
        method: "GET",
        headers: { 
            "Content-Type": "application/json",
            "Authorization": `Bearer ${apiToken}` }
        });

    return response.json();

}