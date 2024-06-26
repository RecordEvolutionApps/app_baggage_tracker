import { Elysia } from "elysia";
import { staticPlugin } from '@elysiajs/static'
import { html } from '@elysiajs/html'
import { cors } from '@elysiajs/cors'
import { getUSBCameras, getStreamSetup, selectCamera } from './cameras.js'
import { getMask, saveMask } from "./mask.js";
console.log('CURRENT', process.cwd())
const app = new Elysia();
app.use(staticPlugin({
  assets: "frontend/dist",
  prefix: "/"
}))
app.use(html())
app.use(cors())
app.get('/mask', getMask)
app.post('/mask/save', saveMask)
app.get('/cameras', getUSBCameras)
app.get('/cameras/setup', getStreamSetup)
app.post('/cameras/select', selectCamera)
app.get('/', async () => {
  return Bun.file('frontend/dist/index.html')
})
app.listen(1100);

console.log(
  `🦊 Elysia is running at ${app.server?.hostname}:${app.server?.port}`
);
