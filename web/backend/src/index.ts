import { Elysia } from "elysia";
import { staticPlugin } from '@elysiajs/static'
import { html } from '@elysiajs/html'
import { cors } from '@elysiajs/cors'
import { getCameras, getStreamSetup, selectCamera } from './cameras.js'
import { getStatus } from './MIRcontroller.js'
import { getMask, saveMask } from "./mask.js";
console.log('CURRENT', process.cwd())
const app = new Elysia();
app.use(staticPlugin({
  assets: "web/dist",
  prefix: "/"
}))
app.use(html())
app.use(cors())
app.get('/mask', getMask)
app.post('/mask/save', saveMask)
app.get('/cameras', getCameras)
app.get('/cameras/setup', getStreamSetup)
app.post('/cameras/select', selectCamera)
app.post('/mir/status', getStatus)
app.get('/', async () => {
  return Bun.file('web/dist/index.html')
})
app.listen(1100);

console.log(
  `🦊 Elysia is running at ${app.server?.hostname}:${app.server?.port}`
);
