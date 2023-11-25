import { Elysia } from "elysia";
import { staticPlugin } from '@elysiajs/static'
import { html } from '@elysiajs/html'
import { cors } from '@elysiajs/cors'
import { getCameras, selectCamera } from './cameras.js'
import { getStatus } from './MIRcontroller.js'

const app = new Elysia();
app.use(staticPlugin({
  assets: "../web/dist/",
  prefix: "/"
}))
app.use(html())
app.use(cors())
app.get('/cameras', getCameras)
app.post('/cameras/select', selectCamera)
app.post('/mir/status', getStatus)
app.get('/', async () => {
  return Bun.file('../web/dist/index.html')
})
app.listen(1100);

console.log(
  `🦊 Elysia is running at ${app.server?.hostname}:${app.server?.port}`
);
