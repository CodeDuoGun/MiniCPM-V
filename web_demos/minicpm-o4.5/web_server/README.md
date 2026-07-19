# 前端迁移

现有 `minicpm-o_2.6/web_server` 的 HTTP/SSE 接口可以直接切换 API 地址到 `:32560`：

- `POST /api/v1/init_options`
- `POST /api/v1/stream`
- `POST /api/v1/completions`
- `GET/POST /api/v1/slots...`

要使用真正的全双工返回，应在现有 `VideoCall.vue`/`VoiceCall.vue` 中使用
`src/minicpmo45Client.js`，每 100--200 ms 发送一段 16 kHz 单声道 WAV；服务端会为每段返回
`response.chunk`。收到 `choices[0].audio` 后立即加入播放队列，收到用户插话时调用 `cancel()`
并清空尚未播放的音频。患者一轮完整转写应在最后一块附带 `endOfTurn=true`，用于更新槽位和历史。

不要在浏览器端缓存患者音视频；页面关闭时调用 `/api/v1/session/close`。

