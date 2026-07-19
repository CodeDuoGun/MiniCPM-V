# 多轮问诊槽位配置

`model_server.py` 在每轮患者语音结束后执行以下流程：

1. 优先读取流请求中的 `input_audio.transcript`；没有时使用 DashScope ASR 转写本轮音频。
2. 将患者本轮文本、最近对话和当前槽位交给独立文本模型抽取结构化更新。
3. 按 `visit_type` 选择 `slots`（初诊）或 `followup_slots`（复诊）。
4. 在 MiniCPM `streaming_generate` 前注入已确认信息及仍需收集的槽位。
5. 生成结束后把医生回复加入历史，供下一轮抽取使用。

推荐在项目根目录 `.env` 配置：

```bash
SLOT_LLM_BASE_URL=https://your-openai-compatible-host/v1
SLOT_LLM_API_KEY=your-key
SLOT_LLM_MODEL=your-json-capable-text-model

SLOT_ASR_API_KEY=your-dashscope-key
SLOT_ASR_MODEL=paraformer-realtime-v2
# 国内站通常可留空使用 SDK 默认值；国际站按部署环境配置。
SLOT_ASR_WS_URL=

SLOT_CONFIG=/absolute/path/to/data/slots/doctor_wuweiping.json
SLOT_HISTORY_TURNS=12
```

若前端或上游服务已经有 ASR，可在音频消息中直接附带最终转写，服务端将跳过二次 ASR：

```json
{
  "type": "input_audio",
  "input_audio": {
    "data": "base64-wav",
    "format": "wav",
    "transcript": "脸上的痘已经反复两个月了"
  }
}
```

槽位接口：

- `GET /api/v1/slots`：查询当前槽位、缺失项和最近历史。
- `POST /api/v1/slots/update`：上游 ASR 或文本客户端提交 `transcript`，也可提交人工确认的 `updates`。
- `POST /api/v1/slots/reset`：开始新患者会话或切换初诊/复诊。

所有接口沿用现有 `uid` 请求头。检查报告通过 `/api/v1/reports/analyze` 成功分析后，
会自动写入 `report_uploaded` 信号及 `exam_report`、`exam_analysis` 槽位。
