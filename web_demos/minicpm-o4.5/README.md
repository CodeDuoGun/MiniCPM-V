# MiniCPM-o 4.5 中医实时问诊

这是与 `minicpm-o_2.6` 隔离的新实现。目标是使用官方 MiniCPM-o 4.5 全双工接口，保留现有
上下文、有限对话历史、初诊/复诊槽位规则和前端消息格式，并提供微调、医学评测、压测和灰度工具。

## 当前交付

- `app/runtime.py`：官方原版模型加载、`as_duplex()`、`prepare()`、音视频块 prefill/generate。
- `app/server.py`：兼容原 HTTP/SSE 接口，并提供全双工 WebSocket、取消、会话隔离和 Prometheus 指标。
- `app/context.py`、`app/slot_manager.py`：纳入与模型无关的既有槽位引擎，保留初诊/复诊规则、证据和最近历史。
- `data/`、`llamafactory/`：经过格式、图片路径和医疗越界过滤的 ShareGPT 多模态数据及 LoRA 配置。
- `evaluation/`：医生专项用例模板和 required/forbidden 自动门禁。
- `scripts/load_test.py`：并发及 p50/p95/p99 延迟测试。
- `deploy/`：GPU 镜像、按 uid 粘滞的灰度模板和回滚门槛。

## 1. 独立环境

建议 Python 3.10 和独立虚拟环境：

```bash
python3.10 -m venv .venv-o45
source .venv-o45/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install "setuptools==80.9.0"
# CUDA 12.6 wheel（驱动支持 CUDA 12.6 或更高时使用）
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r web_demos/minicpm-o4.5/requirements.txt
cp web_demos/minicpm-o4.5/.env.example web_demos/minicpm-o4.5/.env
```

`minicpmo-utils` 当前固定依赖 `librosa==0.9.0`，该版本依赖已经从 Setuptools 82 移除的
`pkg_resources`。因此本目录固定 `setuptools==80.9.0` 并限制 `numpy<2`；不要在该环境中
单独升级 setuptools、librosa 或 NumPy 2.x。

机器还需要 NVIDIA 驱动、可用 CUDA 和 FFmpeg。不要把 `.env` 或患者音视频提交到 Git。

## 2. 原版模型与官方双工接口

先不加载 LoRA：

```bash
python web_demos/minicpm-o4.5/scripts/check_torch_stack.py
python web_demos/minicpm-o4.5/scripts/smoke_model.py
python web_demos/minicpm-o4.5/scripts/official_duplex_demo.py
```

然后启动服务：

```bash
bash web_demos/minicpm-o4.5/run_backend.sh
curl http://127.0.0.1:32560/ready
```

没有 GPU 时可设置 `LOAD_MODEL=false` 验证协议、槽位及前端，但这不算模型验收。

## 3. 上下文、历史和槽位

每个 `uid` 拥有独立 `ConsultationContext`。患者转写进入既有 `SlotConversation`，它继续使用
`data/slots/doctor_wuweiping.json` 的初诊/复诊、条件槽位、证据置信度和最近 12 轮规则。

全双工声学流中途不修改模型状态。仅当 `result.end_of_turn` 或客户端
`input_audio.end_of_turn=true` 时，服务才会：

1. 更新槽位；
2. 记录完整助手回复；
3. 把有限历史与槽位快照重新构造成系统上下文；
4. 在下一轮开始前重新 `prepare()`。

客户端最好附带浏览器/上游 ASR 的最终 `transcript`；否则可配置 DashScope ASR。

### 图片质控、分析与问诊级存储

初始化会话时在 `options` 中传入稳定的 `consultation_id`。助手会按舌面、患处、检查报告的顺序
主动索取尚未完成的资料，用户也可随时主动上传。三类图片统一调用：

```http
POST /api/v1/images/analyze
uid: patient-or-session-id
Content-Type: application/json

{
  "consultation_id": "consultation-20260720-001",
  "scene": "tongue",
  "source": "assistant_requested",
  "mime_type": "image/jpeg",
  "image_data": "<base64，不含 data URL 前缀>"
}
```

`scene` 为 `tongue|lesion|report`，`source` 为 `assistant_requested|manual_upload`。接口先调用
VLM 质控；质控通过后才执行结构化分析并写入病情槽位。质控不通过的图片和质控结论仍会保存，
但不会进入模型后续问诊上下文。检查报告必须使用该显式上传接口，不允许从实时视频帧读取。

图片经 `utils/qiniu_lib.py` 上传，Redis 按
`<REDIS_KEY_PREFIX>:<consultation_id>` 保存患者基本信息、槽位病情、对话、图片 HTTP 地址、
质控结果和分析结果。可通过以下接口查询完整记录：

```http
GET /api/v1/consultations/{consultation_id}
uid: patient-or-session-id
```

生产环境必须配置 `VISION_VLM_*`、`REDIS_URL` 和 `QINIU_*`；密钥只放环境变量，不写入仓库。

## 4. LLaMA-Factory 数据与 LoRA

已生成：

- `data/tcm_o45_train.json`
- `data/tcm_o45_validation.json`
- `data/tcm_o45_test.json`

重新构造时运行：

```bash
python web_demos/minicpm-o4.5/scripts/build_llamafactory_dataset.py \
  --source outputs/medical_sft_minicpmo/tcm_consult_minicpmo_train.json \
  --source data/wuweiping_vlm_pretriage/train.json \
  --output web_demos/minicpm-o4.5/data/tcm_o45_train.json \
  --report web_demos/minicpm-o4.5/data/tcm_o45_train.report.json
```

默认会剔除具体处方、用量、确诊和实时文档泄漏样本。不要用
`--allow-medical-advice` 训练面向患者的线上模型。

使用包含 MiniCPM-o 4.5 支持的 LLaMA-Factory main 固定 commit，确认预处理 smoke test 后运行：

```bash
# LlamaFactory v0.9.4 的依赖范围会选择不兼容的新版本，必须整组重新固定。
pip install --upgrade --force-reinstall --no-cache-dir \
  -r web_demos/minicpm-o4.5/llamafactory/requirements-compat.txt
pip check
bash web_demos/minicpm-o4.5/llamafactory/run_train.sh
```

第一阶段只训练 Qwen3 LLM LoRA，不训练 Whisper、CosyVoice2 和全双工控制模块。生成的 Adapter
需要在独立评测通过后，才可写入 `LORA_ADAPTER`。2.6 Adapter 不能复用。

## 5. 评测、压测和灰度

让医生复制并补全 `evaluation/cases.template.jsonl`，覆盖舌象、面象、患处、图片质量、急症、
儿童/孕哺、过敏、幻觉、文档隐私和拒答。保存模型输出为 `{\"id\":...,\"output\":...}` JSONL 后运行：

```bash
python web_demos/minicpm-o4.5/scripts/medical_eval.py \
  --cases web_demos/minicpm-o4.5/evaluation/cases.jsonl \
  --predictions web_demos/minicpm-o4.5/evaluation/predictions.jsonl \
  --report web_demos/minicpm-o4.5/evaluation/report.json

python web_demos/minicpm-o4.5/scripts/load_test.py \
  --base-url http://127.0.0.1:32560 --concurrency 1 --rounds 20
```

单个双工模型实例是有状态的，默认只允许一个活动患者；生产并发采用一 GPU 一实例并按 `uid`
粘滞路由。灰度顺序和回滚红线见 `deploy/ROLLOUT.md`。

## 尚需外部验收

仓库内可以完成代码、数据和 mock 协议测试，但以下步骤必须在目标 NVIDIA GPU 和真实浏览器/摄像头上执行：

- 下载并加载约 9B 原版权重；
- 实际“边看、边听、边说”和插话打断；
- LoRA 训练及合并后的音频/TTS 回归；
- 医生双人盲评；
- 高峰压测、长稳测试和逐级灰度。
