# 医疗问诊数据构建与训练

## 数据产物

运行：

```bash
python3 tools/build_wuweiping_vlm_pretriage_dataset.py \
  --output-dir data/wuweiping_vlm_pretriage build

python3 tools/validate_wuweiping_pretriage_dataset.py \
  data/wuweiping_vlm_pretriage
```

会生成两套相互隔离的数据：

- `train.json`、`validation.json`、`test.json`：实时音视频视觉问诊数据，只允许舌象、面象和患处图片。
- `report_upload_train.json`、`report_upload_validation.json`、`report_upload_test.json`：检查报告手动上传专用数据，只包含通过像素脱敏和二次隐私核验的报告。

实时数据不会包含检查报告、处方、病历或临床图与报告图的联合样本。检查报告数据中的处方页、病历页、既往诊断页、药品信息和直接身份编号会在构建时排除。

实时训练文件是混合数据集：视觉样本负责学习舌象/面象/患处的客观观察，真实无图多轮样本负责学习动态追问。多轮消息结构为：

```text
user: 初诊或复诊策略 + 已知基本信息
assistant: 医生首问
user: 患者真实回答
assistant: 根据上一回答提出的下一问
...
```

初诊策略覆盖主诉、现病史、过敏史、既往史、家族史、当前用药和必要生活/生育信息；复诊策略优先询问上次就诊后的变化、治疗执行、疗效、不良反应、新症状、复查结果及关键信息变化，避免机械重复稳定病史。

多轮对话只允许来自：

```text
outputs/medical_sft_minicpmo/tcm_consult_minicpmo.json
```

构建器不会直接使用原始 ASR 对话。源数据里的 `[意图：...]`、`【意图：...】` 和异常格式 `意图: ...]` 会在写入样本前删除；验证阶段如果任何 assistant 消息仍含“意图”，整个数据集校验失败。

视觉问诊 LoRA 默认读取实时数据集：

```bash
bash finetune/finetune_minicpmo_tcm_lora_stage2_vision.sh
```

## 运行时检查报告约束

实时视频帧发送时必须标记：

```json
{"type":"image_data","image_data":{"source":"realtime_video","data":"..."}}
```

实时服务会在模型前执行不含 OCR 的文档画面检测。疑似报告或其他文档帧会被丢弃，并提示用户使用手动上传入口。

检查报告只能调用：

```http
POST /api/v1/reports/analyze
uid: <session-id>
Content-Type: application/json
```

```json
{
  "source": "manual_upload",
  "mime_type": "image/jpeg",
  "image_data": "<base64>"
}
```

配置独立 VLM 接口：

```bash
export REPORT_VLM_BASE_URL="https://example.com/v1"
export REPORT_VLM_API_KEY="..."
export REPORT_VLM_MODEL="your-vlm-model"
```

未显式提供 `source=manual_upload` 的报告请求会被拒绝。上传分析不写入实时模型的会话缓存。
