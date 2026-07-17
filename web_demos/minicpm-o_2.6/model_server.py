import base64
import json
import asyncio
import numpy as np
import os, sys, io
import re
import threading
import time
import aiofiles
import librosa
import soundfile
import urllib.error
import urllib.request
from typing import Dict, List, Any, Optional
import argparse
import logging
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
import uvicorn
from fastapi import FastAPI, Header, Query, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

cur_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.abspath(cur_path))
import vad_utils
from realtime_video_policy import (
    REALTIME_DOCUMENT_NOTICE,
    document_likeness,
    is_manual_report_upload,
)

def setup_logger():
    logger = logging.getLogger("api_logger")
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d-%(levelname)s-[%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create handlers for stdout and stderr
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)  # INFO and DEBUG go to stdout
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)  # WARNING, ERROR, CRITICAL go to stderr
    stderr_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    return logger


app = FastAPI()
logger = setup_logger()

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int , default=32550)
ap.add_argument('--model', type=str , default="openbmb/MiniCPM-o-2_6", help="huggingface model name or local path")
ap.add_argument('--adapter', type=str, default="", help="optional LoRA adapter path")
ap.add_argument('--merge-lora', action=argparse.BooleanOptionalAction, default=True, help="merge LoRA adapter into the base model for inference")
ap.add_argument('--system-prompt', type=str, default="", help="assistant system prompt used for the realtime session")
ap.add_argument('--human-service-url', type=str, default="http://127.0.0.1:8010/aihuman", help="Wav2Lip digital-human service API base URL")
ap.add_argument('--report-vlm-base-url', type=str, default=os.getenv("REPORT_VLM_BASE_URL", ""), help="OpenAI-compatible VLM base URL for manual report uploads")
ap.add_argument('--report-vlm-api-key', type=str, default=os.getenv("REPORT_VLM_API_KEY", ""), help="VLM API key; prefer REPORT_VLM_API_KEY")
ap.add_argument('--report-vlm-model', type=str, default=os.getenv("REPORT_VLM_MODEL", ""), help="VLM model used only by the manual report-upload endpoint")
args = ap.parse_args()

TCM_SYSTEM_PROMPT = (
    "你是中医问诊助手，负责在初诊和复诊场景中进行真实、谨慎、连续的病情采集。"
    "你需要围绕主诉、现病史、既往史、过敏史、当前用药和外用药反应、饮食、睡眠、二便、"
    "寒热汗出、口渴口苦、疼痛性质、情绪压力、女性月经/孕产情况等逐步追问。"
    "如果用户提供舌面、面部、患处或其他图片/视频，应结合可见信息提出后续问诊问题，但不要仅凭图片下最终诊断。"
    "实时视频画面严禁读取、转写、概括或解释检查报告、处方、病历、证件及其他文档内容。"
    "如果镜头中出现疑似文档，只能请用户停止展示，并通过检查报告手动上传入口提交，由独立VLM接口分析。"
    "你的回答要像线上问诊医生一样简洁自然，一次优先问1到3个关键问题，避免长篇科普。"
    "不要替代医生做最终诊断、不要直接开处方或承诺疗效；涉及急症、严重过敏、持续高热、胸痛、呼吸困难、意识异常、"
    "孕产妇/儿童高风险情况时，应建议及时线下就医。"
)

TCM_IDENTITY_PROMPT = (
    "你的固定身份是吴卫平医生的数字助手。"
    "当用户询问‘你是谁’、‘你是什么模型’、‘你来自哪里’或类似身份问题时，"
    "应明确回答：‘我是吴卫平医生的数字助手。’"
    "不要自称面壁智能、MiniCPM、语言模型、大模型或其他公司/产品的助手，"
    "也不要主动讨论底层模型、训练机构或技术实现。"
)

CHINESE_VOICE_PROMPT = "模仿输入音频中的声音特征，并始终使用简体中文生成回复。"

REPORT_UPLOAD_PROMPT = """你是检查报告手动上传分析助手。图片来自用户主动选择文件后的独立上传入口，不是实时视频帧。
只读取图片中真实可见的检查项目、结果、单位、参考范围、异常箭头或异常标记；不得复述姓名、证件号、手机号、
地址、就诊号、住院号、条码或二维码。不得诊断、推测病因、推荐药物或开处方。看不清时明确写“未读清”。
仅返回 JSON 对象：
{
  "report_type": "报告类型或未知",
  "report_date": "YYYY-MM-DD或未知",
  "quality": "usable|limited|unusable",
  "items": [{"name":"项目名","value":"结果","unit":"单位或无","reference":"参考范围或无","flag":"高|低|正常|异常|未知"}],
  "summary": "不含身份信息的客观摘要",
  "limitations": ["OCR可能有误，请以报告原件和医生复核为准"],
  "next_questions": ["检查原因和当前症状"]
}
最多返回30个最清晰项目。仅输出 JSON。"""
REPORT_PRIVATE_FIELD_RE = re.compile(
    r"姓名|身份证|手机号|电话|地址|就诊号|住院号|病历号|条码|二维码|处方号"
)
REPORT_UNSUPPORTED_FIELD_RE = re.compile(
    r"处方|诊断|主诉|现病史|体格检查|病历|药品|用法|用量|嘱托|治疗记录"
)
DIRECT_IDENTIFIER_RE = re.compile(r"(?<!\d)(?:1\d{10}|\d{17}[\dXx]|\d{15})(?!\d)")


def _report_vlm_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    return base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"


def _extract_json_object(text_value: str) -> Dict[str, Any]:
    text_value = text_value.strip()
    if text_value.startswith("```"):
        text_value = text_value.strip("`").removeprefix("json").strip()
    start, end = text_value.find("{"), text_value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("VLM response does not contain a JSON object")
    value = json.loads(text_value[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("VLM response must be a JSON object")
    return value


def _sanitize_report_analysis(value: Dict[str, Any]) -> Dict[str, Any]:
    items = []
    unsupported = False
    for raw_item in value.get("items") or []:
        if not isinstance(raw_item, dict):
            continue
        name = str(raw_item.get("name") or "")[:80]
        item_value = str(raw_item.get("value") or "")[:120]
        if REPORT_PRIVATE_FIELD_RE.search(name) or DIRECT_IDENTIFIER_RE.search(item_value):
            continue
        if REPORT_UNSUPPORTED_FIELD_RE.search(name):
            unsupported = True
            continue
        items.append({
            "name": name,
            "value": item_value,
            "unit": str(raw_item.get("unit") or "")[:40],
            "reference": str(raw_item.get("reference") or "")[:80],
            "flag": str(raw_item.get("flag") or "未知")[:10],
        })
        if len(items) >= 30:
            break
    summary = DIRECT_IDENTIFIER_RE.sub("[已脱敏]", str(value.get("summary") or "")[:500])
    if unsupported and not items:
        summary = "该图片疑似处方、病历或诊断记录，不属于当前检查报告分析范围，请上传化验、影像、病理或其他客观检查报告。"
    return {
        "report_type": str(value.get("report_type") or "未知")[:80],
        "report_date": str(value.get("report_date") or "未知")[:20],
        "quality": value.get("quality") if value.get("quality") in {"usable", "limited", "unusable"} else "limited",
        "items": items,
        "summary": summary,
        "limitations": ["OCR可能有误，请以报告原件和医生复核为准"],
        "next_questions": [str(item)[:160] for item in (value.get("next_questions") or [])[:5]],
        "unsupported_document": unsupported and not items,
    }


def analyze_manually_uploaded_report(image_data: str, mime_type: str) -> Dict[str, Any]:
    if not args.report_vlm_base_url or not args.report_vlm_api_key or not args.report_vlm_model:
        raise RuntimeError("manual report VLM is not configured")
    try:
        image_bytes = base64.b64decode(image_data, validate=True)
        if not image_bytes or len(image_bytes) > 15 * 1024 * 1024:
            raise ValueError("report image must be between 1 byte and 15 MB")
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
    except Exception as exc:
        raise ValueError("invalid report image_data") from exc
    payload = {
        "model": args.report_vlm_model,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                {"type": "text", "text": REPORT_UPLOAD_PROMPT},
            ],
        }],
    }
    request = urllib.request.Request(
        _report_vlm_url(args.report_vlm_base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {args.report_vlm_api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    content = result["choices"][0]["message"]["content"]
    return _sanitize_report_analysis(_extract_json_object(content))


class StreamManager:
    def __init__(self):
        self.uid = None

        self.is_streaming_complete = threading.Event()
        self.conversation_started = threading.Event()
        self.last_request_time = None
        self.last_stream_time = None
        self.timeout = 900  # seconds timeout
        self.stream_timeout = 3  # seconds no stream
        self.num_stream = 0
        self.stream_started = False
        self.stop_response = False

        # VAD settings
        self.vad_options = vad_utils.VadOptions()
        self.vad_sequence_length = 5
        self.vad_sequence = []
        self.audio_prefill = []
        self.audio_input = []
        self.image_prefill = None
        self.report_frame_blocked = False
        self.audio_chunk = 200

        # customized options
        self.customized_audio = None
        self.customized_options = None

        # Omni model
        self.target_dtype = torch.bfloat16
        self.device='cuda:0'
        
        self.minicpmo_model_path = args.model #"openbmb/MiniCPM-o-2_6"
        self.model_version = "2.6"
        with torch.no_grad():
            print(f"*****minicpmo_model_path: {self.minicpmo_model_path}")
            self.minicpmo_model = AutoModel.from_pretrained(self.minicpmo_model_path, trust_remote_code=True, torch_dtype=self.target_dtype, attn_implementation='sdpa')
            if args.adapter:
                if PeftModel is None:
                    raise ImportError("peft is required when --adapter is provided. Please install peft first.")
                logger.info(f"Loading LoRA adapter from {args.adapter}")
                try:
                    self.minicpmo_model = PeftModel.from_pretrained(
                        self.minicpmo_model,
                        args.adapter,
                        torch_dtype=self.target_dtype,
                        torch_device="cpu",
                        low_cpu_mem_usage=True,
                    )
                except TypeError:
                    self.minicpmo_model = PeftModel.from_pretrained(
                        self.minicpmo_model,
                        args.adapter,
                        torch_dtype=self.target_dtype,
                    )
                if args.merge_lora:
                    logger.info("Merging LoRA adapter for inference")
                    self.minicpmo_model = self.minicpmo_model.merge_and_unload()
        self.minicpmo_tokenizer = AutoTokenizer.from_pretrained(self.minicpmo_model_path, trust_remote_code=True)
        self.minicpmo_model.init_tts()
        # self.minicpmo_model.tts.float()
        self.minicpmo_model.to(self.device).eval()

        self.ref_path_video_default = os.path.abspath(
            os.path.join(cur_path, "../../assets/ref_audios/Wuweiping_test3_ref_16k_mono.wav")
        )
        self.ref_path_default = "assets/ref_audios/default.wav"
        self.ref_path_female = "assets/ref_audios/female_example.wav"
        self.ref_path_male = "assets/ref_audios/male_example.wav"
        
        self.input_audio_id = 0
        self.input_audio_vad_id = 0
        self.input_image_id = 0
        self.output_audio_id = 0
        self.flag_decode = False
        self.cnts = None
        
        self.all_start_time = time.time()
        self.session_id = 233
        self.sys_prompt_flag = False
        self.vad_time = 0
        self.ls_time = 0
        self.msg_type = 1
        self.digital_human_session_id = None
        self.digital_human_msg_id = None
        self.digital_human_segment_index = 0
        
        self.speaking_time_stamp = 0
        self.cycle_wait_time = 12800/24000 + 0.15
        self.extra_wait_time = 2.5
        self.server_wait = True
        
        self.past_session_id = 0
        self.sys_prompt_init(0)
        self.session_id += 1
        
        
    def start_conversation(self):
        logger.info(f"uid {self.uid}: new conversation started.")
        self.conversation_started.set()
        self.stop_response = False

    def update_last_request_time(self):
        self.last_request_time = time.time()
        #logger.info(f"update last_request_time {self.last_request_time}")

    def update_last_stream_time(self):
        self.last_stream_time = time.time()
        #logger.info(f"update last_stream_time {self.last_stream_time}")

    def move_to_device(self, obj, device):
        if isinstance(obj, torch.Tensor):
            obj_ = obj.to(device)
            if (obj_.dtype == torch.float) or (obj_.dtype == torch.half):
                # cast to `torch.bfloat16`
                obj_ = obj_.to(self.target_dtype)
            return obj_
        elif isinstance(obj, dict):
            return {key: self.move_to_device(value, device) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.move_to_device(item, device) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.move_to_device(item, device) for item in obj)
        elif isinstance(obj, set):
            return {self.move_to_device(item, device) for item in obj}
        else:
            return obj
          
    def reset(self):
        logger.info("reset")
        self.is_streaming_complete.clear()
        self.conversation_started.clear()
        self.last_request_time = None
        self.last_stream_time = None
        self.audio_buffer_raw = bytearray()
        self.num_stream = 0
        self.stream_started = False
        self.stop_response = False
        # self.customized_audio = None
        # self.customized_options = None
        # clear model
        self.clear()

    def merge_wav_files(self, input_bytes_list, output_file, target_sr=16000):
        """Decode, normalize and concatenate browser WAV chunks.

        Browsers are allowed to ignore the requested AudioContext sample rate.  In
        that case a microphone commonly produces 48 kHz chunks.  Concatenating
        their PCM bytes and later relabelling them as 16 kHz slows the speech down
        and creates the characteristic metallic/mechanical sound.  Always decode
        the header and perform a real resample before the model/VAD sees the audio.
        """
        if not input_bytes_list:
            raise ValueError("cannot merge an empty audio chunk list")

        normalized_chunks = []
        for wav_bytes in input_bytes_list:
            audio, source_sr = soundfile.read(
                io.BytesIO(wav_bytes), dtype="float32", always_2d=True
            )
            if audio.size == 0:
                continue

            # The speech model expects mono input. Averaging is safer than silently
            # selecting one channel when a browser/device returns stereo audio.
            audio = np.mean(audio, axis=1)
            if source_sr != target_sr:
                audio = librosa.resample(
                    audio,
                    orig_sr=source_sr,
                    target_sr=target_sr,
                    res_type="kaiser_best",
                )
            normalized_chunks.append(np.asarray(audio, dtype=np.float32))

        if not normalized_chunks:
            raise ValueError("all received audio chunks are empty")

        # Suppress clicks caused by a dropped/restarted browser chunk. The sample
        # recording showed large discontinuities exactly on 200 ms boundaries.
        # Spread only abnormal jumps over 2 ms; normal continuous joins are left
        # untouched so speech detail is not blurred.
        fade_samples = max(1, int(target_sr * 0.002))
        for index in range(1, len(normalized_chunks)):
            previous = normalized_chunks[index - 1]
            current = normalized_chunks[index]
            if previous.size < fade_samples or current.size < fade_samples:
                continue
            join_delta = float(current[0] - previous[-1])
            if abs(join_delta) < 0.15:
                continue
            half_delta = join_delta / 2.0
            previous[-fade_samples:] += np.linspace(
                0.0, half_delta, fade_samples, dtype=np.float32
            )
            current[:fade_samples] -= np.linspace(
                half_delta, 0.0, fade_samples, dtype=np.float32
            )

        merged_audio = np.concatenate(normalized_chunks)
        # Guard against invalid samples and clipping before encoding PCM16.
        merged_audio = np.nan_to_num(merged_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        merged_audio = np.clip(merged_audio, -1.0, 1.0)
        soundfile.write(output_file, merged_audio, target_sr, subtype="PCM_16")

    
    def is_timed_out(self):
        if self.last_request_time is not None:
            return time.time() - self.last_request_time > self.timeout
        return False

    def no_active_stream(self):
        if self.last_stream_time is not None and self.stream_started:
            no_stream_duration = time.time() - self.last_stream_time
            if no_stream_duration > self.stream_timeout:
                #logger.info(f"no active stream for {no_stream_duration} secs.")
                return True
        return False

    def sys_prompt_init(self, msg_type):
        if self.past_session_id == self.session_id:
            return
        logger.info("### sys_prompt_init ###")

        logger.info(f'msg_type is {msg_type}')
        tcm_assistant_prompt = args.system_prompt.strip() or TCM_SYSTEM_PROMPT
        options = self.customized_options or {}
        patient_gender = str(options.get("patient_gender", "女") or options.get("gender") or "未知").strip()
        patient_age = str(options.get("patient_age", 35) or options.get("age") or "未知").strip()
        visit_type = str(options.get("visit_type", "初诊") or "未知").strip()
        patient_context = (
            f"当前用户基本信息：性别：{patient_gender}；年龄：{patient_age}；"
            f"就诊类型：{visit_type}。请结合这些信息开始问诊，不要重复询问已经提供的信息。"
        )
        language_instruction = "无论用户使用何种语言，都必须始终使用简体中文回答。"
        if msg_type <= 1: #audio
            audio_voice_clone_prompt = CHINESE_VOICE_PROMPT
            audio_assistant_prompt = tcm_assistant_prompt
            ref_path = self.ref_path_default

            
            if self.customized_options is not None:
                audio_assistant_prompt = self.customized_options.get('assistant_prompt') or tcm_assistant_prompt
                if self.customized_options['use_audio_prompt'] == 1:
                    ref_path = self.ref_path_default
                elif self.customized_options['use_audio_prompt'] == 2:
                    ref_path = self.ref_path_female
                elif self.customized_options['use_audio_prompt'] == 3:
                    ref_path = self.ref_path_male

            audio_assistant_prompt = (
                f"{TCM_IDENTITY_PROMPT}\n{patient_context}\n"
                f"{audio_assistant_prompt}\n{language_instruction}"
            )
            audio_prompt, sr = librosa.load(ref_path, sr=16000, mono=True)
            sys_msg = {'role': 'system', 'content': [audio_voice_clone_prompt + "\n", audio_prompt, "\n" + audio_assistant_prompt]}
        elif msg_type == 2: #video
            voice_clone_prompt=CHINESE_VOICE_PROMPT
            assistant_prompt=tcm_assistant_prompt
            ref_path = self.ref_path_video_default
            
            if self.customized_options is not None:
                assistant_prompt = self.customized_options.get('assistant_prompt') or tcm_assistant_prompt
                
            assistant_prompt = (
                f"{TCM_IDENTITY_PROMPT}\n{patient_context}\n"
                f"{assistant_prompt}\n{language_instruction}"
            )
            audio_prompt, sr = librosa.load(ref_path, sr=16000, mono=True)
            sys_msg = {'role': 'system', 'content': [voice_clone_prompt, audio_prompt, assistant_prompt]}
        # elif msg_type == 3: #user start
        #     assistant_prompt="作为助手，你将使用这种声音风格说话。"
        #     if self.customized_options is not None:
        #         assistant_prompt = self.customized_options['assistant_prompt']
                
        #     sys_msg = {'role': 'user', 'content': [assistant_prompt]}
        
        self.msg_type = msg_type
        msgs = [sys_msg]
        use_audio_prompt = (
            self.customized_options is None
            or self.customized_options.get('use_audio_prompt', 1) > 0
        )
        # Newer MiniCPM-o releases expose explicit cache reset APIs, while the
        # default 2.6 remote code may not. Re-anchor the vocoder when supported
        # without breaking deployments that still use the 2.6 API.
        if hasattr(self.minicpmo_model, "reset_session"):
            self.minicpmo_model.reset_session(reset_token2wav_cache=True)
        if hasattr(self.minicpmo_model, "init_token2wav_cache"):
            self.minicpmo_model.init_token2wav_cache(prompt_speech_16k=audio_prompt)
        if msg_type in (0, 2) or use_audio_prompt:
            self.minicpmo_model.streaming_prefill(
                session_id=str(self.session_id),
                msgs=msgs,
                tokenizer=self.minicpmo_tokenizer,
                is_last_chunk=True,
            )
            
        self.savedir = os.path.join(f"./log_data/{args.port}/", str(time.time()))
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        if not os.path.exists(self.savedir + "/input_audio_log"):
            os.makedirs(self.savedir + "/input_audio_log")
        if not os.path.exists(self.savedir + "/input_audio_vad_log"):
            os.makedirs(self.savedir + "/input_audio_vad_log")
        if not os.path.exists(self.savedir + "/input_image_log"):
            os.makedirs(self.savedir + "/input_image_log")
        if not os.path.exists(self.savedir + "/output_audio_log"):
            os.makedirs(self.savedir + "/output_audio_log")
        if not os.path.exists(self.savedir + "/feedback_log"):
            os.makedirs(self.savedir + "/feedback_log")
        if not os.path.exists(self.savedir + "/input_audio"):
            os.makedirs(self.savedir + "/input_audio")
        
        self.past_session_id = self.session_id
        self.audio_prefill = []
        self.audio_input = []
        
    def clear(self):
        try:
            self.flag_decode = False
            self.stream_started = False
            self.cnts = None
            self.vad_sequence = []
            self.audio_prefill = []
            self.audio_input = []
            self.image_prefill = None
            self.report_frame_blocked = False
            
            if self.minicpmo_model.llm_past_key_values[0][0].shape[2]>8192:
                self.session_id += 1  # to clear all kv cache
                self.sys_prompt_flag = False

            self.vad_time = 0
            self.ls_time = 0
            self.msg_type = 1
            
        except Exception as e:
            raise ValueError(f"Clear error: {str(e)}")
    
    
    def process_message(self, message: Dict[str, Any]):
        try:
            # Process content items
            audio_data = None
            image_data = None
            for content_item in message["content"]:
                if content_item["type"] == "stop_response":
                    logger.info("process_message: received request to stop_response")
                    self.stop_response = True
                    return "stop"
                elif content_item["type"] == "input_audio":
                    audio_data = content_item["input_audio"]["data"]
                    audio_timestamp = content_item["input_audio"].get("timestamp", "")
                elif content_item["type"] == "image_data":
                    if content_item["image_data"].get("source", "realtime_video") != "realtime_video":
                        raise ValueError("stream image_data only accepts source=realtime_video")
                    image_data = content_item["image_data"]["data"]
            if audio_data is None:
                return "empty audio"

            if self.conversation_started.is_set() and self.is_streaming_complete.is_set():
                logger.info("conversation not started or still in generation, skip stream message.")
                return "skip"

            if self.flag_decode:
                return "skip"

            try:
                audio_bytes = base64.b64decode(audio_data)

                image = None
                if image_data is not None:
                    if len(image_data) > 0:
                        image_bytes = base64.b64decode(image_data)
                        image_buffer = io.BytesIO(image_bytes)
                        image_buffer.seek(0)
                        candidate_image = Image.open(image_buffer).convert("RGB")
                        blocked, metrics = document_likeness(candidate_image)
                        if blocked:
                            logger.warning(f"blocked document-like realtime frame: {metrics}")
                            self.image_prefill = None
                            self.report_frame_blocked = True
                        else:
                            image = candidate_image
                        # logger.info("read image")

                if self.sys_prompt_flag is False:
                    self.all_start_time = time.time()
                    self.sys_prompt_flag = True
                    if image_data is not None:
                        self.sys_prompt_init(2)
                    else:
                        self.sys_prompt_init(1)
                    
                self.prefill(audio_bytes, image, False)
                
                self.vad_sequence.append(audio_bytes)
                if len(self.vad_sequence) < self.vad_sequence_length:
                    # logger.info('length of vad_sequence is {}, insufficient'.format(self.vad_sequence_length))
                    return "done"
                elif len(self.vad_sequence) > self.vad_sequence_length:
                    # logger.info('length of vad_sequence exceeds {}'.format(self.vad_sequence_length))
                    self.vad_sequence.pop(0)
                self.vad_check_audio_bytes(audio_bytes, image, 16000)

                return "done"

            except Exception as e:
                raise ValueError(f"Audio processing error: {str(e)}")

        except Exception as e:
            raise ValueError(f"Message processing error: {str(e)}")

    def resample_audio(self, input_path, src_sr, tar_sr, output_path):
        audio_data, _ = librosa.load(input_path, sr=src_sr)
        audio_new = librosa.resample(audio_data, orig_sr=src_sr, target_sr=tar_sr)
        soundfile.write(output_path, audio_new, tar_sr)

    def calculate_rms(self, input_path, sr):
        audio_data, _ = librosa.load(input_path, sr=sr)
        return (np.sqrt(np.mean(audio_data**2)) > 0.002)

    def vad_check_audio_bytes(self, audio, image, sr):
        try:
            input_audio_vad_path = self.savedir + f"/input_audio_vad_log/vad_{self.input_audio_vad_id}.wav"
            self.input_audio_vad_id += 1
            self.merge_wav_files(self.vad_sequence, input_audio_vad_path)

            with open(input_audio_vad_path,"rb") as f:
                temp_audio = f.read()
            dur_vad, vad_audio_bytes, time_vad = vad_utils.run_vad(temp_audio, sr, self.vad_options)
            if self.customized_options is not None:
                vad_threshold = 1 - self.customized_options['vad_threshold']
            else:
                vad_threshold = 0.2
                
            if self.calculate_rms(input_audio_vad_path, sr) and dur_vad > 0.4:
                if self.stream_started == False:
                    self.vad_time = time.time()
                    self.stream_started = True
            elif dur_vad < vad_threshold:
                if self.stream_started:
                    self.stream_started = False
                    if (time.time() - self.vad_time >= 0.6):
                        self.prefill(audio, image, True)
                        self.is_streaming_complete.set()
                        # self.ls_time = time.time()
                                
        except Exception as e:
            logger.error(f"VAD error: {e}")
            raise
        return

    def prefill(self, audio, image, is_end):
        if self.server_wait:   
            now = time.time()
            await_time = self.speaking_time_stamp - now + self.extra_wait_time
            if await_time > 0:
                return False
        
        if self.flag_decode:
            return False
        
        if image is not None:
            self.image_prefill = image
        try:
            if is_end == False:
                self.audio_prefill.append(audio)
                self.audio_input.append(audio)
            slice_nums = 1
            if is_end and self.customized_options is not None:
                if self.customized_options['hd_video']:
                    slice_nums = 6
                else:
                    return True
            if (len(self.audio_prefill) == (1000/self.audio_chunk)) or (is_end and len(self.audio_prefill)>0):
                time_prefill = time.time()
                input_audio_path = self.savedir + f"/input_audio_log/input_audio_{self.input_audio_id}.wav"
                self.merge_wav_files(self.audio_prefill, input_audio_path)
                audio_np, sr = soundfile.read(input_audio_path, dtype='float32')
                self.audio_prefill = []

                if len(audio_np) > 16000:
                    audio_np = audio_np[:16000] 

                with torch.no_grad():
                    if self.image_prefill is not None:
                        input_image_path = self.savedir + f'/input_image_log/input_image_{self.input_audio_id}.png'
                        self.image_prefill.save(input_image_path, 'PNG')
                        self.image_prefill = self.image_prefill.convert("RGB")
                        
                    cnts = None
                    if self.image_prefill is not None:
                        cnts = ["<unit>", self.image_prefill, audio_np]
                    elif self.report_frame_blocked:
                        cnts = [REALTIME_DOCUMENT_NOTICE, audio_np]
                        self.report_frame_blocked = False
                    else:
                        cnts = [audio_np]
                        
                    if cnts is not None:
                        msg = {"role":"user", "content": cnts}
                        msgs = [msg]
                        res = self.minicpmo_model.streaming_prefill(
                            session_id=str(self.session_id),
                            msgs=msgs, 
                            tokenizer=self.minicpmo_tokenizer,
                            max_slice_nums=slice_nums,
                        )

                self.input_audio_id += 1
            return True

        except Exception as e:
            logger.error(f"prefill error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def generate_end(self):
        self.input_audio_id += 10
        self.output_audio_id += 10
        self.flag_decode = False
        self.reset()
        return

    async def generate(self):
        """ return audio bytes and response text (optional) """
        if self.stop_response:
            self.generate_end()
            return

        self.flag_decode = True
        try:
            with torch.no_grad():
                logger.info("=== model gen start ===")
                time_gen = time.time()
                input_audio_path = self.savedir + f"/input_audio/all_input_audio_{self.input_audio_id}.wav"
                self.merge_wav_files(self.audio_input, input_audio_path)
                audio_stream = None
                try:
                    with open(input_audio_path, 'rb') as wav_file:
                        audio_stream = wav_file.read()
                except FileNotFoundError:
                    print(f"File {input_audio_path} not found.")
                yield base64.b64encode(audio_stream).decode('utf-8'), "assistant:\n", None
                
                print('=== gen start: ', time.time() - time_gen)
                first_time = True
                temp_time = time.time()
                temp_time1 = time.time()
                with torch.inference_mode():
                    if self.stop_response:
                        self.generate_end()
                        return
                    self.minicpmo_model.config.stream_input=True
                    msg = {"role":"user", "content": self.cnts}
                    msgs = [msg]
                    text = ''
                    self.speaking_time_stamp = time.time()
                    try:
                        for r in self.minicpmo_model.streaming_generate(
                            session_id=str(self.session_id),
                            tokenizer=self.minicpmo_tokenizer,
                            generate_audio=True,
                            # enable_regenerate=True,
                        ):
                            if self.stop_response:
                                self.generate_end()
                                return
                            audio_np, sr, text = r["audio_wav"], r["sampling_rate"], r["text"]

                            output_audio_path = self.savedir + f'/output_audio_log/output_audio_{self.output_audio_id}.wav'
                            self.output_audio_id += 1
                            soundfile.write(output_audio_path, audio_np, samplerate=sr)
                            audio_stream = None
                            try:
                                with open(output_audio_path, 'rb') as wav_file:
                                    audio_stream = wav_file.read()
                            except FileNotFoundError:
                                print(f"File {output_audio_path} not found.")
                            temp_time1 = time.time()
                            print('text: ', text)
                            human_delivered = None
                            if self.digital_human_session_id:
                                segment_index = self.digital_human_segment_index
                                self.digital_human_segment_index += 1
                                segment_id = f"{self.digital_human_msg_id}_{segment_index}"
                                try:
                                    await post_human_service(
                                        "/external_audio",
                                        {
                                            "audio_source": "minicpm_o_tts",
                                            "session_id": self.digital_human_session_id,
                                            "msg_id": self.digital_human_msg_id,
                                            "segment_id": segment_id,
                                            "segment_index": segment_index,
                                            "text": text,
                                            "audio": base64.b64encode(audio_stream).decode('utf-8'),
                                        },
                                    )
                                    human_delivered = True
                                except Exception as exc:
                                    logger.error(f"digital human audio forwarding failed: {exc}")
                                    human_delivered = False
                            yield base64.b64encode(audio_stream).decode('utf-8'), text, human_delivered
                            self.speaking_time_stamp += self.cycle_wait_time
                    except Exception as e:
                        logger.error(f"Error happened during generation: {str(e)}")
                    yield None, '\n<end>', None

        except Exception as e:
            logger.error(f"发生异常:{e}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            logger.info(f"uid {self.uid}: generation finished!")
            self.generate_end()

    async def check_activity(self):
        while True:
            # Check for overall inactivity (30 minutes)
            if self.is_timed_out():
                self.reset()
            if self.no_active_stream() and not self.is_streaming_complete.is_set():
               self.is_streaming_complete.set()

            await asyncio.sleep(1)  # Check every second

    def upload_customized_audio(self, audio_data, audio_fmt):
        self.customized_audio = None
        try:
            if audio_data is not None and len(audio_data) > 0:
                # if audio_fmt == "mp3" or audio_fmt == "wav":
                audio_bytes = base64.b64decode(audio_data)
                fio = io.BytesIO(audio_bytes)
                fio.seek(0)
                audio_np, sr = librosa.load(fio, sr=16000, mono=True)
                if audio_np is not None and len(audio_np) > 1000:
                    output_audio_path = self.savedir + f'/customized_audio.wav'
                    soundfile.write(output_audio_path, audio_np, sr)
                    self.customized_audio = output_audio_path
                    logger.info(f"processed customized {audio_fmt} audio")
                    print(audio_np.shape, type(audio_np), sr)
            else:
                logger.info(f"empty customized audio, use default value instead.")
                self.customized_audio = None
        except Exception as e:
            raise ValueError(f"Process customized audio error: {str(e)}")

    def update_customized_options(self, uid, options):
        self.customized_options = None
        if options is None:
            raise ValueError("Invalid None type for options, expected dict type")
        self.customized_options = options
        logger.info(f"uid: {uid} set customized_options to {options}")


stream_manager = StreamManager()


async def post_human_service(path: str, payload: dict) -> dict:
    url = f"{args.human_service_url.rstrip('/')}{path}"

    def _post():
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"human service returned {exc.code}: {detail}") from exc

    return await asyncio.to_thread(_post)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting application and activity checker")
    asyncio.create_task(stream_manager.check_activity())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application")

@app.post("/stream")
@app.post("/api/v1/stream")
async def stream(request: Request, uid: Optional[str] = Header(None)):
    global stream_manager

    stream_manager.update_last_request_time()
    stream_manager.update_last_stream_time()

    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid in headers")
    if stream_manager.uid is not None and stream_manager.uid != uid:
        logger.error(f"uid changed during steram: previous uid {stream_manager.uid}, new uid {uid}")
        raise HTTPException(status_code=400, detail="uid changed in stream")

    try:
        # Parse JSON request
        data = await request.json()

        # Validate basic structure
        if not isinstance(data, dict) or "messages" not in data:
            raise HTTPException(status_code=400, detail="Invalid request format")

        # Process messages
        reason = ""
        for message in data["messages"]:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise HTTPException(status_code=400, detail="Invalid message format")
            reason = stream_manager.process_message(message)

        # Return response using uid from header
        response = {
            "id": uid,
            "choices": {
                "role": "assistant",
                "content": "success",
                "finish_reason": reason
            }
        }
        return JSONResponse(content=response, status_code=200)

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/stream")
@app.websocket("/ws/api/v1/stream")
async def websocket_stream(websocket: WebSocket,
                           uid: Optional[str] = Query(None)):
    global stream_manager

    if not uid:
        await websocket.close(code=400, reason="Missing uid in request")
        return

    # Accept the WebSocket connection
    await websocket.accept()

    #if stream_manager.uid is not None and stream_manager.uid != uid:
    #    logger.error(f"uid changed during steram: previous uid {stream_manager.uid}, new uid {uid}")
    #    await websocket.close(code=400, reason="Uid changed in stream.")
    #    return

    try:
        while True:
           # Continuously listen for incoming messages from the client
           data = await websocket.receive_text()

           # Parse JSON request
           try:
               request_data = json.loads(data)
           except json.JSONDecodeError:
               await websocket.send_json({"error": "Invalid JSON"})
               continue

           stream_manager.update_last_request_time()
           stream_manager.update_last_stream_time()

           if stream_manager.uid is not None and stream_manager.uid != uid:
               logger.error(f"uid changed during stream: previous uid {stream_manager.uid}, new uid {uid}")
               await websocket.send_json({"error": "UID changed in stream"})
               continue

           # Validate basic structure
           if not isinstance(request_data, dict) or "messages" not in request_data:
               await websocket.send_json({"error": "Invalid request format"})
               continue

           # Process messages
           try:
               reason = ""
               for message in request_data["messages"]:
                   if not isinstance(message, dict) or "role" not in message or "content" not in message:
                       await websocket.send_json({"error": "Invalid message format"})
                       continue
                   reason = stream_manager.process_message(message)

               # Respond with success message
               response = {
                   "id": uid,
                   "choices": {
                       "role": "assistant",
                       "content": "success",
                       "finish_reason": reason,
                   },
               }
               await websocket.send_json(response)
           except WebSocketDisconnect:
               # Handle WebSocket disconnection
               break
           except Exception as e:
               logger.error(f"process message error: {str(e)}")
               await websocket.close(code=1011, reason =f"Internal server error: {str(e)}")

    except WebSocketDisconnect:
        # Handle WebSocket disconnection
        return
    except Exception as e:
        logger.error(f"ws_stream error: {str(e)}")
        await websocket.close(code=1011, reason =f"Unexpected error: {str(e)}")


async def generate_sse_response(request: Request, uid: Optional[str] = Header(None)):
    global stream_manager
    print(f"uid: {uid}")
    try:
        # Wait for streaming to complete or timeout
        while not stream_manager.is_streaming_complete.is_set():
            # if stream_manager.is_timed_out():
            #     yield f"data: {json.dumps({'error': 'Stream timeout'})}\n\n"
            #     return
            # print(f"{uid} whille not stream_manager.is_streaming_complete.is_set(), asyncio.sleep(0.1)")
            await asyncio.sleep(0.1)

        logger.info("streaming complete\n")
        # Generate response
        try:
            yield f"event: message\n"
            async for audio, text, human_delivered in stream_manager.generate():
                if text == "stop":
                    break
                res = {
                    "id": stream_manager.uid,
                    "response_id": stream_manager.output_audio_id,
                    "choices": [
                        {
                            "role": "assistant",
                            "audio": audio,
                            "text": text,
                            "digital_human_audio": human_delivered,
                            "finish_reason": "processing"
                        }
                    ]
                }
                # logger.info("generate_sse_response yield response")
                yield f"data: {json.dumps(res)}\n\n"
                await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Error while generation: {str(e)}")
            yield f'data:{{"error": "{str(exc)}"}}\n\n'
    except Exception as e:
        yield f'data:{{"error": "{str(e)}"}}\n\n'

@app.post("/completions")
@app.post("/api/v1/completions")
async def completions(request: Request, uid: Optional[str] = Header(None)):
    global stream_manager

    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid in headers")

    try:
        # if stream_manager.uid is not None and stream_manager.uid != uid:
        if stream_manager.uid != uid:
        #     stream_manager.stop_response = True
        #     logger.info(f"uid changed, reset model: previous uid {stream_manager.uid}, new uid {uid}")
            stream_manager.session_id += 1
            stream_manager.sys_prompt_flag = False
            stream_manager.reset()

            # raise HTTPException(
            #    status_code=409,
            #    detail="User id changed, reset context."
            # )
        stream_manager.speaking_time_stamp = 0
        stream_manager.update_last_request_time()
        stream_manager.uid = uid
        stream_manager.start_conversation()

        data = await request.json()
        human_session_id = data.get("digital_human_session_id")
        stream_manager.digital_human_session_id = int(human_session_id) if human_session_id else None
        stream_manager.digital_human_msg_id = str(
            data.get("digital_human_msg_id") or f"minicpm_{stream_manager.session_id}_{int(time.time() * 1000)}"
        )
        stream_manager.digital_human_segment_index = 0

        return StreamingResponse(
            generate_sse_response(request, uid),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked"
            }
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Server busy, please try again later"
        )
    except Exception as e:
        logger.error(f"Error processing request for user {uid}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop")
@app.post("/api/v1/stop")
async def stop_response(request: Request, uid: Optional[str] = Header(None)):
    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid in headers")

    global stream_manager
    # stream_manager.session_id += 1
    logger.info(f"uid {uid}: received stop_response")
    stream_manager.stop_response = True
    if stream_manager.digital_human_session_id:
        try:
            await post_human_service(
                "/interrupt_talk",
                {
                    "session_id": stream_manager.digital_human_session_id,
                    "msg_id": stream_manager.digital_human_msg_id or f"stop_{int(time.time() * 1000)}",
                },
            )
        except Exception as exc:
            logger.warning(f"digital human interrupt failed: {exc}")
    response = {
        "id": uid,
        "choices": {
            "role": "assistant",
            "content": "success",
            "finish_reason": "stop"
        }
    }
    return JSONResponse(content=response, status_code=200)


@app.post("/api/v1/reports/analyze")
async def analyze_report_upload(request: Request, uid: Optional[str] = Header(None)):
    """Analyze an explicitly uploaded report outside the realtime session."""
    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid in headers")
    payload = await request.json()
    if not is_manual_report_upload(payload):
        raise HTTPException(
            status_code=400,
            detail="Reports must use the explicit manual-upload path with source=manual_upload",
        )
    image_data = payload.get("image_data", "")
    mime_type = payload.get("mime_type", "image/jpeg")
    if not isinstance(image_data, str) or not image_data:
        raise HTTPException(status_code=400, detail="Missing base64 image_data")
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Unsupported report image type")
    try:
        result = await asyncio.to_thread(analyze_manually_uploaded_report, image_data, mime_type)
        return JSONResponse(
            content={
                "id": uid,
                "source": "manual_upload",
                "analysis": result,
                "disclaimer": "OCR可能有误，请以报告原件和医生复核为准。",
            },
            status_code=200,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except urllib.error.HTTPError as exc:
        logger.error(f"report VLM HTTP error: {exc.code}")
        raise HTTPException(status_code=502, detail="Report VLM request failed")
    except urllib.error.URLError as exc:
        logger.error(f"report VLM network error: {exc.reason}")
        raise HTTPException(status_code=502, detail="Report VLM is unavailable")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/v1/digital-human/offer")
async def digital_human_offer(request: Request):
    payload = await request.json()
    payload["external_audio"] = True
    try:
        return JSONResponse(content=await post_human_service("/offer", payload), status_code=200)
    except Exception as exc:
        logger.error(f"digital human offer failed: {exc}")
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/api/v1/digital-human/close")
async def digital_human_close(request: Request):
    payload = await request.json()
    try:
        return JSONResponse(content=await post_human_service("/webrtc_graceful_close", payload), status_code=200)
    except Exception as exc:
        logger.warning(f"digital human close failed: {exc}")
        return JSONResponse(content={"code": -1, "msg": str(exc)}, status_code=200)

@app.post("/feedback")
@app.post("/api/v1/feedback")
async def feedback(request: Request, uid: Optional[str] = Header(None)):
    global stream_manager

    # Validate the 'uid' header
    if not uid:
        raise HTTPException(status_code=400, detail="Missing 'uid' header")

    try:
        data = await request.json()
        if "response_id" not in data or "rating" not in data:
            raise HTTPException(status_code=400, detail="Invalid request: must have response_id and rating")
        response_id = data.get("response_id", "")
        rating = data.get("rating", "")
        comment = data.get("comment", "")
        # Validate the rating
        if rating not in ["like", "dislike"]:
            raise HTTPException(status_code=400, detail=f"Invalid rating value: {rating}")

        # Define the log file path
        log_file_path = f"{stream_manager.savedir}/feedback_log/{response_id}.{rating}"
        # Write the feedback to the file asynchronously
        async with aiofiles.open(log_file_path, mode="a") as file:
            await file.write(f"model: {stream_manager.minicpmo_model_path}\nuid {uid}: {comment}\n")
        response = {
            "id": uid,
            "choices": {
                "role": "assistant",
                "content": "success",
                "finish_reason": "done"
            }
        }
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        logger.error(f"Error processing feedback for user {uid}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/init_options")
@app.post("/api/v1/init_options")
async def init_options(request: Request, uid: Optional[str] = Header(None)):
    global stream_manager

    stream_manager.update_last_request_time()

    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid in headers")
    try:
        # Parse JSON request
        data = await request.json()

        # Validate basic structure
        if not isinstance(data, dict) or "messages" not in data:
            raise HTTPException(status_code=400, detail="Invalid request format")

        messages = data.get("messages", [])
        for message in messages:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise HTTPException(status_code=400, detail="Invalid message format")

            for content in message.get("content", []):
                if content["type"] == "input_audio":
                    audio_data = content["input_audio"].get("data", "")
                    audio_fmt = content["input_audio"].get("format", "")
                    stream_manager.upload_customized_audio(audio_data, audio_fmt)
                elif content["type"] == "options":
                    stream_manager.update_customized_options(uid, content["options"])
                else:
                    ctype = content["type"]
                    raise HTTPException(status_code=400, detail=f"Invalid content type: {ctype}")
        version = stream_manager.model_version
        print(version)
        response = {
            "id": uid,
            "choices": {
                "role": "assistant",
                "content": version,
                "finish_reason": "done"
            }
        }
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"init options error: {str(e)}")


@app.get('/health')
@app.get('/api/v1/health')
async def health_check():
    return {"status": "OK"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
