<template>
    <!-- <ExtraInfo webVersion="非websocket_0111" :modelVersion="modelVersion" /> -->
    <div class="video-page">
        <div class="video-page-header">
            <div class="voice-container" v-if="!isCalling">
                <SvgIcon name="voice" class="voice-icon" />
                <SvgIcon name="voice" class="voice-icon" />
                <SvgIcon name="voice" class="voice-icon" />
            </div>
            <div class="voice-container" v-else>
                <Voice
                    :dataArray="dataArray"
                    :isCalling="isCalling"
                    :isPlaying="playing"
                    :configList="videoConfigList"
                    :boxStyle="{ height: '45px' }"
                    :itemStyle="{ width: '3px', margin: '0 1px' }"
                />
            </div>
            <!-- <SelectTimbre v-model:timbre="timbre" v-model:audioData="audioData" v-model:disabled="isCalling" /> -->
            <div class="digital-human-switch">
                <span>数字人</span>
                <el-switch v-model="digitalHumanEnabled" :disabled="isCalling" />
            </div>
        </div>
        <div class="video-page-content">
            <div class="video-page-content-video patient-panel" v-loading="loading" element-loading-background="#f3f3f3">
                <div class="panel-title">患者</div>
                <video ref="videoRef" autoplay playsinline muted />
                <canvas ref="canvasRef" canvas-id="canvasId" style="display: none" />
                <div class="switch-camera" v-if="isMobile()" @click="switchCamera">
                    <SvgIcon name="switch-camera" class="icon" />
                </div>
            </div>
            <div
                class="video-page-content-right avatar-panel"
                :class="{ 'human-disabled': !digitalHumanEnabled }"
            >
                <div class="panel-title">{{ digitalHumanEnabled ? '数字人医生' : '问诊记录' }}</div>
                <div v-if="digitalHumanEnabled" class="avatar-video-wrap">
                    <video ref="avatarVideoRef" autoplay playsinline />
                    <div v-if="!digitalHumanReady" class="avatar-state">
                        {{ digitalHumanConnecting ? '数字人连接中…' : '数字人服务未连接' }}
                    </div>
                </div>
                <div class="output-content">
                    <ModelOutput
                        v-if="outputData.length > 0"
                        :outputData="outputData"
                        containerClass="output-content"
                    />
                </div>
                <div class="skip-box">
                    <!-- <DelayTips
                        v-if="delayTimestamp > 200 || delayCount > 2"
                        :delayTimestamp="delayTimestamp"
                        :delayCount="delayCount"
                    /> -->
                    <LikeAndDislike v-model:feedbackStatus="feedbackStatus" v-model:curResponseId="curResponseId" />
                    <SkipBtn :disabled="skipDisabled" @click="skipVoice" />
                </div>
            </div>
        </div>
        <div class="video-page-btn">
            <el-upload
                :auto-upload="false"
                :show-file-list="false"
                accept="image/jpeg,image/png,image/webp"
                :on-change="handleReportUpload"
                :disabled="reportUploading"
            >
                <el-button :loading="reportUploading">手动上传检查报告</el-button>
            </el-upload>
            <el-button v-show="!isCalling" type="success" :disabled="callDisabled" @click="initRecording">
                {{ callDisabled ? t('notReadyBtn') : t('videoCallBtn') }}
            </el-button>
            <el-button v-show="isCalling" @click="stopRecording" type="danger">
                <SvgIcon name="phone-icon" className="phone-icon" />
                <span class="btn-text">{{ t('hangUpBtn') }}</span>
                <CountDown v-model="isCalling" @timeUp="stopRecording" />
            </el-button>
        </div>
        <IdeasList v-if="showIdeasList" :ideasList="videoIdeasList" />
    </div>
</template>
<script setup>
    import { analyzeUploadedReport, sendMessage, stopMessage, uploadConfig } from '@/apis';
    import { encodeWAV } from '@/hooks/useVoice';
    import { getNewUserId, setNewUserId } from '@/hooks/useRandomId';
    import { fetchEventSource } from '@microsoft/fetch-event-source';
    import { MicVAD } from '@ricky0123/vad-web';
    import { videoIdeasList, videoConfigList, showIdeasList } from '@/enums';
    import { isMobile, maxCount, getChunkLength } from '@/utils';
    import { mergeBase64ToBlob } from './merge';
    import { useI18n } from 'vue-i18n';

    const { t } = useI18n();
    import WebSocketService from '@/utils/websocket';

    let ctrl = new AbortController();
    let socket = null;
    const audioData = ref({
        base64Str: '',
        type: 'mp3'
    }); // 自定义音色base64
    const isCalling = defineModel();
    const videoRef = ref();
    const avatarVideoRef = ref();
    const digitalHumanEnabled = ref(false);
    const digitalHumanReady = ref(false);
    const digitalHumanConnecting = ref(false);
    const digitalHumanSessionId = ref(null);
    const videoStream = ref(null);
    const interval = ref();
    const canvasRef = ref();
    const videoImage = ref([]);
    const videoLoaded = ref(false);
    const taskQueue = ref([]);
    const running = ref(false);
    const outputData = ref([]);
    const isFirstReturn = ref(true);
    const audioPlayQueue = ref([]);
    const base64List = ref([]);
    const playing = ref(false);
    const timbre = ref([1]);
    const isReturnError = ref(false);

    const textQueue = ref('');
    const textAnimationInterval = ref();

    const analyser = ref();
    const dataArray = ref();
    const animationFrameId = ref();
    const skipDisabled = ref(true);
    const stop = ref(false);
    const isFrontCamera = ref(true);
    const loading = ref(false);
    const reportUploading = ref(false);

    const isEnd = ref(false); // sse接口关闭，认为模型已完成本次返回

    const isFirstPiece = ref(true);
    const allVoice = ref([]);
    const callDisabled = ref(true);

    const feedbackStatus = ref('');
    const curResponseId = ref('');
    const delayTimestamp = ref(0); // 当前发送片延时
    const delayCount = ref(0); // 当前剩余多少ms未发送到接口

    const modelVersion = ref('');

    let mediaStream;
    let audioRecorder;
    let audioStream;
    let intervalId;
    let audioContext;
    let audioChunks = [];
    let count = 0;
    let audioDOM;
    let digitalHumanPeer;
    let digitalHumanStream;

    onBeforeUnmount(() => {
        stopRecording();
    });
    const vadStartTime = ref();
    let myvad = null;
    let vadTimer = null; // vad定时器，用于检测1s内人声是否停止，1s内停止，可认为是vad误触，直接忽略，1s内未停止，则认为是人声，已自动跳过当前对话
    const vadStart = async () => {
        myvad = await MicVAD.new({
            onSpeechStart: () => {
                console.log('Speech start', +new Date());
                // if (!skipDisabled.value) {
                vadTimer && clearTimeout(vadTimer);
                vadTimer = setTimeout(() => {
                    // vadStartTime.value = +new Date();
                    console.log('打断时间: ', +new Date());
                    skipVoice();
                }, 500);
                // }
            },
            onSpeechEnd: audio => {
                vadTimer && clearTimeout(vadTimer);
                console.log('Speech end', +new Date());
                // debugger;
                // do something with `audio` (Float32Array of audio samples at sample rate 16000)...
            },
            baseAssetPath: '/'
        });
        myvad.start();
    };
    onMounted(async () => {
        const { code, message } = await stopMessage();
        if (code !== 0) {
            ElMessage({
                type: 'error',
                message: message,
                duration: 3000,
                customClass: 'system-error'
            });
            return;
        }
        callDisabled.value = false;
    });
    const delay = ms => {
        return new Promise(resolve => setTimeout(resolve, ms));
    };
    const readFileAsDataUrl = file =>
        new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    const handleReportUpload = async uploadFile => {
        const file = uploadFile.raw;
        if (!file || !['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
            ElMessage.error('请手动选择 JPG、PNG 或 WebP 格式的检查报告图片');
            return;
        }
        if (file.size > 15 * 1024 * 1024) {
            ElMessage.error('检查报告图片不能超过 15MB');
            return;
        }
        reportUploading.value = true;
        try {
            const dataUrl = await readFileAsDataUrl(file);
            const { code, message, data } = await analyzeUploadedReport({
                mime_type: file.type,
                image_data: dataUrl.split(',')[1]
            });
            if (code !== 0) throw new Error(message);
            const analysis = data.analysis || {};
            const items = (analysis.items || [])
                .slice(0, 12)
                .map(item => `${item.name || '项目'}：${item.value || '未读清'}${item.unit || ''}${item.flag && item.flag !== '正常' ? `（${item.flag}）` : ''}`)
                .join('；');
            outputData.value.push({
                type: 'BOT',
                text: `检查报告上传分析：${analysis.summary || items || '未读取到可靠项目'}\n${data.disclaimer}`,
                audio: ''
            });
            ElMessage.success('检查报告已通过独立 VLM 接口分析');
        } catch (error) {
            ElMessage.error(error.message || '检查报告分析失败');
        } finally {
            reportUploading.value = false;
        }
    };
    const initRecording = async () => {
        uploadUserConfig()
            .then(async () => {
                if (!audioDOM) {
                    audioDOM = new Audio();
                    audioDOM.playsinline = true;
                    audioDOM.preload = 'auto';
                }
                // 每次call都需要生成新uid
                setNewUserId();
                await initVideoStream('environment');
                if (digitalHumanEnabled.value) {
                    await initDigitalHuman();
                }
                buildConnect();
                await delay(100);
                // if (socket) {
                //     socket.close();
                // }
                // socket = new WebSocketService(
                //     `/ws/stream${window.location.search}&uid=${getNewUserId()}&service=minicpmo-server`
                // );
                // socket.connect();
                if (localStorage.getItem('canStopByVoice') === 'true') {
                    console.log('vad start');
                    vadStart();
                }
            })
            .catch(() => {});
    };
    // 切换摄像头
    const switchCamera = () => {
        if (!isCalling.value) {
            return;
        }
        isFrontCamera.value = !isFrontCamera.value;
        const facingMode = isFrontCamera.value ? 'environment' : 'user'; // 'user' 前置, 'environment' 后置
        initVideoStream(facingMode);
    };
    const initVideoStream = async facingMode => {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            videoStream.value = null;
        }
        outputData.value = [];
        isCalling.value = true;
        loading.value = true;
        if (!videoStream.value) {
            try {
                mediaStream = await window.navigator.mediaDevices.getUserMedia({
                    video: { facingMode },
                    audio: {
                        echoCancellation: { ideal: true },
                        noiseSuppression: { ideal: true },
                        autoGainControl: { ideal: true },
                        channelCount: { ideal: 1 },
                        sampleRate: { ideal: 16000 },
                        sampleSize: { ideal: 16 }
                    }
                });
                const audioTrack = mediaStream.getAudioTracks()[0];
                console.info('Microphone audio settings:', audioTrack?.getSettings?.());
                audioChunks = [];
                videoStream.value = mediaStream;
                videoRef.value.srcObject = mediaStream;
                loading.value = false;
                console.log('打开后： ', +new Date());
                // takePhotos();
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                console.log('samplate: ', audioContext);
                const audioSource = audioContext.createMediaStreamSource(mediaStream);
                interval.value = setInterval(() => dealImage(), 50);
                // 创建 ScriptProcessorNode 用于捕获音频数据
                const processor = audioContext.createScriptProcessor(256, 1, 1);

                processor.onaudioprocess = event => {
                    if (!isCalling.value) return;
                    if (isReturnError.value) {
                        stopRecording();
                        return;
                    }
                    const data = event.inputBuffer.getChannelData(0);
                    audioChunks.push(new Float32Array(data));
                    // 检查是否已经收集到1秒钟的数据
                    const totalBufferLength = audioChunks.reduce((total, curr) => total + curr.length, 0);
                    const chunkLength = getChunkLength(audioContext.sampleRate);
                    if (totalBufferLength >= chunkLength) {
                        // 合并到一个完整的数据数组，并裁剪成1秒钟
                        const mergedBuffer = mergeBuffers(audioChunks, totalBufferLength);
                        const oneSecondBuffer = mergedBuffer.slice(0, audioContext.sampleRate);

                        // 保存并处理成WAV格式
                        addQueue(+new Date(), () => saveAudioChunk(oneSecondBuffer, +new Date()));

                        // 保留多余的数据备用
                        audioChunks = [mergedBuffer.slice(audioContext.sampleRate)];
                    }
                };
                analyser.value = audioContext.createAnalyser();
                // 将音频节点连接到分析器
                audioSource.connect(analyser.value);
                // 分析器设置
                analyser.value.fftSize = 256;
                const bufferLength = analyser.value.frequencyBinCount;
                dataArray.value = new Uint8Array(bufferLength);
                // 开始绘制音波
                drawBars();

                audioSource.connect(processor);
                processor.connect(audioContext.destination);
            } catch {}
        }
    };
    const waitForIceGathering = peer => {
        if (peer.iceGatheringState === 'complete') return Promise.resolve();
        return new Promise(resolve => {
            const handler = () => {
                if (peer.iceGatheringState === 'complete') {
                    peer.removeEventListener('icegatheringstatechange', handler);
                    resolve();
                }
            };
            peer.addEventListener('icegatheringstatechange', handler);
            setTimeout(resolve, 5000);
        });
    };
    const waitForPeerConnected = peer => {
        if (peer.connectionState === 'connected') return Promise.resolve();
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                peer.removeEventListener('connectionstatechange', handler);
                reject(new Error('数字人 WebRTC 连接超时'));
            }, 8000);
            const handler = () => {
                if (peer.connectionState === 'connected') {
                    clearTimeout(timeout);
                    peer.removeEventListener('connectionstatechange', handler);
                    resolve();
                } else if (['failed', 'closed'].includes(peer.connectionState)) {
                    clearTimeout(timeout);
                    peer.removeEventListener('connectionstatechange', handler);
                    reject(new Error(`数字人 WebRTC 状态异常: ${peer.connectionState}`));
                }
            };
            peer.addEventListener('connectionstatechange', handler);
        });
    };
    const initDigitalHuman = async () => {
        if (!digitalHumanEnabled.value) return;
        closeDigitalHuman();
        digitalHumanConnecting.value = true;
        digitalHumanSessionId.value = Date.now() * 1000 + Math.floor(Math.random() * 1000);
        const msgId = `offer_${digitalHumanSessionId.value}`;
        try {
            digitalHumanPeer = new RTCPeerConnection();
            digitalHumanStream = new MediaStream();
            digitalHumanPeer.addTransceiver('audio', { direction: 'recvonly' });
            digitalHumanPeer.addTransceiver('video', { direction: 'recvonly' });
            digitalHumanPeer.ontrack = event => {
                digitalHumanStream.addTrack(event.track);
                if (avatarVideoRef.value) {
                    avatarVideoRef.value.srcObject = digitalHumanStream;
                    avatarVideoRef.value.play().catch(() => {});
                }
            };
            digitalHumanPeer.onconnectionstatechange = () => {
                digitalHumanReady.value = digitalHumanPeer?.connectionState === 'connected';
            };
            const offer = await digitalHumanPeer.createOffer();
            await digitalHumanPeer.setLocalDescription(offer);
            await waitForIceGathering(digitalHumanPeer);
            const response = await fetch('/api/v1/digital-human/offer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sdp: digitalHumanPeer.localDescription.sdp,
                    type: digitalHumanPeer.localDescription.type,
                    session_id: digitalHumanSessionId.value,
                    msg_id: msgId
                })
            });
            if (!response.ok) throw new Error(await response.text());
            const answer = await response.json();
            await digitalHumanPeer.setRemoteDescription(answer);
            await waitForPeerConnected(digitalHumanPeer);
        } catch (error) {
            console.error('数字人连接失败，将回退为原始音频播放', error);
            closeDigitalHuman();
        } finally {
            digitalHumanConnecting.value = false;
        }
    };
    const closeDigitalHuman = () => {
        const sessionId = digitalHumanSessionId.value;
        digitalHumanPeer?.close();
        digitalHumanPeer = null;
        digitalHumanStream = null;
        digitalHumanReady.value = false;
        if (avatarVideoRef.value) avatarVideoRef.value.srcObject = null;
        if (sessionId) {
            fetch('/api/v1/digital-human/close', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, msg_id: `close_${Date.now()}` }),
                keepalive: true
            }).catch(() => {});
        }
        digitalHumanSessionId.value = null;
    };
    const drawText = async () => {
        if (textQueue.value.length > 0) {
            outputData.value[outputData.value.length - 1].text += textQueue.value[0];
            textQueue.value = textQueue.value.slice(1);
        } else {
            cancelAnimationFrame(textAnimationInterval.value);
        }
        textAnimationInterval.value = requestAnimationFrame(drawText);
    };
    const getStopValue = () => {
        return stop.value;
    };
    const getPlayingValue = () => {
        return playing.value;
    };
    const getStopStatus = () => {
        return localStorage.getItem('canStopByVoice') === 'true';
    };
    const saveAudioChunk = (buffer, timestamp) => {
        return new Promise(resolve => {
            if (!getStopStatus() && getPlayingValue()) {
                resolve();
                return;
            }
            const wavBlob = encodeWAV(buffer, audioContext.sampleRate);
            let reader = new FileReader();
            reader.readAsDataURL(wavBlob);

            reader.onloadend = async function () {
                let base64data = reader.result.split(',')[1];
                const imgBase64 = videoImage.value[videoImage.value.length - 1]?.src;
                if (!(base64data && imgBase64)) {
                    resolve();
                    return;
                }
                const strBase64 = imgBase64.split(',')[1];
                count++;
                let obj = {
                    messages: [
                        {
                            role: 'user',
                            content: [
                                {
                                    type: 'input_audio',
                                    input_audio: {
                                        data: base64data,
                                        format: 'wav',
                                        timestamp: String(timestamp)
                                    }
                                }
                            ]
                        }
                    ]
                };
                obj.messages[0].content.unshift({
                    type: 'image_data',
                    image_data: {
                        data: count === maxCount ? strBase64 : '',
                        type: 2,
                        source: 'realtime_video'
                    }
                });
                if (count === maxCount) {
                    count = 0;
                }
                // socket.send(JSON.stringify(obj));
                // socket.on('message', data => {
                //     console.log('message: ', data);
                //     delayTimestamp.value = +new Date() - timestamp;
                //     delayCount.value = taskQueue.value.length;
                //     resolve();
                // });
                // 将Base64音频数据发送到后端
                try {
                    await sendMessage(obj);
                    delayTimestamp.value = +new Date() - timestamp;
                    delayCount.value = taskQueue.value.length;
                } catch (err) {}
                resolve();
            };
        });
    };
    const mergeBuffers = (buffers, length) => {
        const result = new Float32Array(length);
        let offset = 0;
        for (let buffer of buffers) {
            result.set(buffer, offset);
            offset += buffer.length;
        }
        return result;
    };
    const stopRecording = () => {
        isCalling.value = false;
        clearInterval(interval.value);
        interval.value = null;
        if (audioRecorder && audioRecorder.state !== 'inactive') {
            audioRecorder.stop();
        }
        if (animationFrameId.value) {
            cancelAnimationFrame(animationFrameId.value);
        }
        if (audioContext && audioContext.state !== 'closed') {
            audioContext.close();
        }
        destroyVideoStream();
        taskQueue.value = [];
        audioPlayQueue.value = [];
        base64List.value = [];
        ctrl.abort();
        ctrl = new AbortController();
        isReturnError.value = false;
        skipDisabled.value = true;
        playing.value = false;
        audioDOM?.pause();
        closeDigitalHuman();
        stopMessage();
        if (socket) {
            socket.close();
        }
        if (
            outputData.value[outputData.value.length - 1]?.type === 'BOT' &&
            outputData.value[outputData.value.length - 1].audio === '' &&
            allVoice.value.length > 0
        ) {
            outputData.value[outputData.value.length - 1].audio = mergeBase64ToBlob(allVoice.value);
        }
        myvad && myvad.destroy();
    };
    // 建立连接
    const buildConnect = () => {
        const obj = {
            messages: [
                {
                    role: 'user',
                    content: [{ type: 'none' }]
                }
            ],
            stream: true
        };
        if (digitalHumanPeer && digitalHumanSessionId.value) {
            obj.digital_human_session_id = digitalHumanSessionId.value;
            obj.digital_human_msg_id = `reply_${Date.now()}`;
        }
        isEnd.value = false;
        ctrl.abort();
        ctrl = new AbortController();
        const url = `/api/v1/completions${window.location.search}`;

        fetchEventSource(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                service: 'minicpmo-server',
                uid: getNewUserId()
            },
            body: JSON.stringify(obj),
            signal: ctrl.signal,
            openWhenHidden: true,
            async onopen(response) {
                isFirstPiece.value = true;
                isFirstReturn.value = true;
                allVoice.value = [];
                base64List.value = [];
                console.log('onopen', response);
                if (response.status !== 200) {
                    ElMessage({
                        type: 'error',
                        message: 'At limit. Please try again soon.',
                        duration: 3000,
                        customClass: 'system-error'
                    });
                    isReturnError.value = true;
                } else {
                    isReturnError.value = false;
                    drawText();
                }
            },
            onmessage(msg) {
                const data = JSON.parse(msg.data);
                if (data.response_id) {
                    curResponseId.value = data.response_id;
                }
                if (data.choices[0]?.digital_human_audio === false) {
                    console.warn('数字人音频注入失败，本段回退为浏览器原始音频播放');
                    digitalHumanReady.value = false;
                }
                if (data.choices[0]?.text) {
                    textQueue.value += data.choices[0].text.replace('<end>', '');
                    console.warn('text return time -------------------------------', +new Date());
                }
                // 首次返回的是前端发给后端的音频片段，需要单独处理
                if (isFirstReturn.value) {
                    console.log('第一次');
                    isFirstReturn.value = false;
                    // 如果后端返回的音频为空，需要重连
                    if (!data.choices[0].audio) {
                        buildConnect();
                        return;
                    }
                    outputData.value.push({
                        type: 'USER',
                        audio: `data:audio/wav;base64,${data.choices[0].audio}`
                    });
                    outputData.value.push({
                        type: 'BOT',
                        text: '',
                        audio: ''
                    });
                    return;
                }
                if (data.choices[0]?.audio) {
                    console.log('audio return time -------------------------------', +new Date());
                    if (!getStopValue() && isCalling.value) {
                        skipDisabled.value = false;
                        base64List.value.push(`data:audio/wav;base64,${data.choices[0].audio}`);
                        addAudioQueue(() => truePlay(data.choices[0].audio));
                    }
                    allVoice.value.push(`data:audio/wav;base64,${data.choices[0].audio}`);
                } else {
                    // 发生异常了，直接重连
                    buildConnect();
                }
                if (data.choices[0].text.includes('<end>')) {
                    // isEnd.value = true;
                    console.log('收到结束标记了:', +new Date());
                    if (
                        outputData.value[outputData.value.length - 1]?.type === 'BOT' &&
                        outputData.value[outputData.value.length - 1].audio === '' &&
                        allVoice.value.length > 0
                    ) {
                        outputData.value[outputData.value.length - 1].audio = mergeBase64ToBlob(allVoice.value);
                    }
                }
            },
            onclose() {
                console.log('onclose', +new Date());
                isEnd.value = true;
                if (
                    outputData.value[outputData.value.length - 1]?.type === 'BOT' &&
                    outputData.value[outputData.value.length - 1].audio === '' &&
                    allVoice.value.length > 0
                ) {
                    outputData.value[outputData.value.length - 1].audio = mergeBase64ToBlob(allVoice.value);
                }
                // sse关闭后，如果待播放的音频列表为空，说明模型出错了，此次连接没有返回音频，则直接重连
                vadStartTime.value = +new Date();
                if (audioPlayQueue.value.length === 0) {
                    let startIndex = taskQueue.value.findIndex(item => item.time >= vadStartTime.value - 1000);
                    if (startIndex !== -1) {
                        taskQueue.value = taskQueue.value.slice(startIndex);
                    }
                    buildConnect();
                }
            },
            onerror(err) {
                console.log('onerror', err);
                ctrl.abort();
                ctrl = new AbortController();
                throw err;
            }
        });
    };
    // 返回的语音放到队列里，挨个播放
    const addAudioQueue = async item => {
        audioPlayQueue.value.push(item);
        if (isFirstPiece.value) {
            await delay(1500);
            isFirstPiece.value = false;
        }
        if (audioPlayQueue.value.length > 0 && !playing.value) {
            playing.value = true;
            playAudio();
        }
    };
    // 控制播放队列执行
    const playAudio = () => {
        console.log('剩余播放列表:', audioPlayQueue.value, +new Date());

        if (!isEnd.value && base64List.value.length >= 2) {
            const remainLen = base64List.value.length;
            const blob = mergeBase64ToBlob(base64List.value);
            audioDOM.src = blob;
            audioDOM.muted = digitalHumanReady.value;
            audioDOM.play();
            console.error('前期合并后播放开始时间: ', +new Date());
            audioDOM.onended = () => {
                console.error('前期合并后播放结束时间: ', +new Date());
                base64List.value = base64List.value.slice(remainLen);
                audioPlayQueue.value = audioPlayQueue.value.slice(remainLen);
                playAudio();
            };
            return;
        }
        if (isEnd.value && base64List.value.length >= 2) {
            const blob = mergeBase64ToBlob(base64List.value);
            audioDOM.src = blob;
            audioDOM.muted = digitalHumanReady.value;
            audioDOM.play();
            console.error('合并后播放开始时间: ', +new Date());
            audioDOM.onended = () => {
                console.error('合并后播放结束时间: ', +new Date());
                // URL.revokeObjectURL(url);
                base64List.value = [];
                audioPlayQueue.value = [];
                playing.value = false;
                skipDisabled.value = true;
                if (isCalling.value && !isReturnError.value) {
                    // skipDisabled.value = true;
                    taskQueue.value = [];
                    // 打断前记录一下打断时间或vad触发事件
                    // vadStartTime.value = +new Date();
                    // // 每次完成后只保留当前时刻往前推1s的语音
                    // console.log(
                    //     '截取前长度:',
                    //     taskQueue.value.map(item => item.time)
                    // );
                    // let startIndex = taskQueue.value.findIndex(item => item.time >= vadStartTime.value - 1000);
                    // if (startIndex !== -1) {
                    //     taskQueue.value = taskQueue.value.slice(startIndex);
                    //     console.log(
                    //         '截取后长度:',
                    //         taskQueue.value.map(item => item.time),
                    //         vadStartTime.value
                    //     );
                    // }
                    buildConnect();
                }
            };
            return;
        }
        base64List.value.shift();
        const _truePlay = audioPlayQueue.value.shift();
        if (_truePlay) {
            _truePlay().finally(() => {
                playAudio();
            });
        } else {
            playing.value = false;
            if (isEnd.value) {
                console.warn('play done................');
                skipDisabled.value = true;
            }
            // 播放完成后且正在通话且接口未返回错误时开始下一次连接
            if (isEnd.value && isCalling.value && !isReturnError.value) {
                // skipDisabled.value = true;
                taskQueue.value = [];
                // 跳过之后，只保留当前时间点两秒内到之后的音频片段
                // vadStartTime.value = +new Date();
                // console.log(
                //     '截取前长度:',
                //     taskQueue.value.map(item => item.time)
                // );
                // let startIndex = taskQueue.value.findIndex(item => item.time >= vadStartTime.value - 1000);
                // if (startIndex !== -1) {
                //     taskQueue.value = taskQueue.value.slice(startIndex);
                //     console.log(
                //         '截取后长度:',
                //         taskQueue.value.map(item => item.time),
                //         vadStartTime.value
                //     );
                // }
                buildConnect();
            }
        }
    };
    // 播放音频
    const truePlay = voice => {
        console.log('promise: ', +new Date());
        return new Promise(resolve => {
            audioDOM.src = 'data:audio/wav;base64,' + voice;
            audioDOM.muted = digitalHumanReady.value;
            console.error('播放开始时间:', +new Date());
            audioDOM
                .play()
                .then(() => {
                    console.log('Audio played successfully');
                })
                .catch(error => {
                    if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
                        console.error('User interaction required or permission issue:', error);
                        // ElMessage.warning('音频播放失败');
                        console.error('播放失败时间');
                        // alert('Please interact with the page (like clicking a button) to enable audio playback.');
                    } else {
                        console.error('Error playing audio:', error);
                    }
                });
            // .finally(() => {
            //     resolve();
            // });
            audioDOM.onerror = () => {
                console.error('播放失败时间', +new Date());
                resolve();
            };
            audioDOM.onended = () => {
                console.error('播放结束时间: ', +new Date());
                // URL.revokeObjectURL(url);
                resolve();
            };
        });
    };
    // 当队列中任务数大于0时，开始处理队列中的任务
    const addQueue = (time, item) => {
        taskQueue.value.push({ func: item, time });
        if (taskQueue.value.length > 0 && !running.value) {
            running.value = true;
            processQueue();
        }
    };
    const processQueue = () => {
        const item = taskQueue.value.shift();
        if (item?.func) {
            item.func()
                .then(res => {
                    console.log('已处理事件: ', res);
                })
                .finally(() => processQueue());
        } else {
            running.value = false;
        }
    };
    const destroyVideoStream = () => {
        videoStream.value?.getTracks().forEach(track => track.stop());
        videoStream.value = null;
        // 将srcObject设置为null以切断与MediaStream 对象的链接，以便将其释放
        videoRef.value.srcObject = null;

        videoImage.value = [];
        videoLoaded.value = false;

        clearInterval(intervalId);
        clearInterval(interval.value);
        interval.value = null;
    };
    const dealImage = () => {
        if (!videoRef.value) {
            return;
        }
        const canvas = canvasRef.value;
        canvasRef.value.width = videoRef.value.videoWidth;
        canvasRef.value.height = videoRef.value.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(videoRef.value, 0, 0, canvasRef.value.width, canvasRef.value.height);
        const imageDataUrl = canvas.toDataURL('image/webp', 0.8);

        videoImage.value.push({ src: imageDataUrl });
    };
    const drawBars = () => {
        // AnalyserNode接口的 getByteFrequencyData() 方法将当前频率数据复制到传入的 Uint8Array（无符号字节数组）中。
        analyser.value.getByteFrequencyData(dataArray.value);
        animationFrameId.value = requestAnimationFrame(drawBars);
    };
    // 跳过当前片段
    const skipVoice = async () => {
        // 打断前记录一下打断时间或vad触发事件
        vadStartTime.value = +new Date();
        if (!skipDisabled.value) {
            if (
                outputData.value[outputData.value.length - 1]?.type === 'BOT' &&
                outputData.value[outputData.value.length - 1].audio === ''
            ) {
                outputData.value[outputData.value.length - 1].audio = mergeBase64ToBlob(allVoice.value);
            }
            base64List.value = [];
            audioPlayQueue.value = [];
            // 跳过之后，只保留当前时间点两秒内到之后的音频片段
            console.log(
                '截取前长度:',
                taskQueue.value.map(item => item.time)
            );
            let startIndex = taskQueue.value.findIndex(item => item.time >= vadStartTime.value - 1000);
            if (startIndex !== -1) {
                taskQueue.value = taskQueue.value.slice(startIndex);
                console.log(
                    '截取后长度:',
                    taskQueue.value.map(item => item.time),
                    vadStartTime.value
                );
            }
            stop.value = true;
            audioDOM?.pause();
            setTimeout(() => {
                skipDisabled.value = true;
            }, 300);
            try {
                playing.value = false;
                await stopMessage();
                stop.value = false;
                // playing.value = false;
                buildConnect();
                // cancelAnimationFrame(animationFrameId.value);
            } catch (err) {}
        }
    };
    // 每次call先上传当前用户配置
    const uploadUserConfig = async () => {
        if (!localStorage.getItem('configData')) {
            return new Promise(resolve => resolve());
        }
        const {
            videoQuality,
            useAudioPrompt,
            voiceClonePrompt,
            assistantPrompt,
            patientGender,
            patientAge,
            visitType,
            vadThreshold,
            audioFormat,
            base64Str
        } = JSON.parse(localStorage.getItem('configData'));
        const obj = {
            messages: [
                {
                    role: 'user',
                    content: [
                        {
                            type: 'input_audio',
                            input_audio: {
                                data: base64Str,
                                format: audioFormat
                            }
                        },
                        {
                            type: 'options',
                            options: {
                                hd_video: videoQuality,
                                use_audio_prompt: useAudioPrompt,
                                vad_threshold: vadThreshold,
                                voice_clone_prompt: voiceClonePrompt,
                                assistant_prompt: assistantPrompt,
                                patient_gender: patientGender,
                                patient_age: patientAge,
                                visit_type: visitType
                            }
                        }
                    ]
                }
            ]
        };
        const { code, message, data } = await uploadConfig(obj);
        modelVersion.value = data?.choices?.content || '';
        return new Promise((resolve, reject) => {
            if (code !== 0) {
                ElMessage({
                    type: 'error',
                    message: message,
                    duration: 3000,
                    customClass: 'system-error'
                });
                reject();
            } else {
                resolve();
            }
        });
    };
</script>
<style lang="less" scoped>
    .video-page {
        flex: 1;
        height: 100%;
        display: flex;
        flex-direction: column;
        &-header {
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0 16px 16px;
            box-shadow: 0 0.5px 0 0 #e0e0e0;
            margin-bottom: 16px;
            position: relative;
            .digital-human-switch {
                position: absolute;
                right: 16px;
                display: flex;
                align-items: center;
                gap: 8px;
                color: rgba(23, 23, 23, 0.72);
                font-size: 14px;
            }
            .header-icon {
                display: flex;
                align-items: center;
                img {
                    width: 24px;
                    height: 24px;
                    margin-right: 8px;
                }
                span {
                    color: rgba(23, 23, 23, 0.9);
                    font-family: PingFang SC;
                    font-size: 16px;
                    font-style: normal;
                    font-weight: 500;
                    line-height: normal;
                    margin-right: 40px;
                    flex-shrink: 0;
                }
            }
            .voice-container {
                display: flex;
                .voice-icon {
                    width: 191px;
                    height: 45px;
                }
            }
        }
        &-content {
            flex: 1;
            margin-bottom: 16px;
            display: flex;
            height: 0;
            &-video {
                width: 50%;
                height: 100%;
                background: #f3f3f3;
                flex-shrink: 0;
                position: relative;
                border-radius: 12px;
                overflow: hidden;
                video {
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                }
                .switch-camera {
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    width: 36px;
                    height: 36px;
                    background: #ffffff;
                    border-radius: 6px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    font-size: 24px;
                    z-index: 999;
                    .icon {
                        width: 20px;
                        height: 20px;
                    }
                }
            }
            &-right {
                margin-left: 16px;
                flex: 1;
                display: flex;
                flex-direction: column;
                min-width: 0;
                position: relative;
                border-radius: 12px;
                overflow: hidden;
                background: #f3f3f3;
                .avatar-video-wrap {
                    position: relative;
                    flex: 1;
                    min-height: 0;
                    video {
                        width: 100%;
                        height: 100%;
                        object-fit: contain;
                        background: #17191f;
                    }
                    .avatar-state {
                        position: absolute;
                        inset: 0;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: #8a8f9c;
                        background: #eef0f4;
                    }
                }
                .output-content {
                    height: 132px;
                    flex-shrink: 0;
                    overflow: auto;
                    padding: 12px 16px;
                    background: #fff;
                    border-top: 1px solid #e6e8ee;
                }
                .skip-box {
                    display: flex;
                    align-items: center;
                    justify-content: flex-end;
                    padding: 8px 16px;
                    background: #fff;
                }
                &.human-disabled {
                    .output-content {
                        height: auto;
                        flex: 1;
                        padding-top: 52px;
                        border-top: 0;
                    }
                }
            }
            .panel-title {
                position: absolute;
                top: 12px;
                left: 12px;
                z-index: 3;
                padding: 5px 10px;
                border-radius: 999px;
                color: #fff;
                background: rgba(20, 24, 32, 0.66);
                font-size: 13px;
                backdrop-filter: blur(6px);
            }
        }
        &-btn {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 12px;
            padding: 8px 0;
            .el-button {
                width: 284px;
                height: 46px;
                border-radius: 8px;
            }
            .el-button.el-button--success {
                background: #647fff;
                border-color: #647fff;
                &:hover {
                    opacity: 0.8;
                }
                span {
                    color: #fff;
                    font-family: PingFang SC;
                    font-size: 16px;
                    font-style: normal;
                    font-weight: 500;
                    line-height: normal;
                }
            }
            .el-button.el-button--success.is-disabled {
                background: #f3f3f3;
                border-color: #f3f3f3;
                span {
                    color: #d1d1d1;
                }
            }
            .el-button.el-button--danger {
                border-color: #dc3545;
                background-color: #dc3545;
                color: #ffffff;
                font-family: PingFang SC;
                font-size: 16px;
                font-style: normal;
                font-weight: 500;
                line-height: normal;
                .phone-icon {
                    margin-right: 10px;
                }
                .btn-text {
                    margin-right: 10px;
                }
                .btn-desc {
                    margin-right: 16px;
                }
            }
        }
    }
    .video-size {
        position: absolute;
        bottom: 10px;
        right: 10px;
        background: rgba(0, 0, 0, 0.5);
        color: #fff;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
    }
</style>
