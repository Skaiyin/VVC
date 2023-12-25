import pysrt
from espnet2.bin.tts_inference import Text2Speech
import numpy as np
import soundfile as sf
subs = pysrt.open('tr1.vtt')
# 提取时间戳和文本
sub_data = [
    (sub.start.ordinal, sub.end.ordinal, sub.text) for sub in subs
]
audio_segments = []
model = Text2Speech.from_pretrained("mio/amadeus")
print("start")
for start, end, text in sub_data:
    print(text)
    wav = model(text)["wav"]
    # sf.write('test.wav', wav, model.fs)
    audio_segments.append((start, end, wav))

def get_silence(duration, rate):
    """生成指定时长的静音片段"""
    return np.zeros(int(rate * duration/1000))
# 初始化最终合并后的音频
final_audio = np.array([], dtype=np.float32)
last_end = 0

for start, end, segment in audio_segments:
    # 计算每个片段前的静音长度（如果有）
    current_length = len(final_audio) / model.fs * 1000
    silence_duration = start - current_length
    # print(silence_duration)
    if silence_duration > 0:
        final_audio = np.concatenate((final_audio, get_silence(silence_duration, model.fs)),axis=0)

    # 添加当前音频片段
    final_audio = np.concatenate((final_audio, segment),axis=0)
    # print(model.fs)
    last_end = end
# 保存最终音频文件
sf.write('final_audio.wav', final_audio, model.fs)