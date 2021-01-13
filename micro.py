import pyaudio
import numpy as np
import wave

def Monitor_MIC(filename):
    print('START')
    CHUNK = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100 	#录音时的采样率
    WAVE_OUTPUT_FILENAME = filename + ".wav"
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []

    #print("ready for recording" + str(time.localtime(time.time()).tm_sec))
    for i in range(0, 5):
        data = stream.read(CHUNK)
        frames.append(data)
    audio_data = np.fromstring(data, dtype=np.short)
    temp = np.max(audio_data)
    print("detected a signal")
    less = []
    frames2 = []
    for ti in range(60):
        for i in range(0, 31):
            data2 = stream.read(CHUNK)
            frames2.append(data2)
        audio_data2 = np.fromstring(data2, dtype=np.short)
        temp2 = np.max(audio_data2)
        print("recording,current strength：",temp2)

    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames2))
    wf.close()

if __name__ == '__main__':
    Monitor_MIC('lala')