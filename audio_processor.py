import os
import numpy as np
import librosa
import soundfile as sf


class AudioProcessor:
    # Клас для обробки та аналізу аудіо
    def __init__(self, file_path="", sr=22050, n_fft=2048):
        self._file_path = file_path    # Шлях до файлу
        self._sr = sr                  # Частота
        self._n_fft = n_fft            # Розмір FFT
        self.__signal = None           # Сигнал
        self.__results = {}            # Результати
        self.__spec_db = None          # Спектрограма
        self.__chroma = None           # Хроматограма

    # Отримання приватних даних
    def get_signal(self):
        return self.__signal

    def get_sr(self):
        return self._sr

    def get_results(self):
        return self.__results

    def get_spec_db(self):
        return self.__spec_db

    def get_chroma(self):
        return self.__chroma

    def create_demo_file(self, filename="demo.wav"):
        # Створення тестового аудіо (синусоїда)
        duration = 4.0
        t = np.linspace(0, duration, int(self._sr * duration), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        beat = 0.5 + 0.5 * np.maximum(0, np.sin(2 * np.pi * 2.0 * t))
        signal = librosa.util.normalize(tone * beat)

        sf.write(filename, signal, self._sr)
        self._file_path = filename
        return filename

    def load_and_clean(self):
        # Перевірка файлу та формату
        if not os.path.exists(self._file_path):
            raise FileNotFoundError("Файл не знайдено!")

        ext = os.path.splitext(self._file_path)[1].lower()
        if ext not in (".wav", ".mp3", ".flac", ".ogg"):
            raise ValueError("Невірний формат! Потрібен .wav, .mp3, .flac або .ogg")

        # Завантаження та очищення від тиші
        y, sr = librosa.load(self._file_path, sr=self._sr, mono=True)
        if len(y) == 0:
            raise ValueError("Аудіофайл порожній!")

        y_clean, _ = librosa.effects.trim(y, top_db=30)
        self.__signal = librosa.util.normalize(y_clean)
        self._sr = sr

    def analyze(self):
        # 1) Темп (BPM)
        tempo, _ = librosa.beat.beat_track(y=self.__signal, sr=self._sr)
        bpm = round(float(np.atleast_1d(tempo)[0]), 1)

        # 2) Спектральні характеристики
        stft = np.abs(librosa.stft(self.__signal, n_fft=self._n_fft))
        self.__spec_db = librosa.amplitude_to_db(stft, ref=np.max)

        centroid = librosa.feature.spectral_centroid(S=stft, sr=self._sr)
        bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=self._sr)
        rolloff = librosa.feature.spectral_rolloff(S=stft, sr=self._sr)

        # 3) Гучність та ZCR
        rms = librosa.feature.rms(y=self.__signal)
        zcr = librosa.feature.zero_crossing_rate(y=self.__signal)

        # 4) Тональність (нота)
        self.__chroma = librosa.feature.chroma_stft(S=stft**2, sr=self._sr)
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        mean_chroma = np.mean(self.__chroma, axis=1)
        main_note = notes[int(np.argmax(mean_chroma))]

        # Зберігання результатів
        duration = round(float(librosa.get_duration(y=self.__signal, sr=self._sr)), 2)
        self.__results = {
            "Файл": os.path.basename(self._file_path),
            "Тривалість (с)": duration,
            "Частота дискретизації (Гц)": self._sr,
            "Темп (BPM)": bpm,
            "Основна тональність (нота)": main_note,
            "Спектральний центроїд (Гц)": round(float(np.mean(centroid)), 2),
            "Спектральна ширина (Гц)": round(float(np.mean(bandwidth)), 2),
            "Спектральний спад (Гц)": round(float(np.mean(rolloff)), 2),
            "Середня гучність (RMS)": round(float(np.mean(rms)), 4),
            "Zero Crossing Rate": round(float(np.mean(zcr)), 4)
        }
        return self.__results

    def __str__(self):
        return f"Аудіофайл '{os.path.basename(self._file_path)}' ({self._sr} Гц)"
