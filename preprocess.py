import torch
import pandas as pd
import numpy as np
import librosa
from torch.utils.data import Dataset
from Hyperparameters import Hyperparameters as hp

class SpeechDataset(Dataset):
    def __init__(self, wav_paths = hp.meta_path, transcriptions = hp.meta_path, transform=None):
        self.wav_paths = pd.read_csv(wav_paths)["path"].tolist()
        self.transcriptions = pd.read_csv(transcriptions)["text"].tolist()
        self.n_mels = hp.n_mels
        self.max_len = None
        self.sample_rate = 44100
        self.transform = transform

    def __len__(self):
        return len(self.wav_paths)

    # def __getitem__(self, idx):
    #     wav_path = self.wav_paths[idx]
    #     transcription = self.transcriptions[idx]
        
    #     # 音声データの読み込みと変換（必要に応じて）
    #     # ここでは例としてランダムなノイズを生成
    #     mel_spectrogram = torch.randn(1, 100, 80)  # 例: メルスペクトログラムのダミーデータ
        
    #     if self.transform:
    #         mel_spectrogram = self.transform(mel_spectrogram)
        
    #     return mel_spectrogram, transcription

    
    def __getitem__(self, idx):
        # 音声ファイルの読み込み
        wav_path = self.wav_paths[idx]
        transcription = self.transcriptions[idx]
        
        # 音声波形を読み込む
        waveform, sr = librosa.load(wav_path, sr=self.sample_rate)

        # メルスペクトログラムを生成
        mel_spectrogram = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=self.n_mels
        )
        
        # 対数メルスペクトログラムに変換
        log_mel_spectrogram = np.log1p(mel_spectrogram)

        # パディング（必要に応じて）
        if self.max_len:
            log_mel_spectrogram = self._pad_spectrogram(log_mel_spectrogram, self.max_len)

        # numpy を PyTorch のテンソルに変換
        log_mel_spectrogram = torch.tensor(log_mel_spectrogram, dtype=torch.float32)
        transcription = torch.tensor(self._text_to_sequence(transcription), dtype=torch.long)

        return log_mel_spectrogram, transcription

    def _pad_spectrogram(self, spectrogram, max_len):
        """
        メルスペクトログラムを最大フレーム数でパディング
        """
        if spectrogram.shape[1] < max_len:
            pad_width = max_len - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        else:
            spectrogram = spectrogram[:, :max_len]
        return spectrogram

    def _text_to_sequence(self, text):
        """
        テキストをシーケンス（整数のリスト）に変換
        """
        # 簡易的な例（アルファベットを数値に変換）
        # 実際のモデルでは text の符号化方式が別途指定されるべき
        char_map = {ch: idx for idx, ch in enumerate("abcdefghijklmnopqrstuvwxyz ", start=1)}
        sequence = [char_map.get(ch, 0) for ch in text.lower()]
        return sequence
    
dataset = SpeechDataset()