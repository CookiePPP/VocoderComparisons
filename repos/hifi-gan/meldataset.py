import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from nvSTFT import load_wav_to_torch
from nvSTFT import STFT as STFT_Class
from glob import glob

def get_dataset_filelist(a):
    if a.input_wavs_dir is None:
        with open(a.input_training_file, 'r', encoding='utf-8') as fi:
            training_files = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

        with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
            validation_files = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]
    else:
        print("Searching for WAV files in '--input_wav_dir' arg...")
        wav_files = sorted(glob(os.path.join(a.input_wavs_dir, '**', '*.wav'), recursive=True))
        print(f"Found {len(wav_files)} WAV Files.")
        random.Random(1).shuffle(wav_files)
        
        training_files   = wav_files[:int(len(wav_files*0.95)) ]
        validation_files = wav_files[ int(len(wav_files*0.95)):]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.STFT = STFT_Class(sampling_rate, num_mels, n_fft, win_size, hop_size, fmin, fmax)
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav_to_torch(filename, target_sr=self.sampling_rate)
            if not self.fine_tuning:
                audio = torch.from_numpy(normalize(audio.numpy()) * 0.95)
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1
        
        #audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        
        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
            
            mel = self.STFT.get_mel(audio)
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)
            
            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
        
        mel_loss = mel
        
        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
