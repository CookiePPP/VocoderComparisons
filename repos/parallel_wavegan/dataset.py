# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import random
import torch

from glob import glob
from torch.utils.data.distributed import DistributedSampler
from wavegrad.nvSTFT import STFT as STFT_Class
from wavegrad.nvSTFT import load_wav_to_torch
STFT = STFT_Class(hop_length=256)

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line_strip.split(split) for line_strip in (line.strip() for line in f) if line_strip and line_strip[0] is not ";"]
    return filepaths_and_text

class NumpyDataset(torch.utils.data.Dataset):
  def __init__(self, fl_path):
    super().__init__()
    self.filenames = [x[0] for x in load_filepaths_and_text(fl_path)]

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    spec_filename = f'{audio_filename}.spec.npy'
    signal, sr = load_wav_to_torch(audio_filename)
    
    spectrogram = None
    if os.path.exists(spec_filename):
        try:
            spectrogram = torch.load(spec_filename)
        except:
            pass
    if spectrogram is None:
        spectrogram = STFT(audio_filename)
        try:
            torch.save(spectrogram, spec_filename)
        except:
            pass
    
    return {
        'audio': signal, # normalized audio [T], range [-1.0, 1.0]
        'spectrogram': spectrogram.T # [n_mel, T] -> [T, n_mel]
    }


class Collator:
  def __init__(self, params):
    self.params = params
    self.use_noise_input = params.get("generator_type", "ParallelWaveGANGenerator") != "MelGANGenerator"

  def collate(self, minibatch):
    samples_per_frame = self.params["hop_size"]
    n_spect_frames = self.params["batch_max_steps"]//samples_per_frame
    for record in minibatch:
      # Filter out records that aren't long enough.
      if len(record['spectrogram']) < n_spect_frames:
        del record['spectrogram']
        del record['audio']
        continue
      
      start = random.randint(0, record['spectrogram'].shape[0] - n_spect_frames)
      end = start + n_spect_frames
      record['spectrogram'] = record['spectrogram'][start:end].T
      
      start *= samples_per_frame
      end *= samples_per_frame
      record['audio'] = record['audio'][start:end]
      record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant')

    audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
    spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
    
    spectrogram = torch.from_numpy(spectrogram)#  [B, n_mel, T]
    audio = torch.from_numpy(audio).unsqueeze(1)# [B,     1, T]
    
    outputs = {
        "spect": spectrogram,
        "audio": audio,
    }
    if self.use_noise_input:
        outputs["noise"] = audio.clone().normal_()
    return outputs


def from_path(fl_path, params, is_distributed=False):
  dataset = NumpyDataset(fl_path)
  sampler = DistributedSampler(dataset) if is_distributed else None
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params["batch_size"],
      collate_fn=Collator(params).collate,
      shuffle=not is_distributed,
      num_workers=os.cpu_count(),
      sampler=sampler,
      pin_memory=True,
      drop_last=True), sampler
