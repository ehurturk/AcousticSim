import time
import os
import random
import glob
import librosa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as P
from scipy.signal import decimate
from scipy import signal
import soundfile as sf
from typing import Union, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Config():
    SAMPLE_RATE_KHZ = 44.1
    SAMPLE_RATE_HZ = SAMPLE_RATE_KHZ * 1000

    DATASET_NAME = ""
    TEMPO_USED = "fast"

    data_path_base = "/content/drive/MyDrive/acousticsim/"
    segments_path = f"{data_path_base}/segments/"
    models_base = f"{data_path_base}/models/"

    acoustic_mic_base = f"{data_path_base}/{DATASET_NAME}/acoustic_mic/{TEMPO_USED}/"
    acoustic_pickup_base = f"{data_path_base}/{DATASET_NAME}/acoustic_mic/{TEMPO_USED}/"
    career_sg_base = f"{data_path_base}/{DATASET_NAME}/career_sg/{TEMPO_USED}/"
    ibanez_base = f"{data_path_base}/{DATASET_NAME}/ibanez/{TEMPO_USED}/"

cfg = Config()


# load data
def construct_dataset_paths_glob(cfg):
    dataset_paths = []
    
    acoustic_pattern = os.path.join(cfg.acoustic_mic_base, "*", "*.wav")
    acoustic_files = glob.glob(acoustic_pattern)
    
    for acoustic_path in acoustic_files:
        acoustic_dir = os.path.dirname(acoustic_path)
        genre = os.path.basename(acoustic_dir)
        acoustic_filename = os.path.basename(acoustic_path)
        
        base_name = acoustic_filename.replace("acoustic_mic_", "")
        
        sg_pattern = os.path.join(cfg.career_sg_base, genre, f"*{base_name}")
        sg_matches = glob.glob(sg_pattern)
        
        if sg_matches:
            dataset_paths.append((sg_matches[0], acoustic_path))
        
        ibanez_pattern = os.path.join(cfg.ibanez_base, genre, f"*{base_name}")
        ibanez_matches = glob.glob(ibanez_pattern)
        
        if ibanez_matches:
            dataset_paths.append((ibanez_matches[0], acoustic_path))
    
    return dataset_paths

dataset_paths = construct_dataset_paths_glob(cfg)
print(dataset_paths)
# prints [('/content/drive/MyDrive/acousticsim///career_sg/fast/jazz/jazz_2_200BPM.wav', '/content/drive/MyDrive/acousticsim///acoustic_mic/fast/jazz/jazz_2_200BPM.wav'),...]

def pre_emphasis(x, coeff=0.95):
  return torch.concat([x, x-coeff*x], 1)

def error_to_signal(target, pred):
  target, pred = pre_emphasis(target), pre_emphasis(pred)
  return torch.sum(torch.pow(target - pred, 2), axis=0) / (torch.sum(torch.pow(target, 2), axis=0)+1e-10)

def peak_norm(signal):
  signal_max = np.max(signal)
  signal_min = np.min(signal)
  signal_norm = np.max(signal_max, np.abs(signal_min))
  return signal / signal_norm

input = []
target = []

for path in dataset_paths:
  i = np.array(librosa.load(path[0], sr=cfg.SAMPLE_RATE_HZ)[0]).astype(np.float32).flatten()  # Clean signal (input)
  t = np.array(librosa.load(path[1], sr=cfg.SAMPLE_RATE_HZ)[0]).astype(np.float32).flatten()  # Acoustic signal (target)

  i = peak_norm(i).reshape(len(i), 1)
  t = peak_norm(t).reshape(len(t), 1)

  input.append(i)
  target.append(t)

print(f"Loaded {len(input)} audio pairs")

class TrainConfig:
    SEGMENT_LENGTH_SEC = 1.0  # 1 second segments
    OVERLAP_RATIO = 0.5  # 50% overlap
    HIDDEN_SIZE = 32  # LSTM hidden size - keep small for real-time
    NUM_LAYERS = 1  # Single layer for real-time performance
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    MODEL_TYPE = 'LSTM'  # 'LSTM' or 'GRU'
    
train_cfg = TrainConfig()

def segment_audio(audio_list, segment_length_samples, overlap_samples):
    segments = []
    for audio in audio_list:
        audio_len = len(audio)
        step_size = segment_length_samples - overlap_samples
        
        for start in range(0, audio_len - segment_length_samples + 1, step_size):
            end = start + segment_length_samples
            segment = audio[start:end]
            segments.append(segment)
    
    return segments

segment_length_samples = int(train_cfg.SEGMENT_LENGTH_SEC * cfg.SAMPLE_RATE_HZ)
overlap_samples = int(segment_length_samples * train_cfg.OVERLAP_RATIO)

print(f"Segmenting audio: {segment_length_samples} samples per segment, {overlap_samples} overlap")

input_segments = segment_audio(input, segment_length_samples, overlap_samples)
target_segments = segment_audio(target, segment_length_samples, overlap_samples)

print(f"Created {len(input_segments)} training segments")

# 2) PyTorch Dataset class
class AudioDataset(Dataset):
    def __init__(self, input_segments, target_segments):
        assert len(input_segments) == len(target_segments), "Input and target must have same length"
        self.input_segments = input_segments
        self.target_segments = target_segments
    
    def __len__(self):
        return len(self.input_segments)
    
    def __getitem__(self, idx):
        input_tensor = torch.FloatTensor(self.input_segments[idx])
        target_tensor = torch.FloatTensor(self.target_segments[idx])
        return input_tensor, target_tensor

# Create dataset and split
dataset_size = len(input_segments)
val_size = int(dataset_size * train_cfg.VALIDATION_SPLIT)
train_size = dataset_size - val_size

# Shuffle indices for random split
indices = list(range(dataset_size))
random.shuffle(indices)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_input = [input_segments[i] for i in train_indices]
train_target = [target_segments[i] for i in train_indices]
val_input = [input_segments[i] for i in val_indices]
val_target = [target_segments[i] for i in val_indices]

train_dataset = AudioDataset(train_input, train_target)
val_dataset = AudioDataset(val_input, val_target)

train_loader = DataLoader(train_dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# 3) Real-time optimized model
class AcousticSimulator(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, model_type='LSTM'):
        super(AcousticSimulator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("model_type must be 'LSTM' or 'GRU'")
            
        # Simple output layer for audio synthesis
        self.output = nn.Linear(hidden_size, input_size)
        self.tanh = nn.Tanh()  # Keep output in reasonable range
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, sequence_length, input_size)
        rnn_out, hidden = self.rnn(x, hidden)
        output = self.output(rnn_out)
        output = self.tanh(output)  # Apply tanh to prevent clipping
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        if self.model_type == 'LSTM':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            return (h0, c0)
        else:  # GRU
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            return h0

# 4) Setup device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = AcousticSimulator(
    input_size=1,
    hidden_size=train_cfg.HIDDEN_SIZE,
    num_layers=train_cfg.NUM_LAYERS,
    model_type=train_cfg.MODEL_TYPE
).to(device)

# Count parameters for real-time feasibility
total_params = sum(p.numel() for p in model.parameters())
print(f"Model has {total_params} parameters")

# 5) Training setup
def esr_loss(target, pred):
    """Error-to-Signal Ratio loss"""
    target_pre, pred_pre = pre_emphasis(target), pre_emphasis(pred)
    numerator = torch.sum(torch.pow(target_pre - pred_pre, 2), dim=-1)
    denominator = torch.sum(torch.pow(target_pre, 2), dim=-1) + 1e-10
    return torch.mean(numerator / denominator)

optimizer = optim.Adam(model.parameters(), lr=train_cfg.LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

# Training tracking
train_losses = []
val_losses = []
best_val_loss = float('inf')

# 6) Training loop
print(f"Starting training for {train_cfg.NUM_EPOCHS} epochs...")

for epoch in range(train_cfg.NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss_epoch = 0.0
    num_train_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Initialize hidden state
        hidden = model.init_hidden(inputs.size(0), device)
        
        optimizer.zero_grad()
        
        outputs, _ = model(inputs, hidden)
        loss = esr_loss(targets, outputs)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss_epoch += loss.item()
        num_train_batches += 1
        
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch+1}/{train_cfg.NUM_EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
    
    avg_train_loss = train_loss_epoch / num_train_batches
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_loss_epoch = 0.0
    num_val_batches = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            hidden = model.init_hidden(inputs.size(0), device)
            outputs, _ = model(inputs, hidden)
            loss = esr_loss(targets, outputs)
            
            val_loss_epoch += loss.item()
            num_val_batches += 1
    
    avg_val_loss = val_loss_epoch / num_val_batches
    val_losses.append(avg_val_loss)
    
    scheduler.step(avg_val_loss)
    
    print(f'Epoch {epoch+1}/{train_cfg.NUM_EPOCHS}:')
    print(f'  Train Loss: {avg_train_loss:.6f}')
    print(f'  Val Loss: {avg_val_loss:.6f}')
    print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.8f}')
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'config': train_cfg.__dict__
        }, os.path.join(cfg.models_base, 'best_model.pth'))
        print(f'  → New best model saved (val_loss: {best_val_loss:.6f})')
    
    print('-' * 60)

# 7) Save final model and training history
torch.save({
    'model_state_dict': model.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'config': train_cfg.__dict__
}, os.path.join(cfg.models_base, 'final_model.pth'))

print("Training completed!")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Final model saved to: {os.path.join(cfg.models_base, 'final_model.pth')}")

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('ESR Loss')
plt.title('Training History')
plt.legend()
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.plot(train_losses[-50:], label='Training Loss (last 50 epochs)')
plt.plot(val_losses[-50:], label='Validation Loss (last 50 epochs)')
plt.xlabel('Epoch')
plt.ylabel('ESR Loss')
plt.title('Training History (Zoomed)')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(cfg.models_base, 'training_history.png'))
plt.show()

# 8) Export model for JUCE integration
# Save in ONNX format for easier C++ integration
model.eval()
dummy_input = torch.randn(1, segment_length_samples, 1).to(device)
dummy_hidden = model.init_hidden(1, device)

torch.onnx.export(
    model,
    (dummy_input, dummy_hidden),
    os.path.join(cfg.models_base, 'acoustic_sim_model.onnx'),
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['audio_input', 'hidden_state'],
    output_names=['audio_output', 'hidden_output'],
    dynamic_axes={
        'audio_input': {1: 'sequence_length'},
        'audio_output': {1: 'sequence_length'}
    }
)

print(f"ONNX model saved for JUCE integration: {os.path.join(cfg.models_base, 'acoustic_sim_model.onnx')}")
print("Model is ready for real-time inference in JUCE!")

# Model summary for JUCE integration
print("\n" + "="*60)
print("JUCE INTEGRATION SUMMARY")
print("="*60)
print(f"Model Type: {train_cfg.MODEL_TYPE}")
print(f"Hidden Size: {train_cfg.HIDDEN_SIZE}")
print(f"Number of Layers: {train_cfg.NUM_LAYERS}")
print(f"Total Parameters: {total_params}")
print(f"Input Size: 1 (mono audio)")
print(f"Sample Rate: {cfg.SAMPLE_RATE_HZ} Hz")
print(f"Recommended Buffer Size: {segment_length_samples} samples ({train_cfg.SEGMENT_LENGTH_SEC}s)")
print("="*60)


# Infer
session = ort.InferenceSession(f"{os.path.join(cfg.models_base, 'acoustic_sim_model.onnx')}")

audio, sr = sf.read(f"{cfg.career_sg_base}/latin_8_180BPM.wav")
audio = peak_norm(audio)
print(f"Audio shape: {audio.shape}, sample rate: {sr}")
viz.plot_waveforms(audio)

for input in session.get_inputs():
  print(f"Input name: {input.name}, shape: {input.shape}, type: {input.type}")
# Input name: audio_input, shape: [1, 'sequence_length', 1], type: tensor(float)
# Input name: hidden_state, shape: [1, 1, 32], type: tensor(float)
# Input name: onnx::LSTM_2, shape: [1, 1, 32], type: tensor(float)

audio_input = audio.astype(np.float32)[np.newaxis, :] # (1, samples, 1)
print(audio_input)
hidden_state = np.zeros((1,1,32), dtype=np.float32)
cell_state = np.zeros((1,1,32), dtype=np.float32)

inputs = {
    'audio_input': audio_input,
    'hidden_state': hidden_state,
    'onnx::LSTM_2': cell_state
}

result = session.run(None, inputs)

output_audio = result[0].squeeze()
sf.write(f"{cfg.career_sg_base}/acousitc_latin_out.wav", output_audio, sr)


# infer with pytorch:
model_import = AcousticSimulator(input_size=1,
    hidden_size=train_cfg.HIDDEN_SIZE,
    num_layers=train_cfg.NUM_LAYERS,
    model_type=train_cfg.MODEL_TYPE)

sd = torch.load(os.path.join(cfg.models_base, "final_model.pth"), weights_only=True)
print(sd.keys())
print(sd["model_state_dict"])

# prints:
# dict_keys(['model_state_dict', 'train_losses', 'val_losses', 'config'])
# OrderedDict({'rnn.weight_ih_l0': tensor([[ 1.3435e-01],
#         [ 1.2751e-01],
#         [ 9.6346e-02],
#         [-2.0656e-05],
#         [-1.4493e-01],
#         [-1.0915e-02],
#         [-1.6428e-01],
#         [ 1.5294e-01],
#         [ 1.6276e-01],
#         [-1.6851e-01],
#         [ 4.6979e-02],
#         [-6.0870e-03],
#         [ 2.8793e-02],
#         [-9.1162e-02],
#         [ 7.7717e-02],
#         [ 4.5701e-02],
#         [-1.4210e-01],
#         [-2.0997e-02],
#         [-6.6713e-02],
#         [ 7.2653e-02],
#         [-9.5228e-02],
#         [-1.1759e-01],
#         [ 1.5487e-04],
#         [ 5.6535e-02],
#         [-1.7815e-01],
#         [-1.6021e-01],
#         [ 3.9202e-02],
#         [-9.4408e-02],
#         [-2.1432e-01],
#         [ 1.5531e-01],
#         [-1.0782e-01],
#         [-1.4622e-01],
#         [-1.6112e-01],
#         [-5.0220e-02],
#         [-1.1044e-01],
#         [ 6.4075e-02],
#         [-9.9885e-02],
#         [ 1.5043e-01],
#         [-1.4061e-01],
#         [ 5.0374e-02],
#         [ 1.3751e-01],
#         [ 6.6113e-02],
#         [-6.4430e-02],
#         [ 1.2134e-01],
#         [-4.0941e-03],
#         [-1.3977e-01],
#         [-6.9078e-02],
#         [-1.3902e-01],
#         [ 2.3492e-03],
#         [-1.4653e-01],
#         [-1.6642e-02],
#         [-1.1315e-01],
#         [-6.5321e-02],
#         [-1.9566e-01],
#         [ 4.2874e-02],
#         [ 1.0315e-01],
#         [ 6.3666e-04],
#         [-2.3382e-01],
#         [-1.8843e-01],
#         [-1.2214e-01],
#         [ 3.0571e-03],
#         [ 9.3702e-02],
#         [-6.6230e-02],
#         [-5.2262e-03],
#         [ 6.3240e-02],
#         [ 1.9469e-02],
#         [ 3.3164e-02],
#         [ 5.8702e-02],
#         [ 3.0350e-02],
#         [ 8.0029e-02],
#         [-1.1792e-01],
#         [ 2.9255e-02],
#         [ 1.1013e-01],
#         [ 4.9862e-02],
#         [ 7.4729e-02],
#         [ 1.0998e-01],
#         [-8.9441e-02],
#         [ 1.4038e-01],
#         [-1.3282e-02],
#         [-3.5456e-02],
#         [ 1.4832e-01],
#         [ 1.4775e-01],
#         [ 5.3790e-02],
#         [ 1.1331e-01],
#         [-7.5365e-02],
#         [-1.4284e-01],
#         [ 9.8354e-02],
#         [ 8.5554e-02],
#         [-7.4095e-02],
#         [-1.0792e-02],
#         [-4.5260e-02],
#         [-1.0857e-01],
#         [-6.9767e-02],
#         [ 4.2351e-02],
#         [ 1.1820e-01],
#         [-4.8854e-03],
#         [ 3.0919e-02],
#         [ 7.2824e-02],
#         [-1.1038e-01],
#         [-4.3625e-03],
#         [-2.1677e-01],
#         [-6.8094e-02],
#         [ 1.0817e-01],
#         [-3.3356e-02],
#         [ 1.8668e-01],
#         [ 1.6976e-01],
#         [ 1.0759e-01],
#         [ 9.2029e-02],
#         [ 9.8223e-02],
#         [-1.4745e-01],
#         [-9.3177e-02],
#         [ 6.5042e-04],
#         [ 1.7024e-01],
#         [ 2.0680e-02],
#         [-1.0660e-01],
#         [-1.1963e-01],
#         [ 1.1880e-01],
#         [ 6.7342e-02],
#         [-7.0557e-02],
#         [-1.5415e-01],
#         [ 7.0105e-02],
#         [-2.4668e-01],
#         [ 1.5299e-01],
#         [ 3.8422e-03],
#         [-1.2436e-01],
#         [-6.6801e-02],
#         [ 3.2629e-02],
#         [-3.4132e-01]], device='cuda:0'), 'rnn.weight_hh_l0': tensor([[-0.1253,  0.0819,  0.0133,  ...,  0.1227, -0.1785, -0.2571],
#         [ 0.0992,  0.1154,  0.0864,  ..., -0.0928, -0.0192,  0.1054],
#         [-0.1590, -0.1543, -0.0809,  ...,  0.0474, -0.1242, -0.0010],
#         ...,
#         [ 0.1168,  0.0221,  0.0555,  ..., -0.0353, -0.0392, -0.1137],
#         [ 0.0031, -0.1288, -0.0907,  ..., -0.0148, -0.1467,  0.0776],
#         [-0.2054,  0.1953, -0.2988,  ..., -0.0555, -0.1301, -0.2825]],
#        device='cuda:0'), 'rnn.bias_ih_l0': tensor([-0.1323,  0.0949,  0.1538,  0.1555,  0.0470, -0.0506,  0.0984,  0.1439,
#          0.0257, -0.0508,  0.0594, -0.1012, -0.0865,  0.1256,  0.2445, -0.0115,
#          0.0338,  0.0940, -0.1760, -0.0697, -0.1017, -0.2358, -0.0911,  0.0121,
#         -0.1717,  0.0214, -0.0455,  0.0419, -0.1310, -0.0281,  0.0103, -0.0004,
#          0.0936, -0.1601,  0.1282, -0.0295, -0.2757, -0.2022, -0.1403, -0.2705,
#          0.0880, -0.0451, -0.0599, -0.1527, -0.0805, -0.0807, -0.1164, -0.1881,
#         -0.1795, -0.0938, -0.0375, -0.1087, -0.1120, -0.1501,  0.1212, -0.1585,
#         -0.1461, -0.1350, -0.1847, -0.0698, -0.1656, -0.0290, -0.0780, -0.0944,
#         -0.0037,  0.0298, -0.0281, -0.0029, -0.0470, -0.1650, -0.0142,  0.1683,
#         -0.1118, -0.0799, -0.1283,  0.0305, -0.1003,  0.1770,  0.1145,  0.0572,
#          0.1550, -0.0298, -0.0501,  0.0238, -0.0992, -0.0371,  0.0107, -0.1254,
#          0.0316, -0.1545,  0.1799, -0.0412, -0.1258,  0.1500,  0.0871,  0.1088,
#          0.1760, -0.0086, -0.1433, -0.1152,  0.1039,  0.0318, -0.0968,  0.0115,
#         -0.0935,  0.0454,  0.1119, -0.0610, -0.0132, -0.1595,  0.2812,  0.1510,
#         -0.1433,  0.1660,  0.1305, -0.1377, -0.1105, -0.0368,  0.0560, -0.1295,
#         -0.0549,  0.1179,  0.0472,  0.1114, -0.0439, -0.0924,  0.0273,  0.0221],
#        device='cuda:0'), 'rnn.bias_hh_l0': tensor([ 0.0229,  0.0711,  0.1218, -0.1446,  0.0634, -0.1546,  0.0751,  0.0575,
#         -0.1826, -0.0791, -0.0505, -0.1222,  0.0086,  0.0501,  0.0382,  0.0632,
#         -0.0824, -0.0284, -0.1635,  0.1447,  0.0174, -0.0487, -0.1635, -0.1431,
#         -0.0375,  0.1469,  0.1197,  0.0611, -0.1098,  0.0186, -0.1842,  0.2833,
#         -0.1699,  0.0661, -0.1126,  0.0325, -0.1227, -0.0154,  0.0482,  0.0446,
#          0.0798, -0.0708, -0.0725, -0.1359, -0.0060, -0.1754, -0.0069,  0.0560,
#         -0.0477, -0.1489, -0.1425, -0.0694,  0.0281, -0.0891, -0.0123, -0.0832,
#         -0.1545,  0.0004, -0.1570,  0.0172,  0.0181,  0.1028, -0.0919, -0.1961,
#          0.1009, -0.0621,  0.1133,  0.0264,  0.1109,  0.1030,  0.1248, -0.0135,
#          0.0517,  0.1072,  0.1660, -0.1101, -0.0767,  0.1825,  0.1920, -0.0237,
#          0.0648, -0.1507,  0.0977, -0.1400,  0.1290, -0.0577,  0.0089,  0.1824,
#         -0.0277,  0.0747,  0.1400, -0.0971, -0.1141, -0.0115,  0.0955, -0.0050,
#         -0.1246,  0.0423, -0.0406, -0.0938,  0.0218,  0.0793, -0.0456,  0.2781,
#         -0.0351, -0.1532,  0.0976, -0.1607, -0.1527,  0.0829,  0.0264, -0.0967,
#         -0.0739, -0.1038,  0.1241, -0.0368, -0.2218, -0.0250,  0.0718, -0.1687,
#          0.0704,  0.0076,  0.0536, -0.0365,  0.1299,  0.1125, -0.0404,  0.2068],
#        device='cuda:0'), 'output.weight': tensor([[ 0.0643, -0.0242,  0.1154,  0.0175,  0.0093, -0.0995,  0.0716,  0.1117,
#          -0.0059,  0.1047,  0.1347,  0.0921,  0.1063,  0.0653,  0.0787,  0.0242,
#           0.0662, -0.0585, -0.0228, -0.0403,  0.0011,  0.0975, -0.0565,  0.1267,
#           0.1074, -0.0229,  0.0537, -0.1591, -0.0180,  0.0927, -0.0753,  0.0463]],
#        device='cuda:0'), 'output.bias': tensor([-0.0432], device='cuda:0')})
# AcousticSimulator(
#   (rnn): LSTM(1, 32, batch_first=True)
#   (output): Linear(in_features=32, out_features=1, bias=True)
#   (tanh): Tanh()
# )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_import.load_state_dict(sd["model_state_dict"])
model_import = model_import.to(device)
model_import.eval()

input_sound, sr = librosa.load(f"{cfg.career_sg_base}/latin_8_180BPM.wav", sr = cfg.SAMPLE_RATE_HZ)
audio_tensor = peak_norm(input_sound)

print(f"Original audio stats: min={audio_tensor.min():.6f}, max={audio_tensor.max():.6f}, mean={audio_tensor.mean():.6f}")

if len(audio_tensor.shape) == 1:
    audio = audio_tensor.reshape(-1, 1)

audio_input = audio.astype(np.float32)[np.newaxis, :]
print(f"Audio input shape: {audio_input.shape}")
print(f"Audio input stats: min={audio_input.min():.6f}, max={audio_input.max():.6f}")

# Original audio stats: min=-1.022457, max=1.000000, mean=-0.000041
# Audio input shape: (1, 1176000, 1)
# Audio input stats: min=-1.022457, max=1.000000

# Try CPU inference to avoid cuDNN issues
print("Trying CPU inference to bypass cuDNN...")
model_import = model_import.cpu()
device_cpu = torch.device('cpu')

# Convert to tensor for CPU
audio_tensor = torch.from_numpy(audio_input).to(device_cpu).contiguous()
print(f"Input tensor shape: {audio_tensor.shape}")
print(f"Input tensor device: {audio_tensor.device}")

with torch.no_grad():
    hidden = model_import.init_hidden(1, device_cpu)
    
    print(f"Hidden shapes: h={hidden[0].shape}, c={hidden[1].shape}")
    
    # Process in smaller chunks to be safe
    chunk_size = int(2 * sr)  # 2 seconds
    output_chunks = []
    
    print(f"Processing {audio_tensor.size(1)} samples in chunks of {chunk_size}")
    
    for i in range(0, audio_tensor.size(1), chunk_size):
        end_idx = min(i + chunk_size, audio_tensor.size(1))
        chunk = audio_tensor[:, i:end_idx, :].contiguous()
        print(f"Processing chunk {i//chunk_size + 1}/{(audio_tensor.size(1) + chunk_size - 1)//chunk_size}, shape: {chunk.shape}")
        
        chunk_output, hidden = model_import(chunk, hidden)
        output_chunks.append(chunk_output)
    
    output_tensor = torch.cat(output_chunks, dim=1)
    
print(f"Output tensor shape: {output_tensor.shape}")
print(f"Output tensor stats: min={output_tensor.min():.6f}, max={output_tensor.max():.6f}, mean={output_tensor.mean():.6f}")

# Convert back to numpy and save
output_audio = output_tensor.squeeze().cpu().numpy()
print(f"Final output shape: {output_audio.shape}")
print(f"Final output stats: min={output_audio.min():.6f}, max={output_audio.max():.6f}")

# Try amplifying the output (quick test)
amplified_output = output_audio * 10000  # Amplify by 10,000x
print(f"Amplified stats: min={amplified_output.min():.6f}, max={amplified_output.max():.6f}")

# Save both versions
sf.write(f"{cfg.career_sg_base}/pytorch_latin_out_original.wav", output_audio, int(sr))
sf.write(f"{cfg.career_sg_base}/pytorch_latin_out_amplified.wav", amplified_output, int(sr))

# Also try peak normalization (same as input processing)
if np.max(np.abs(output_audio)) > 0:
    output_normalized = peak_norm(output_audio)
    sf.write(f"{cfg.career_sg_base}/pytorch_latin_out_normalized.wav", output_normalized, int(sr))
    print(f"Normalized stats: min={output_normalized.min():.6f}, max={output_normalized.max():.6f}")
print("PyTorch inference completed!")


