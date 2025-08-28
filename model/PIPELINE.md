# Electric-to-Acoustic Model Development Plan

## Phase 1: Foundation & Data Pipeline

### Data Exploration

- [x] Load & inspect data
- [x] Visualize waveforms (electric vs acoustic)
- [x] Plot spectrograms to identify frequency differences
- [x] Compute and log statistics (RMS, peak levels, frequency content)
- [x] Identify and mark problematic segments (silence and dead notes)
- [x] Create data quality CSV:
  - Signal to noise ratio
  - Correlation between electric/acoustic pairs
  - Frequency response differences
  - Dynamic range analysis
  - Onset alignment accuracy

### Preprocessing Pipeline

- [ ] Implement core preprocessing pipeline
  - Audio loading with 44.1 kHz sample rate
  - Alignment algorithm using 1) DTW-based time warping or cross-correlation
  - Silence/dead note removal
  - Peak normalization to [-1,1]
- [ ] Create train/val/test split:
  - Shuffle to create genre distribution across slits
- [ ] Save data (+ with metadata JSON)
- [ ] Do data augmentation:
  - Amplitude variation
  - Pitch shifts
  - Time stretch

### Model Development

- [ ] Implement baseline Conv1D model (5 layer Conv1D with skip connections)
  - Implement forward pass, add gradient checkpointing
- [ ] Implement multi-component loss function:
  - Time-domain L1 loss
  - Multi-resolution STFT loss (512,1024,2048 FFT)
  - Perceptual loss (A-weighted frequency)
  - Energy conservation loss
- [ ] Implement training infrastructure
  - Gradient clipping
  - Learning rate scheduling
  - Early stopping
  - **Model checkpointing**
  - Train for 50 epochs
    - Check silent outputs -> fix normalization
    - Verify loss id decreasing -> adjust learning rate
    - Listen to outputs every 10 epochs

### Improving the Model

- [ ] TCN Architecture
  - Implement temporal convolutional network
  - Causal dilated convolutions
  - Exponentially growing receptive field
  - Residual connections
- [ ] Add frequency-aware components:
  - Add frequency-domain processing branch
  - Learned filterbank (SincNet-style)
  - Sub-band processing
  - Harmonic enhancement layer
  - Phase-aware loss function
- [ ] WaveNet Model
  - Implement WaveNet style architecture with the forward loop
- [ ] LSTM32 Model
  - Aim for low parameter model for <10ms real time round-trip latency
