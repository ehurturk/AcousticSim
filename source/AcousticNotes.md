# Acoustic notes

Potential ideas to model acoustic signal from a electric guitar DI.

Differences between electric and acoustic:

- DI (pickup) = string excitation: clear, dry, little body
- Mic'd acoustic = excitation * body/air transfer + room/mic color (as reverb)

Need a $T(f)$ transfer function that turns DI into a mic-like tone + layer amplitude and transient-dependent tweaks.

## Signal Flow

```mermaid
flowchart TD
  A[DI] --> B{Pre-cleanup: HP @ 40-60 Hz, hum notch (50/60 Hz), DC blocker};
  B --> C{De-quack I (static): gentle dip at 2-4 kHz (Q ~ 1.2)}
  C --> D{Body tone: Short IR convolution (body modes, air)}
  D --> E{Early reflections: 3-12 ms stereo FIR}
  E --> F{Envelope->spectral tilt (-dB/oct when soft, brighter when hard)}
  E --> G{Dynamic de-quack}
  E --> H{Transient softener}
  F --> I{Mic distance (low-shelf + HF loss)}
  G --> I
  H --> I
  I --> J{Off-axis}
  J --> K{Post polish}
  K --> L{OUT}
```
