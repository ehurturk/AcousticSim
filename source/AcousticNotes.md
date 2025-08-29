# Acoustic notes

Potential ideas to model acoustic signal from a electric guitar DI.

Differences between electric and acoustic:

- DI (pickup) = string excitation: clear, dry, little body
- Mic'd acoustic = excitation * body/air transfer + room/mic color (as reverb)

Need a $T(f)$ transfer function that turns DI into a mic-like tone + layer amplitude and transient-dependent tweaks.

## Signal Flow

```mermaid
flowchart TD
  A[DI] --> B{Pre-cleanup: HP @ 40-60 Hz};
  B --> C{De-quack}
  C --> D{Body tone: Short IR convolution}
  D --> E{Early reflections: 3-12 ms stereo FIR}
  E --> F{Envelope->spectral tilt }
  E --> G{Dynamic de-quack}
  E --> H{Transient softener}
  F --> I{Mic distance}
  G --> I
  H --> I
  I --> J{Off-axis}
  J --> K{Post polish}
  K --> L{OUT}
```
