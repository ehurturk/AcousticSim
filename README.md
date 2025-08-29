# AcousticSim

![logo](packaging/icon.png)

A JUCE application that simulates acoustic guitar, with aim to integrate with an ASIO interface.

To generate the build folder and necessary build scripts, run:

```
cmake --build ./build --config [DEBUG/RELEASE] --target all -j 8 --
```

## MacOS

To build DMG release, run:

```
cmake --build ./build --target dmg --config Release
```

# Notes

Original intention was to use LSTM32 model for real time inference of electric to acoustic signal conversion.
But due to complexity of acoustic guitar features (such as timbre, resonance, wood characteristics, mic placement, pluck etc.) the model
had a hard time learning the features.

So, for now I will utilize DSP techniques to model acoustic sound, and later try to fiddle with the DL approach.

## Acoustic sound from the electric guitar using DSP techniques

- extensions of Karplus-Strong model
- general digital waveguide modeling
- aim: magnetic pickup sound -> acoustic guitar (with acoustic-like timbre)
- digital filtering?
- acoustic guitar chain:
    excitation -> string -> bridge -> body -> radiation
    Essential functions for body are:
      (a) amplification of sound by higher efficiency of string energy transfer than the string itself has
      (b) adding resonance modes to the low frequency portion of the spectrum
      (c) adding dense distribution of mid-to-high frequency modes with audible body reverberation
    Characteristics of the acoustic guitar:
        - Time-frequency response of the body (not just the magnitude response)

## Potential ideas for LSTM/WaveNet/GRU

1) Instead of pure time-domain ESR loss, combine the frequency domain with time domain with biased weights.
2) Create a learnable output parameter in the model
