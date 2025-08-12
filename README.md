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