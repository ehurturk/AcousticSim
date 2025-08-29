#pragma once

#include "juce_audio_basics/juce_audio_basics.h"
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

#if (MSVC)
    #include "ipps.h"
#endif

struct EnvelopeFollower
{
    void prepare (double sr, float atkMs = 3.0f, float relMs = 80.0f);
    void setTimes (float atkMs, float relMs);
    float processSample (float xAbs);
    float processBlockRMS (const juce::AudioBuffer<float>& buf);

    float env { 0 }, aAtk { 0 }, aRel { 0 };

    double samplerate { 44100.0 };
};

class PluginProcessor : public juce::AudioProcessor
{
public:
    PluginProcessor();
    ~PluginProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();

    void loadIRFile (const juce::File& file);

    juce::AudioProcessorValueTreeState& getAPVTS();

private:
    juce::AudioProcessorValueTreeState apvts;
    // DSP modules
    using IIR = juce::dsp::IIR::Filter<float>;
    using Coeffs = juce::dsp::IIR::Coefficients<float>;
    juce::dsp::ProcessorDuplicator<IIR, Coeffs> preHPF, staticDeQuack, tiltLowShelf, tiltHighShelf;
    juce::dsp::Convolution bodyConv; // body IR

    juce::LinearSmoothedValue<float> tiltSmoothedDbPerOct { 0.0f };

    juce::dsp::DelayLine<float, juce::dsp::DelayLineInterpolationTypes::Linear> earlyL { 48000 };
    juce::dsp::DelayLine<float, juce::dsp::DelayLineInterpolationTypes::Linear> earlyR { 48000 };

    EnvelopeFollower envFollower;

    float erGain = juce::Decibels::decibelsToGain (-20.0f);
    int erTap1Samp = 160; // 3.6 ms with 44.1kHz
    int erTap2Samp = 420; // 9.5 ms with 44.1kHz

    void updateTiltFilters (float dbPerOct, double sr);
    void updateMicModel (double sr);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginProcessor)
};
