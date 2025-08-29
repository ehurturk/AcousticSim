#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "juce_audio_processors/juce_audio_processors.h"
#include "juce_core/juce_core.h"
#include <juce_dsp/juce_dsp.h>

void EnvelopeFollower::prepare (double sr, float atkMs, float relMs)
{
    samplerate = sr;
    setTimes (atkMs, relMs);
    env = 0.0f;
}

void EnvelopeFollower::setTimes (float atkMs, float relMs)
{
    aAtk = (float) std::exp (-1.0 / (samplerate * (atkMs / 1000.0)));
    aRel = (float) std::exp (-1.0 / (samplerate * (relMs / 1000.0)));
}

float EnvelopeFollower::processSample (float xAbs)
{
    float coeff = (xAbs > env) ? aAtk : aRel;
    env = coeff * env + (1.0f - coeff) * xAbs;
    return env;
}

float EnvelopeFollower::processBlockRMS (const juce::AudioBuffer<float>& buf)
{
    const int n = buf.getNumSamples();
    float acc = 0.f;
    for (int ch = 0; ch < buf.getNumChannels(); ++ch)
    {
        const float* d = buf.getReadPointer (ch);
        for (int i = 0; i < n; ++i)
            acc += d[i] * d[i];
    }
    float rms = std::sqrt (acc / (float) (n * juce::jmax (1, buf.getNumChannels())));
    return processSample (rms);
}

//==============================================================================
PluginProcessor::PluginProcessor()
    : AudioProcessor (BusesProperties()
#if !JucePlugin_IsMidiEffect
    #if !JucePlugin_IsSynth
              .withInput ("Input", juce::AudioChannelSet::stereo(), true)
    #endif
              .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
#endif
              ),
      apvts (*this, nullptr, "PARAMS", createLayout())
{
}

PluginProcessor::~PluginProcessor()
{
}

//==============================================================================
const juce::String PluginProcessor::getName() const
{
    return JucePlugin_Name;
}

bool PluginProcessor::acceptsMidi() const
{
#if JucePlugin_WantsMidiInput
    return true;
#else
    return false;
#endif
}

bool PluginProcessor::producesMidi() const
{
#if JucePlugin_ProducesMidiOutput
    return true;
#else
    return false;
#endif
}

bool PluginProcessor::isMidiEffect() const
{
#if JucePlugin_IsMidiEffect
    return true;
#else
    return false;
#endif
}

double PluginProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int PluginProcessor::getNumPrograms()
{
    return 1; // NB: some hosts don't cope very well if you tell them there are 0 programs,
    // so this should be at least 1, even if you're not really implementing programs.
}

int PluginProcessor::getCurrentProgram()
{
    return 0;
}

void PluginProcessor::setCurrentProgram (int index)
{
    juce::ignoreUnused (index);
}

const juce::String PluginProcessor::getProgramName (int index)
{
    juce::ignoreUnused (index);
    return {};
}

void PluginProcessor::changeProgramName (int index, const juce::String& newName)
{
    juce::ignoreUnused (index, newName);
}

//==============================================================================
void PluginProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    envFollower.prepare (sampleRate, 3.0f, 120.0f);
    tiltSmoothedDbPerOct.reset (sampleRate, 0.03f);

    const auto nCh = (juce::uint32) juce::jmax (1, getMainBusNumOutputChannels());
    juce::dsp::ProcessSpec spec { sampleRate, (juce::uint32) samplesPerBlock, nCh };

    preHPF.reset();
    preHPF.prepare (spec);
    staticDeQuack.reset();
    staticDeQuack.prepare (spec);
    tiltLowShelf.reset();
    tiltLowShelf.prepare (spec);
    tiltHighShelf.reset();
    tiltHighShelf.prepare (spec);

    preHPF.state = Coeffs::makeHighPass (sampleRate, 55.0, 0.707f);
    staticDeQuack.state = Coeffs::makePeakFilter (sampleRate, 2800.0, 1.2f, juce::Decibels::decibelsToGain (-2.5f));

    tiltLowShelf.state = Coeffs::makeLowShelf (sampleRate, 150.0, 0.707f, 1.0f);
    tiltHighShelf.state = Coeffs::makeHighShelf (sampleRate, 6000.0, 0.707f, 1.0f);

    updateTiltFilters (-3.0f, sampleRate); // start with -3 db/oct

    bodyConv.reset();
    bodyConv.prepare (spec);
    setLatencySamples ((int) bodyConv.getLatency());
}

void PluginProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

bool PluginProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
#if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
#else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
        && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
    #if !JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
    #endif

    return true;
#endif
}

void PluginProcessor::processBlock (juce::AudioBuffer<float>& buffer,
    juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused (midiMessages);

    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    // clear garbage output channels
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    juce::dsp::AudioBlock<float> block (buffer);
    juce::dsp::ProcessContextReplacing<float> ctx (block);

    float env = envFollower.processBlockRMS (buffer);
    float tiltStrength = apvts.getRawParameterValue ("tilt")->load(); // 0..1
    float dbPerOct = juce::jmap (tiltStrength, 0.0f, 1.0f, -4.0f, -1.0f); // -4.0..-1.0

    dbPerOct += juce::jlimit (-1.0f, 1.0f, 3.0f * (env - 0.08f));
    tiltSmoothedDbPerOct.setTargetValue (dbPerOct);

    // pre cleanup
    preHPF.process (ctx);
    staticDeQuack.process (ctx);

    // body
    if (apvts.getRawParameterValue ("useIR")->load() > 0.5f)
    {
        bodyConv.process (ctx);
    }
    else
    {
        // maybe a parametric EQ chain instead of IR?
    }

    // dynamic tilt
    updateTiltFilters (tiltSmoothedDbPerOct.getNextValue(), getSampleRate());
    tiltLowShelf.process (ctx);
    tiltHighShelf.process (ctx);

    const int N = buffer.getNumSamples();
    for (int ch = 0; ch < juce::jmin (2, buffer.getNumChannels()); ++ch)
    {
        float* d = buffer.getWritePointer (ch);
        for (int i = 0; i < N; ++i)
        {
            float x = d[i];
            earlyL.pushSample (0, x);
            earlyR.pushSample (1, x);
            juce::Logger::outputDebugString ("BINARY SEARCH LMAO");
            float er = 0.0f;
            er += earlyL.popSample (0, erTap1Samp);
            er += earlyR.popSample (1, erTap2Samp) * 0.8f;
            d[i] = d[i] + erGain * er; // tiny room
        }
    }
    float wet = apvts.getRawParameterValue ("wet")->load();
    if (wet < 1.0f)
    {
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        {
            float* d = buffer.getWritePointer (ch);
            for (int i = 0; i < N; ++i)
            {
                d[i] = wet * d[i] + (1.0f - wet) * d[i];
            }
        }
    }
    buffer.applyGain (juce::Decibels::decibelsToGain (apvts.getRawParameterValue ("output")->load()));
}

//==============================================================================
bool PluginProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* PluginProcessor::createEditor()
{
    return new PluginEditor (*this, apvts);
}

//==============================================================================
void PluginProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    juce::ValueTree state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void PluginProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));
    if (xmlState.get() != nullptr)
    {
        if (xmlState->hasTagName (apvts.state.getType()))
        {
            apvts.replaceState (juce::ValueTree::fromXml (*xmlState));
        }
    }
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new PluginProcessor();
}

juce::AudioProcessorValueTreeState::ParameterLayout PluginProcessor::createLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.emplace_back (std::make_unique<juce::AudioParameterFloat> (juce::ParameterID { "wet", 1 }, "Wet", juce::NormalisableRange<float> (0.f, 1.f, 0.001f), 0.7f));
    params.emplace_back (std::make_unique<juce::AudioParameterFloat> (juce::ParameterID { "tilt", 1 }, "Dynamic Tilt Strength", juce::NormalisableRange<float> (0.f, 1.f, 0.001f), 0.6f)); // scales -4..-1 dB/oct
    params.emplace_back (std::make_unique<juce::AudioParameterFloat> (juce::ParameterID { "distance", 1 }, "Mic Distance", juce::NormalisableRange<float> (0.1f, 0.6f, 0.001f), 0.25f));
    params.emplace_back (std::make_unique<juce::AudioParameterBool> (juce::ParameterID { "useIR", 1 }, "Use IR", true));
    params.emplace_back (std::make_unique<juce::AudioParameterFloat> (juce::ParameterID { "output", 1 }, "Output", juce::NormalisableRange<float> (-24.f, 6.f, 0.1f), 0.0f));
    // TODO: ir path parameter?
    return { params.begin(), params.end() };
}

void PluginProcessor::loadIRFile (const juce::File& file)
{
    if (!file.existsAsFile())
    {
        return;
    }

    bodyConv.loadImpulseResponse (file, juce::dsp::Convolution::Stereo::yes, juce::dsp::Convolution::Trim::yes, 1024);
    setLatencySamples ((int) bodyConv.getLatency());
}

void PluginProcessor::updateTiltFilters (float dbPerOct, double sr)
{
    const double lowFc = 150.0;
    const double highFc = 6000.0;
    const float lowGainDb = juce::jlimit (-12.0f, +12.0f, 0.5f * dbPerOct * 3.3f);
    const float highGainDb = juce::jlimit (-12.0f, +12.0f, -0.5f * dbPerOct * 4.3f);

    tiltLowShelf.state = Coeffs::makeLowShelf (sr, 150.0, 0.707f, juce::Decibels::decibelsToGain (lowGainDb));
    tiltHighShelf.state = Coeffs::makeHighShelf (sr, 6000.0, 0.707f, juce::Decibels::decibelsToGain (highGainDb));
}

void PluginProcessor::updateMicModel (double sr)
{
    juce::ignoreUnused (sr);
}

juce::AudioProcessorValueTreeState& PluginProcessor::getAPVTS()
{
    return apvts;
}
