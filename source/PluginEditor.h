#pragma once

#include "BinaryData.h"
#include "PluginProcessor.h"
#include "melatonin_inspector/melatonin_inspector.h"

using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
using ButtonAttachment = juce::AudioProcessorValueTreeState::ButtonAttachment;

//==============================================================================
class PluginEditor : public juce::AudioProcessorEditor
{
public:
    explicit PluginEditor (PluginProcessor&, juce::AudioProcessorValueTreeState&);
    ~PluginEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    PluginProcessor& processorRef;
    // std::unique_ptr<melatonin::Inspector> inspector;
    // juce::TextButton inspectButton { "Test Acoustic Sim" };

    juce::AudioProcessorValueTreeState& apvtsRef;

    juce::Label wetLabel, tiltLabel, distanceLabel, outputLabel;

    juce::Slider wetSlider, tiltSlider, distanceSlider, outputSlider;

    juce::ToggleButton useIRButtton;

    std::unique_ptr<SliderAttachment> wetAttachment, tiltAttachment, distanceAttachment, outputAttachment;
    std::unique_ptr<ButtonAttachment> useIRAttachment;

    std::unique_ptr<juce::FileChooser> fc;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginEditor)
};
