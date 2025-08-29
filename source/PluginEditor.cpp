#include "PluginEditor.h"
#include <memory>

PluginEditor::PluginEditor (PluginProcessor& p, juce::AudioProcessorValueTreeState& apvts)
    : AudioProcessorEditor (&p), processorRef (p), apvtsRef (apvts)
{
    auto setup = [&] (juce::Slider& s, juce::Label& l, const juce::String& id) {
        l.setText (id, juce::dontSendNotification);
        addAndMakeVisible (l);
        addAndMakeVisible (s);
        s.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
        s.setTextBoxStyle (juce::Slider::TextBoxAbove, false, 60, 18);
    };

    setup (wetSlider, wetLabel, "Wet");
    setup (tiltSlider, tiltLabel, "Tilt");
    setup (distanceSlider, distanceLabel, "Mic Distance");
    setup (outputSlider, outputLabel, "Output");

    useIRButtton.setButtonText ("Load IR");
    addAndMakeVisible (useIRButtton);

    // load IR with async file chooser
    useIRButtton.onClick = [this]() {
        fc = std::make_unique<juce::FileChooser> ("Select an impulse response", juce::File::getSpecialLocation (juce::File::userDocumentsDirectory), "*.wav;*.aif;*.aiff");
        auto fcFlags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
        fc->launchAsync (fcFlags, [this] (const juce::FileChooser& chooser) {
            juce::File file (chooser.getResult());
            // processorRef.getAPVTS().getParameterAsValue ("irPath") = file.getFullPathName();
            if (file.existsAsFile())
            {
                processorRef.loadIRFile (file);
            }
        });
    };

    wetAttachment.reset (new SliderAttachment (apvtsRef, "wet", wetSlider));
    tiltAttachment.reset (new SliderAttachment (apvtsRef, "tilt", tiltSlider));
    distanceAttachment.reset (new SliderAttachment (apvtsRef, "distance", distanceSlider));
    outputAttachment.reset (new SliderAttachment (apvtsRef, "output", outputSlider));
    useIRAttachment.reset (new ButtonAttachment (apvts, "useIR", useIRButtton));

    useIRButtton.setClickingTogglesState (true);

    setSize (400, 300);
}

PluginEditor::~PluginEditor()
{
}

void PluginEditor::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));

    // auto area = getLocalBounds();
    // g.setColour (juce::Colours::white);
    // g.setFont (16.0f);
    // auto helloWorld = juce::String ("Hello from ") + PRODUCT_NAME_WITHOUT_VERSION + " v" VERSION + " running in " + CMAKE_BUILD_TYPE;
    // g.drawText (helloWorld, area.removeFromTop (150), juce::Justification::centred, false);
}

void PluginEditor::resized()
{
    // layout the positions of your child components here
    auto area = getLocalBounds().reduced (12);
    useIRButtton.setBounds (area.removeFromTop (28));
    auto row = area.removeFromTop (160);
    auto w = row.getWidth() / 4;

    wetSlider.setBounds (row.removeFromLeft (w).reduced (8));
    tiltSlider.setBounds (row.removeFromLeft (w).reduced (8));
    distanceSlider.setBounds (row.removeFromLeft (w).reduced (8));
    outputSlider.setBounds (row.removeFromLeft (w).reduced (8));

    // area.removeFromBottom (50);
    // inspectButton.setBounds (getLocalBounds().withSizeKeepingCentre (100, 50));
}
