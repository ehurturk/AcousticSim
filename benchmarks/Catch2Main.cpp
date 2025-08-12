#include "juce_gui_basics/juce_gui_basics.h"
#include <catch2/catch_session.hpp>

int main (int argc, char* argv[])
{
    juce::ScopedJuceInitialiser_GUI gui;

    const int result = Catch::Session().run (argc, argv);

    return result;
}

#include "PluginEditor.h"
#include "catch2/benchmark/catch_benchmark_all.hpp"
#include "catch2/catch_test_macros.hpp"

#include "Benchmarks.cpp"
