// Aperiodic Sources: Noise

// LF "Low Frequency" Noise Generators
// arguments: frequency, mul, add
{ LFClipNoise.ar(MouseX.kr(200, 10000, 1), 0.125)}.scope(1);
{ LFClipNoise.ar(XLine.kr(1000, 10000, 10), 0.125) }.scope(1);
{ LFClipNoise.ar(1000, 0.125) }.scope(1);
// Randomly generates the values -1 or +1 at a rate given by the nearest integer division of the sample rate by the freq argument
{ LFNoise0.ar(MouseX.kr(200, 10000, 1), 0.25)}.scope(1);
{ LFNoise0.ar(XLine.kr(1000, 10000, 10), 0.25) }.scope(1);
{ LFNoise0.ar(1000, 0.25) }.scope(1);
// Generates random values at a rate given by the nearest integer division of the sample rate by the freq argument.
{ LFNoise1.ar(MouseX.kr(200, 10000, 1), 0.25)}.scope(1);
{ LFNoise1.ar(XLine.kr(1000, 10000, 10), 0.25) }.scope(1);
{ LFNoise1.ar(1000, 0.25) }.scope(1);
// Generates linearly interpolated random values at a rate given by the nearest integer division of the sample rate by the freq argument.
{ LFNoise2.ar(MouseX.kr(200, 10000, 1), 0.25)}.scope(1);
{ LFNoise2.ar(XLine.kr(1000, 10000, 10), 0.25) }.scope(1);
{ LFNoise2.ar(1000, 0.25) }.scope(1);
// Generates quadratically interpolated random values at a rate given by the nearest integer division of the sample rate by the freq argument.
{ LFPar.ar(LFClipNoise.kr(MouseX.kr(0.5, 64, 1), 200, 400), 0, 0.2) }.scope(1, zoom:8);
{ LFPar.ar(LFNoise0.kr(MouseX.kr(0.5, 64, 1), 200, 400), 0, 0.2) }.scope(1, zoom:8);
{ LFPar.ar(LFNoise1.kr(MouseX.kr(0.5, 64, 1), 200, 400), 0, 0.2) }.scope(1, zoom:8);
{ LFPar.ar(LFNoise2.kr(MouseX.kr(0.5, 64, 1), 200, 400), 0, 0.2) }.scope(1, zoom:8);

// Broad Spectrum Noise Generators
// arguments: mul, add
{ ClipNoise.ar(0.25)}.scope(1);
// Generates noise whose values are either -1 or 1. This produces the maximum energy for the least peak to peak amplitude.
{ WhiteNoise.ar(0.25)}.scope(1);
// Generates noise whose spectrum has equal power at all frequencies
{ PinkNoise.ar(0.25)}.scope(1);
// Generates noise whose spectrum falls off in power by 3 dB per octave, which gives equal power over the span of each octave
{ BrownNoise.ar(0.25)}.scope(1);
// Generates noise whose spectrum falls off in power by 6 dB per octave.
{ GrayNoise.ar(0.25)}.scope(1);
// Generates noise which results from flipping random bits in a word. This type of noise has a high RMS level relative to its peak to peak level. The spectrum is emphasized towards lower frequencies.

// Impulse Noise Generators
// arguments: density, mul, add
{ Dust.ar(2000, 0.25)}.scope(1);
// Generates random impulses from 0 to +1
{ Dust2.ar(2000, 0.25)}.scope(1);
// Generates random impulses from -1 to +1

// Chaotic Noise Generators
// arguments: chaosParam(from just below 1.0 to just above 2.0), mul, add
{ Crackle.ar(1.1, 0.5)}.scope(1);
