// Periodic Sources: Oscillators

// LF- "Low Frequency" Unit Generators
// has geometric waveforms, not band limited. will cause aliasing at higher frequencies.
// arguments: frequency, phase, mul, add
{ LFPar.ar(LFPar.kr(LFPar.kr(0.2, 0.8, 10), 0, 400, 800), 0, 0.1) }.scope(1, zoom: 4);
{ LFPar.ar(LFPar.kr(0.2, 0, 400, 800), 0, 0.1)}.scope(1, zoom: 4);
{ LFPar.ar(800, 0, 0.1)}.scope(1, zoom: 4);
{ LFPar.ar(XLine.kr(100, 15000, 6), 0, 0.1)}.scope(1, zoom: 4);
// parabolic approximation of sine

{ LFCub.ar(LFCub.kr(LFCub.kr(0.2, 0.8, 10), 0, 400, 800), 0, 0.1)}.scope(1, zoom: 4);
{ LFCub.ar(LFCub.kr(0.2, 0, 400, 800), 0, 0.1)}.scope(1, zoom: 4);
{ LFCub.ar(800, 0, 0.1)}.scope(1, zoom: 4);
{ LFCub.ar(XLine.kr(100, 15000, 6), 0, 0.1)}.scope(1, zoom: 4);
// cubic approximation of sine, smoother than LFPar

{ LFTri.ar(LFTri.kr(LFTri.kr(0.2, 0.8, 10), 0, 400, 800), 0, 0.1)}.scope(1, zoom: 4);
{ LFTri.ar(LFTri.kr(0.2, 0, 400, 800), 0, 0.1)}.scope(1, zoom: 4);
{ LFTri.ar(800, 0, 0.1)}.scope(1, zoom: 4);
{ LFTri.ar(XLine.kr(100, 15000, 6), 0, 0.1)}.scope(1, zoom: 4);
// a non-band-limited triangle oscillator. Output ranges from -1 to +1.

{ LFSaw.ar(LFSaw.kr(LFSaw.kr(0.2, 0.8, 10), 0, 400, 800), 0, 0.1)}.scope(1, zoom: 4);
{ LFSaw.ar(LFSaw.kr(0.2, 0, 400, 800), 0, 0.1)}.scope(1, zoom: 4);
{ LFSaw.ar(100, 0, 0.1)}.scope(1, zoom: 4);
{ LFSaw.ar(XLine.kr(100, 15000, 6), 0, 0.1)}.scope(1, zoom: 4);
// a non-band-limited sawtooth oscillator. Output ranges from -1 to +1.

{ Impulse.ar(LFTri.kr(LFTri.kr(0.2, 0.8, 10), 0, 400, 800), 0, 0.1)}.scope(1, zoom: 4);
{ Impulse.ar(LFTri.kr(0.2, 0, 400, 800), 0, 0.1)}.scope(1, zoom: 4);
{ Impulse.ar(100, 0, 0.1)}.scope(1, zoom: 4);
{ Impulse.ar(XLine.kr(100, 15000, 6), 0, 0.1)}.scope(1, zoom: 4);
// outputs non-bandlimited single sample impulses.

// arguments: frequency, phase, width, mul, add
{ LFPulse.ar(LFPulse.kr(LFPulse.kr(0.2, 0, 0.5, 8, 10), 0, 0.5, 400, 800), 0, 0.5, 0.1)}.scope(1, zoom: 4);
{ LFPulse.ar(LFPulse.kr(3, 0, 0.3, 200, 200), 0, 0.2, 0.1)}.scope(1, zoom: 4);
{ LFPulse.ar(XLine.kr(100, 15000, 6), 0, 0.5, 0.1)}.scope(1, zoom: 4);
// a non-band-limited pulse oscillator. Outputs a high value of one and a low value of zero.
{ LFPulse.ar(100, 0, MouseY.kr(0, 1), 0.1)}.scope(1, zoom: 4);
{ LFPulse.ar(100, 0, LFTri.kr(0.2, 0, 0, 0.5, 0.5), 0.1)}.scope(1, zoom: 4);
// pulse width modulation

{ VarSaw.ar(VarSaw.kr(VarSaw.kr(0.2, 0, 0, 2, 8, 10), 0, 0.2, 400, 800),0, 0.2, 0.1)}.scope(1, zoom: 4);
{ VarSaw.ar(VarSaw.kr(0.2, 0 , 0.2, 400, 800), 0, 0.2, 0.1) }.scope(1, zoom: 4);
{ VarSaw.ar(XLine.kr(100, 15000, 6), 0, 0.2, 0.1)}.scope(1, zoom: 4);
// sawtooth-triangle oscillator with variable duty
{ VarSaw.ar(100, 0, MouseY.kr(0, 1), 0.1)}.scope(1, zoom: 4);
{ VarSaw.ar(100, 0, LFTri.kr(0.2, 0, 0.5, 0.5), 0.1)}.scope(1, zoom: 4);
// pulse width modulation