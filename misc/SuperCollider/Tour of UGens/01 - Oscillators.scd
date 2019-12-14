// Periodic Sources: Oscillators

// LF- "Low Frequency" Unit Generators
// has geometric waveforms, not band limited. will cause aliasing at higher frequencies.
// arguments: frequency, phase, mul, add
{ LFPar.ar(LFPar.kr(LFPar.kr(0.2, 0.8, 10), 0, 400, 800), 0, 0.1) }.scope(1, zoom: 4);
{ LFPar.ar(LFPar.kr(0.2, 0, 400, 800), 0, 0.1)}.scope(1, zoom: 4);
{ LFPar.ar(8, 0, 1)}.scope(1, zoom: 4);
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

// arguments: syncFreq, sawFreq, mul, add
{ SyncSaw.ar(100, MouseX.kr(100, 1000), 0.1)}.scope(1, zoom: 4);
{ SyncSaw.ar(100, Line.kr(100, 800, 12), 0.1)}.scope(1, zoom: 4);
// a sawtooth wave that is hard synched to a fundamental pitch. This produces an effect similar to moving formants or pulse width modulation. The sawtooth oscillator has its phase reset when the sync oscillator completes a cycle. This is not a band limited waveform, so it may alias.


// Band Limited Oscillators
// arguments: frequency, phase, mul, add
{ SinOsc.ar(SinOsc.kr(0.2, 0, 400, 800), 0, 0.1) }.scope(1);
{ SinOsc.ar(800, 0, 0.1) }.scope(1, zoom: 4);

{ SinOscFB.ar(440, MouseX.kr(0, 4)) * 0.1 }.scope(1);
// A sine oscillator that has phase modulation feedback; its output plugs back into the phase input

{ FSinOsc.ar(800, 0, 0.1)}.scope(1, zoom: 4);
{ FSinOsc.ar(FSinOsc.kr(FSinOsc.kr(0.2, 0.8, 10), 0, 400, 800), 0, 0.1)}.scope(1, zoom: 4);
// Very fast sine wave generator implemented using a ringing filter. This generates a much cleaner sine wave than a table lookup oscillator and is a lot faster. However, the amplitude of the wave will vary with frequency.

{ Blip.ar(XLine.kr(20000, 200, 6), 100, 0.2)}.scope(1, zoom: 4);
{ Blip.ar(200, Line.kr(1, 100, 20), 0.2) }.scope(1);
// Band Limited ImPulse generator. All harmonics have equal amplitude.

{ Saw.ar(XLine.kr(20000, 200, 6), 0.2)}.scope(1, zoom: 4);
{ Saw.ar(XLine.kr(100, 15000, 6), 0.2)}.scope(1, zoom: 4);
// band limited sawtooth wave generator

{ Pulse.ar(XLine.kr(40, 4000, 6), 0.1, 0.2)}.scope(1, zoom: 4);
{ Pulse.ar(200, SinOsc.kr(0.2).range(0.01, 0.99), 0.2)}.scope(1, zoom: 4);
{ Pulse.ar(200, Line.kr(0.01, 0.99, 8), 0.2) }.scope(1, zoom: 4);
// band limited pulse wave generatir with pulse width modulation

{ Klang.ar(`[[800, 1000, 1200], [0.3, 0.3, 0.3], [pi, pi, pi]], 1, 0) * 0.4}.scope(1);
{ Klang.ar(`[[800, 1000, 1200], nil, nil], 1, 0) * 0.25 }.scope(1);
{ Klang.ar(`[ Array.rand(12, 600.0, 1000.0), nil, nil ], 1, 0) * 0.05 }.scope(1);
{ Klang.ar(`[ {exprand(400, 2000)}.dup(16), nil, nil], 1, 0) * 0.04}.scope(1);
// Klang is a bank of fixed frequency sine oscillators


// Table Oscillators
(
b = Buffer.alloc(s, 2048, 1, bufnum: 80);
b.sine1(1.0/(1..6), true, true, true);
)

{ Osc.ar(80, 100, 0, 0.1)}.scope(1, zoom: 4);

b.sine1(1.0/(1..6), true, true, true);

{ COsc.ar(80, 100, 1, 0.1) }.scope(1, zoom:4);

b.sine1(1.0/(1..12));
b.sine1(1.0/(1..24));
b.sine1(1.0/(1..32));
b.sine1([1.0/(1,3..12), 0].flop.flat.postln);
b.sine1([1.0/(1,3..32).squared, 0].flop.flat.postln);
b.sine1((1.dup(4) ++ 0.dup(8)).scramble.postln);
b.sine1((1.dup(4) ++ 0.dup(8)).scramble.postln);
b.sine1((1.dup(4) ++ 0.dup(8)).scramble.postln);
b.sine1((1.dup(4) ++ 0.dup(8)).scramble.postln);
b.sine1({1.0.rand2.cubed}.dup(8).round(1e-3).postln);
b.sine1({1.0.rand2.cubed}.dup(12).round(1e-3).postln);
b.sine1({1.0.rand2.cubed}.dup(16).round(1e-3).postln);
b.sine1({1.0.rand2.cubed}.dup(24).round(1e-3).postln);

(
// allocate tables 80 to 87
8.do {|i| s.sendMsg(\b_alloc, 80+i, 1024);};
)

(
// fill tables 80 to 87
8.do({ |i|
	var n, a;
	// generate array of harmonic amplitudes
	n = (i+1)**2; // num harmonics for each table: [1, 4, 9, 16, 25, 36, 49, 64]
	a = {|j| ((n-j)/n).squared}.dup(n);
	// fill table
	s.listSendMsg([\b_gen, 80+i, \sine1, 7] ++ a);
})
)

{ VOsc.ar(MouseX.kr(80, 87), 120, 0, 0.3)}.scope(1, zoom: 4);

(  // allocate and fill tables 80 to 87
8.do({|i|
	// generate array of harmonic amplitudes
	a = {1.0.rand2.cubed}.dup((i+1) * 4);
	// fill table
	s.listSendMsg([\b_gen, 80+i, \sine1, 7] ++ a);
});
)