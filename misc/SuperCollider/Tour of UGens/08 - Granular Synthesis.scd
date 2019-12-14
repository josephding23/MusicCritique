// Granular Synthesis

// TGrains - Triggers generate grains from a buffer. Each grain has a Hanning envelope
// (sin2(x) for x from 0 to π)
// and can be panned over multichannel output
/*
arguments:
(1) numChannels,
(2) trigger: At each trigger, the following arguments are sampled and used as the arguments of a new grain. A trigger occurs when a signal changes from non-positive to a positive value
(3) bufnum:  The index of the buffer to use. It must be one channel(mono) buffer.
(4) rate: 1.0 is normal, 2.0 is one octave up, 0.5 is one octave down, -1.0 is backward normal rate...
(5) centerPos: The position in the buffer in seconds at which the grain envelope will reach maximum amplitude
(6) dur: Duration of the grain in seconds.
(7) pan: determines where to pan the output (1: ignored, 2: similar to Pan2, >2: same as PanAZ)
(8) amp: Amplitude of the grain
(9) interp: 1, 2, or 4. Determines whether the grain uses no interpolation, linear interpolation, or cubic interpolation
*/

// read sound
b = Buffer.read(s, Platform.resourceDir +/+ "sounds/a11wlk01.wav");

(
{
	var trate, dur;
	trate = MouseY.kr(2, 200, 1);
	dur =  4 / trate;
	TGrains.ar(2, Impulse.ar(trate), b, 1, MouseX.kr(0, BufDur.kr(b)), dur, 0, 0.1, 2);
}.scope(2, zoom: 4);
)

(
{
	var trate, dur, clk, pos, pan;
	trate = MouseY.kr(8, 120, 1);
	dur = 12 / trate;
	clk = Impulse.kr(trate);
	pos = MouseX.kr(0, BufDur.kr(b)) + TRand.kr(0, 0.01, clk);
	pan = WhiteNoise.kr(0.6);
	TGrains.ar(2, clk, b, 1, pos, dur, pan, 0.1);
}.scope(2, zoom: 4);
)

// 4 channels
(
{
	var trate, dur, clk, pos, pan;
	trate = MouseY.kr(8, 120, 1);
	dur = 12 / trate;
	clk = Impulse.kr(trate);
	pos = MouseX.kr(0, BufDur.kr(b)) + TRand.kr(0, 0.01, clk);
	pan = WhiteNoise.kr(0.6);
	TGrains.ar(4, clk, b, 1, pos, dur, pan, 0.1);
}.scope(4, zoom: 4);
)

(
{
	var trate, dur, clk, pos, pan;
	trate = MouseY.kr(8, 120, 1);
	dur = 4 / trate;
	clk = Dust.kr(trate);
	pos = MouseX.kr(0, BufDur.kr(b)) + TRand.kr(0, 0.01, clk);
	pan = WhiteNoise.kr(0.6);
	TGrains.ar(2, clk, b, 1, pos, dur, pan, 0.1);
}.scope(2, zoom: 4);
)

(
{
	var trate, dur, clk, pos, pan;
	trate = LinExp.kr(LFTri.kr(MouseY.kr(0.1, 2, 1)), -1, 1, 8, 120);
	dur = 12 / trate;
	clk = Impulse.kr(trate);
	pos = MouseX.kr(0, BufDur.kr(b));
	pan = WhiteNoise.kr(0.6);
	TGrains.ar(2, clk, b, 1, pos, dur, pan, 0.1);
}.scope(2, zoom: 4);
)

(
{
	var trate, dur, clk, pos, pan;
	trate = 12;
	dur = MouseY.kr(0.2, 24, 1) / trate;
	clk = Impulse.kr(trate);
	pos = MouseX.kr(0, BufDur.kr(b)) + TRand.kr(0, 0.01, clk);
	pan = WhiteNoise.kr(0.6);
	TGrains.ar(2, clk, b, 1, pos, dur, pan, 0.1);
}.scope(2, zoom: 4);
)

(
{
	var trate, dur, clk, pos, pan;
	trate = 100;
	dur = 8 / trate;
	clk = Impulse.kr(trate);
	pos = MouseX.kr(BrownNoise.kr(0.001));
	pan = WhiteNoise.kr(0.6);
	TGrains.ar(2, clk, b, 1, pos, dur, pan, 0.1);
}.scope(2, zoom: 4);
)

(
{
	var trate, dur;
	trate = MouseY.kr(2, 120, 1);
	dur = 1.2 / trate;
	TGrains.ar(2, Impulse.ar(trate), b, (1.2 ** WhiteNoise.kr(3).round(1)), MouseX.kr(0, BufDur.kr(b)), dur, WhiteNoise.kr(0.6), 0.1);
}.scope(2, zoom: 4);
)

b.free;


// GrainSin: granular synthesis with sine tones
/*
arguments:
(1) numChannels,
(2) trigger: a kr or ar trigger to start a new grain,
(3) dur: size of the grain,
(4) freq: the input to granulate,
(5) pan: determines where to pan the output,
(6) envbufnum: the buffer number containing a signal to use for the grain envelope. -1 uses a built-in Hann envelope,
(7) maxGrains: the maximum number,
(8) mul, (9) add
*/

(
{
	// using default window
	var trigrate, winsize, trig;
	trigrate = MouseX.kr(2, 120);
	winsize = trigrate.reciprocal;
	trig = Impulse.ar(trigrate);
	GrainSin.ar(2, trig, winsize, TRand.ar(440.0, 880.0, trig), LFNoise1.kr(0.2), -1, 512, 0.2);
}.scope(2, zoom: 4);
)

b = Buffer.sendCollection(s, Env([0, 1, 0], [0.5, 0.5], [8, -8]).discretize, 1);

(
{
	var trigrate, winsize, trig;
	trigrate = MouseX.kr(2, 120);
	winsize = trigrate.reciprocal;
	trig = Impulse.ar(trigrate);
	GrainSin.ar(2, trig, winsize, TRand.ar(440.0, 880.0, trig), LFNoise1.kr(0.2), b, 512, 0.2);
}.scope(2, zoom: 4);
)