// Delays and Buffer UGens
// DelayN, DelayL, DelayC - simple delays
/*
N - no interpolation
L - linear interpolation
C - cubic interpolation
*/
// arguments: in, maximum delay time, current delay time, mul, add
(
{
	z = Decay.ar(Dust.ar(1, 0.5), 0.3, WhiteNoise.ar);
	DelayN.ar(z, 0.1, 0.1, 1, z); // input is mixed with delay via the add input
}.scope(1, zoom: 4)
)

(
{
	z = Decay.ar(Impulse.ar(2, 0, 0.4), 0.3, WhiteNoise.ar);
	DelayL.ar(z, 0.3, MouseX.kr(0, 0.3), 1, z);
}.scope(1, zoom: 4)
)


// feedback delays
// arguments: in, maximum delay time, current delay time, echo decay time, mul, add

// used as an echo
{ CombN.ar(Decay.ar(Dust.ar(1, 0.5), 0.2,  WhiteNoise.ar), 0.2, 0.2, 3) }.scope(1, zoom: 4);

// Comb used as a resonator. The resonant fundamental is equal to reciprocal of the delay time.
{ CombN.ar(WhiteNoise.ar(0.02), 0.01, XLine.kr(0.0001, 0.01, 20), 0.2) }.scope(1);
{ CombL.ar(WhiteNoise.ar(0.02), 0.01, XLine.kr(0.0001, 0.01, 20), 0.2) }.scope(1);
{ CombC.ar(WhiteNoise.ar(0.02), 0.01, XLine.kr(0.0001, 0.01, 20), 0.2) }.scope(1);

// with negative feedback
{ CombN.ar(WhiteNoise.ar(0.02), 0.01, XLine.kr(0.0001, 0.01, 20), -0.2) }.scope(1);
{ CombL.ar(WhiteNoise.ar(0.02), 0.01, XLine.kr(0.0001, 0.01, 20), -0.2) }.scope(1);
{ CombC.ar(WhiteNoise.ar(0.02), 0.01, XLine.kr(0.0001, 0.01, 20), -0.2) }.scope(1);


{ CombC.ar(Decay.ar(Dust.ar(1, 0.1), 0.2, WhiteNoise.ar), 1/100, 1/100, 3) }.scope(1);
{ CombC.ar(Decay.ar(Dust.ar(1, 0.1), 0.2, WhiteNoise.ar), 1/200, 1/200, 3) }.scope(1);
{ CombC.ar(Decay.ar(Dust.ar(1, 0.1), 0.2, WhiteNoise.ar), 1/300, 1/300, 3) }.scope(1);
{ CombC.ar(Decay.ar(Dust.ar(1, 0.1), 0.2, WhiteNoise.ar), 1/400, 1/400, 3) }.scope(1);


// allpass delay
// in, maximum delay time, current delay time, echo decay time, mul, add
(
{
	var z;
	z = Decay.ar(Dust.ar(1, 0.5), 0.1, WhiteNoise.ar);
	8.do { z = AllpassL.ar(z, 0.04, 0.04.rand, 2) };
	z
}.scope(1);
)
// A Schroeder allpass filter is given by the difference equations
/*
s(t) = x(t) + k * s(t - D)
y(t) = -k * s(t) + s(t - D)
where x(t) is the input signal, y(t) is the output signal, D is the delay time, and k is the allpass coefficient.
*/
//In this UGen, k is computed as k == 0.001 ** (delay / decay.abs) * decay.sign (0.001 is -60 dBFS).

// buffer playback
// arguments: numChannels, buffer number, rate, trigger, start pos, loop


// read sound
b = Buffer.read(s, Platform.resourceDir +/+ "sounds/a11wlk01.wav");

{ SinOsc.ar(800 + (700 * PlayBuf.ar(1,b, BufRateScale.kr(b),  loop:1)),0,0.3) }.scope(1);

// loop is true
{ PlayBuf.ar(1, b, BufRateScale.kr(b), loop: 1) }.scope(1);

(
{
	var trig;
	trig = Impulse.kr(2.0);
	PlayBuf.ar(1, b, BufRateScale.kr(b), trig, 0, 0)
}.scope(1);
)
// trigger one shot on each pulse

(
{
	var trig;
	trig = Impulse.kr(XLine.kr(0.1, 100, 30));
	PlayBuf.ar(1, b, BufRateScale.kr(b), trig, 5000, 0);
}.scope(1);
)
// trigger one shot on each pulse

(
{
	var trig;
	trig = Impulse.kr(MouseY.kr(0.5, 200, 1));
	PlayBuf.ar(1, b, BufRateScale.kr(b), trig,  MouseX.kr(0, BufFrames.kr(b)), 1)
}.scope(1);
)
// mouse control of trigger rate and startpos

(
{
	var rate;
	rate = XLine.kr(0.1, 100, 60);
	PlayBuf.ar(1, b, BufRateScale.kr(b) * rate, 1, 0, 1);
}.scope(1);
)
// accelerating pitch

(
{
	var rate;
	rate = FSinOsc.kr(XLine.kr(0.2, 8, 30), 0, 3, 0.6);
	PlayBuf.ar(1, b, BufRateScale.kr(b) * rate, 1, 0, 1);
}.scope(1);
)
// sine wave control of playback rate, negative rate plays backwards

(
{
	var rate;
	rate = LFNoise2.kr(XLine.kr(1, 20, 60), 2);
	PlayBuf.ar(1, b, BufRateScale.kr(b) * rate, 1, 0, 1);
}.scope(1);
)
// zig zag around sound

//free sound
b.free;