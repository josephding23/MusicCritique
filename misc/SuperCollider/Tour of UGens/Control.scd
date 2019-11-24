// Control

// Filters for Controls

// Decay -  triggered expotional decay
// arguments: in, decay time(60 dB decay time in seconds), mul, add
{ WhiteNoise.ar * Decay.ar(Impulse.ar(1), 0.9, 0.2) }.scope(1, zoom:  4);
{ WhiteNoise.ar * Decay.ar(Dust.ar(3), 0.9, 0.2) }.scope(1, zoom: 4);
{ SinOsc.ar(Decay.ar(Dust.ar(4), 0.5, 1000, 400), 0, 0.2) }.scope(1, zoom: 4);
plot({ Decay.ar(Impulse.ar(1), 0.01)});

// Decay2 - triggered expotional attack and expotional decay
// rounds off the attack by substracting one Decay from another.
// equivalent to: Decay.ar(in, decayTime) - Decay.ar(in, attackTime)
// arguments: trigger, attack time, decay time, mul, add
{ WhiteNoise.ar * Decay2.ar(Impulse.ar(1), 0.2, 0.9, 0.2) }.scope(1, zoom: 4);
{ WhiteNoise.ar * Decay2.ar(Dust.ar(3), 0.2, 0.9, 0.2) }.scope(1, zoom: 4);
plot({ Decay2.ar(Impulse.ar(1), 0.001, 0.01)});
plot({ Decay2.ar(Impulse.ar(1), 0.01, 0.001)});

// Lag
// This is essentially the same as OnePole except that instead of supplying the coefficient directly, it is calculated from a 60 dB lag time.
// arguments: trigger, lagTime, mul, add
{ SinOsc.ar(Lag.ar(LFPulse.ar(2, 0, 0.5, 800, 400), MouseX.kr(0, 0.5)), 0, 0.2) }.scope(1, zoom: 4);

// Integrator - leaky integrator
// Integrates an input signal with a leak. The formula implemented is:
// out(0) = in(0) + (coef * out(-1))
// arguments: in, coef, mul, add
{ SinOsc.ar(Integrator.ar(Dust2.ar(8), 0.99999, 200, 800), 0, 0.2) }.scope(1);


// Triggers

// Trig, Trig1 - timed duration gate
// arguments: trigger, duration
{ Trig.ar(Dust.ar(2), 0.2) * FSinOsc.ar(800, 0, 0.4) }.scope(1, zoom: 4);
// When a nonpositive to positive transitin occurs at the input, Trig outputs the level of the triggering input for the specified duration, otherwise it outputs zero.
{ Trig1.ar(Dust.ar(2), 0.2) * FSinOsc.ar(800, 0, 0.4) }.scope(1, zoom: 4);
// When a nonpositive to positive transition occurs at the input, Trig1 outputs 1 for the specified duration, otherwise outputs 0.

(
{
	var trig;
	trig = Dust.ar(2);
	[
		(Trig1.ar(trig, 0.05) * FSinOsc.ar(600, 0, 0.2)),
		(Trig1.ar(TDelay.ar(trig, 0.1), 0.05) * FSinOsc.ar(800, 0, 0.2))
		// TDelay - Delays a trigger by a given time.
		// arguments: trigger, delay time
	]
}.scope(2, zoom: 4);
)

// Latch - sample and hold
// arguments: in, trigger
{ Blip.ar(Latch.ar(WhiteNoise.ar, Impulse.ar(9)) * 400 + 500, 4, 0.2) }.play;
{ Blip.ar(Latch.ar(SinOsc.ar(0.3), Impulse.ar(9)) * 400 + 500, 4, 0.2) }.play;
// Holds input signal value when triggered. Latch will output 0 until it receives its first trigger.

// Gate - pass or hold
// arguments: in, trigger
{ Blip.ar(Gate.ar(LFNoise2.ar(40), LFPulse.ar(1)) * 400 + 500, 4, 0.2) }.scope(1, zoom: 4);
{ Gate.kr(WhiteNoise.kr(1, 0), LFPulse.kr(1.333, 0.5)) }.scope(zoom: 20);
// Allows input signal value to pass when gate is positive, otherwise holds last value

// PulseCount - count triggers
// arguments: trigger, reset(resets the counter to zero when triggered)
(
{
	SinOsc.ar(PulseCount.ar(Impulse.ar(10), Impulse.ar(0.4)) * 200, 0, 0.05)
}.scope(2, zoom: 4);
)
// Each trigger increments a counter which is output as a signal

// PulseDivider
// arguments: trigger, div(number of triggers to count before outputting an impulse), start(starting value for the trigger count)
(
{
	var p, a, b;
	p = Impulse.ar(8);
	a = SinOsc.ar(1200, 0, Decay2.ar(p, 0.005, 0.1));
	b = SinOsc.ar(600, 0, Decay2.ar(PulseDivider.ar(p, MouseX.kr(1, 8).round(1)), 0.005, 0.5));
	[a, b] * 0.4;
}.scope(2, zoom: 4);
)
// Outputs one impulse each time it receives a certain number of triggers at its input

// EnvGen - envelope generator
// envelope is specified using an instance of the Env class
{ EnvGen.kr(Env.perc, doneAction: Done.freeSelf) * SinOsc.ar(880, 0, 0.2) }.play;
{ EnvGen.kr(Env.perc(1, 0.005, 1, 4), doneAction: Done.freeSelf) * SinOsc.ar(880, 0, 0.2) }.play;

{ EnvGen.kr(Env.perc, Impulse.kr(2)) * SinOsc.ar(880, 0, 0.2) }.play;
{ EnvGen.kr(Env.perc, Dust.kr(3)) * SinOsc.ar(880, 0, 0.2) }.play;

// for sustaini envelopes a gate is required
z = { arg gate = 1; EnvGen.kr(Env.adsr, gate, doneAction: Done.freeSelf) * SinOsc.ar(880, 0, 0.2) }.play;

(
// randomly generated envelope
z = { arg gate = 1;
	var env, n = 32;
	env = Env(
		[0] ++ {1.0.rand.squared}.dup(n-1) ++ [0],
		{rrand(0.005, 0.2)}.dup(n),
		\lin, n-8, 8);
	EnvGen.kr(env, gate, doneAction: Done.freeSelf) * LFTri.ar(220, 0, 0.4)
}.scope(1, zoom: 4);
)


