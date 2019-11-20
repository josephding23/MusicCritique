(
b = Buffer.read(s, Platform.resourceDir +/+ "sounds/a11wlk01.wav");

SynthDef(\oneLoop, {|out, bufnum, start, time, amp, rate = 1|
	var sig = PlayBuf.ar(1, bufnum,
		rate: rate * BufRateScale.kr(bufnum),
		startPos: start, loop: 0
	),
	env = EnvGen.kr(Env.linen(0.01, time, 0.05, level: amp),
		doneAction: Done.freeSelf
	);
	Out.ar(out, (sig * env).dup);
}).add;

SynthDef(\bell, {|out, accent = 0, amp = 0.1, decayScale = 1|
	var exc = PinkNoise.ar(amp) * Decay2.kr(Impulse.kr(0), 0.01, 0.05),
	sig = Klank.ar(`[
		{ ExpRand(400, 1600)} ! 4,
		1 ! 4,
		{ ExpRand(0.1, 0.4) } ! 4
	], exc, freqscale: accent + 1, decayscale: decayScale);
	DetectSilence.ar(sig, doneAction: Done.freeSelf);
	Out.ar(out, sig.dup);
}).add;
)

(
var start = 0.404561, end = 3.185917,
beatsInLoop = 4,
originalTempo = beatsInLoop / (end - start);

TempoClock.tempo = originalTempo;

p = Ptpar([
	0, Pbind(
		\instrument, \oneLoop,
		\bufnum, b,
		\amp,  0.4,
		\start, start * b.sampleRate,
		\dur, beatsInLoop,
		\time, Pkey(\dur) / Pfunc {thisThread.clock.tempo},
		\rate, Pfunc { thisThread.clock.tempo  / originalTempo }
	),
	0.5, Pn(
		Pfindur(4,
			Pbind(
				\instrument, \bell,
				\accent, Pseq([2, Pn(0, inf)], 1),
				\amp, Pseq([0.3, Pn(0.1, inf)], 1),
				\decayScale, Pseq([6, Pn(1, inf)], 1),
				\dur, Pwrand(#[0.25, 0.5, 0.75, 1]， #[2, 3, 1, 1].normalizeSum, inf)
			)
		),
	inf),
	0.5, Pbind(
		\instrument, \bell,
		\accent, -0.6,
		\amp, 0.2,
		\decayScale, 0.1,
		\dur, 1
	)
], 1).play;
)

TempoClock.tempo = 104/60;