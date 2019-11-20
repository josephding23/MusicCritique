(
TempoClock.default.tempo = 84/60;

p = Pbind(
	\scale, #[0, 2, 3, 5, 7, 8, 10],  //natural minor mode
	\root, 2,
	\degree, Pseq(#[rest, 4, 3, 4, 2, 4, 1, 4, 0, 4, -0.9, 4, 0, 4, 1, 4, 2, 4,
		-3, 4, -1.9, 4, -0.9, 4, 0, 4, -0.9, 4, 0, 4, 1, 4, 2], 1),
	//Subtracting 0.1 from an integer scale degree flattens the note by a semitone; adding 0.1 raises by a semitone.
	\dur, 0.25
	).play;
)
//Brute force, listing all the scale degrees mechanically in order.

(
p = Pbind(
	\scale, #[0, 2, 3, 5, 7, 8, 10],
	\root, 2,
	\degree, Place([#[rest, 3, 2, 1, 0, -0.9, 0, 1, 2, 3, -1.9, -0.9, 0, -0.9, 0, 1, 2],
		(4 ! 16) ++ \rest], 17),
	\dur, 0.25
).play;
)

(
p = Pbind(
	\scale, #[0, 2, 3, 5, 7, 8, 10],
	\root, 2,
	\degree, Ppatlace([Pseq(#[rest, 3, 2, 1, 0, -0.9, 0, 1, 2, -3, -1.9, -0.9, 0, -0.9, 0, 1, 2], 1), Pn(4, 16)], inf),
	\dur, 0.25
).play;
)

(
p = Pbind(
	\degree, Pseries(7, Pwhite(1, 3, inf) * Prand(#[-1, 1], inf), inf).fold(0, 14) + Prand(#[[0, -2, -4], [0, -3, -5], [0, -2, -5], [0, -1, -4]], inf),
	\dur, Pwrand(#[1, 0.5], #[0.8, 0.2], inf)
).play;
)
//When an array is sent to a Synth node, the pattern expands the event to produce multiple synth nodes instead of one.

(
b = Buffer.read(s, Platform.resourceDir +/+ "sounds/a11wlk01.wav");

SynthDef(\stretchFragments, { |out, bufnum, start, time = 1, stretch = 1, amp = 1, attack = 0.01, decay = 0.05|
	var sig = PlayBuf.ar(1, bufnum, rate: stretch.reciprocal, startPos: start), eg;
	sig = PitchShift.ar(sig, pitchRatio: stretch);
	eg = EnvGen.kr(Env.linen(attack, time, decay), sig.abs > 0, doneAction: Done.freeSelf);
	Out.ar(out, (sig * eg) ! 2)
}).add;
) //Play a SynthDef that has no frequency argument.

(
TempoClock.default.tempo = 1;

p = Pbind(
	\instrument, \stretchFragments,
	\bufnum, b,
	\start, Pwhite(0, (b.numFrames * 0.7).asInteger, inf),
	\delta, Pexprand(0.2, 1.5, inf),
	\time, Pkey(\delta),
	\stretch, Pexprand(1.0, 4.0, inf),
	\amp, 0.5,
	\attack, 0.1,
	\decay, 0.2
).play;
)