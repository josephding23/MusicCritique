(
TempoClock.default.tempo = 1;
p = Pbind(
	\degree, Pseq([
		Pseries({rrand(0, 7)}, 1, {rrand(4, 8)}),
		Pseries({rrand(0, 7)}, 1, {rrand(4, 8)}),
		Pseries({rrand(7, 14)}, -1, {rrand(4, 8)})
	], inf),
	\dur, 0.125
).play;
).
//Scale segments,  in the sequence: up, up, down

(
SynthDef(\bass, { |out, freq = 440, gate = 1, amp = 0.5, slideTime = 0.17, ffreq = 1100, width = 0.15, detune = 1.005, preamp = 4|
	var sig,
	env = Env.adsr(0.01, 0.3, 0.4, 0.1);
	sig = Mix(VarSaw.ar([freq, freq * detune], 0, width, preamp)).distort * amp * EnvGen.kr(env, gate, doneAction: Done.freeSelf);
	sig = LPF.ar(sig, ffreq);
	Out.ar(out, sig ! 2)
}).add;
)

(
TempoClock.default.tempo = 132/60;
p = Pxrand([
	Pbind(    // repeated notes
		\instrument, \bass,
		\midinote, 36,
		\dur, Pseq([0.75, 0.25, 0.25, 0.25, 0.5], 1),
		\legato, Pseq([0.9, 0.3, 0.3, 0.3, 0.3], 1),
		\amp, 0.5, \detune, 1.005
	),
	Pmono(\bass,  // octave jump
		\midinote, Pseq([36, 48, 36], 1),
		\dur, Pseq([0.25, 0.25, 0.5], 1),
		\amp, 0.5, \detune, 1.005
	),
	Pmono(\bass,  // tritone jump
		\midinote, Pseq([36, 42, 41, 33], 1),
		\dur, Pseq([0.25, 0.25, 0.25, 0.75], 1),
		\amp, 0.5, \detune, 1.005
	),
	Pmono(\bass,  // diminshed triad
		\midinote, Pseq([36, 39, 36, 42], 1),
		\dur, Pseq([0.25, 0.5, 0.25, 0.5], 1),
		\amp, 0.5, \detune, 1.005
	)
], inf).play(quant: 1);
)

(
var n = 10,
scaleSegments = Array.fill(n, { |i|
	if(i.odd) {
		Pseries(11, -1, rrand(5, 10))
	} {
		Pseries(rrand(-4, 4), 1, i+2)
	}
});
TempoClock.default.tempo = 1;
p = Pbind(
	\degree, Pswitch(scaleSegments, Pseq([Pwhite(0, n-1, 1).select(_.odd)], inf)),
	\dur, 0.125
).play;
)//Patterns can be chosen in arbitrary irder by index

(
~phrases = (
    repeated: Pbind(
        \instrument, \bass,
        \midinote, 36,
        \dur, Pseq([0.75, 0.25, 0.25, 0.25, 0.5], 1),
        \legato, Pseq([0.9, 0.3, 0.3, 0.3, 0.3], 1),
        \amp, 0.5, \detune, 1.005
    ),
    octave: Pmono(\bass,
        \midinote, Pseq([36, 48, 36], 1),
        \dur, Pseq([0.25, 0.25, 0.5], 1),
        \amp, 0.5, \detune, 1.005
    ),
    tritone: Pmono(\bass,
        \midinote, Pseq([36, 42, 41, 33], 1),
        \dur, Pseq([0.25, 0.25, 0.25, 0.75], 1),
        \amp, 0.5, \detune, 1.005
    ),
    dim: Pmono(\bass,
        \midinote, Pseq([36, 39, 36, 42], 1),
        \dur, Pseq([0.25, 0.5, 0.25, 0.5], 1),
        \amp, 0.5, \detune, 1.005
    )
);

TempoClock.default.tempo = 128/60;

// the higher level control pattern is really simple now
p = Psym(Pxrand(#[repeated, octave, tritone, dim], inf), ~phrases).play;
)
//Pysm can manage to make a library of sub-patterns, and call them up one at a time

(
TempoClock.default.tempo = 1;
p = Pbind(
	\degree, Pswitch1([Pwhite(7, 14, inf), Pwhite(-7, 0, inf), Pfunc{0.7.coin.binaryValue}]),
	\dur, 0.25
).play;
)
// random pitches in two distinct ranges, use a coin toss to decide which for this event