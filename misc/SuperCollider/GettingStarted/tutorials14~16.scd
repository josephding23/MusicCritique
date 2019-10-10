//14 Scheduling Events

SystemClock.sched(5, {"hello".postln});

(
var timeNow = TempoClock.default.beats;
"Time is now: ".post; timeNow.postln;
"Scheduling for: ".post; (timeNow + 5).postln;
TempoClock.default.schedAbs(timeNow + 5, {"Time is later: ".post; thisThread.clock.beats.postln; nil});
)

(
var timeNow;
TempoClock.default.tempo = 2;
timeNow = TempoClock.default.beats;
"Time is now: ".post; timeNow.postln;
"Scheduling for: ".post; (timeNow + 5).postln;
TempoClock.default.schedAbs(timeNow + 5, {"Time is later: ".post; thisThread.clock.beats.postln; nil});
)

TempoClock.default.sched(1, {rrand(1, 3).postln;});

TempoClock.default.sched(1, {rrand(1, 3).postln; nil});

//15 Sequencing with Routines and Tasks

r = Routine({
    "abcde".yield;
    "fghij".yield;
    "klmno".yield;
    "pqrst".yield;
    "uvwxy".yield;
	"z{|}~".yield;
});

r.next;
6.do({r.next.postln});

r = Routine({
	var delta;
	loop {
		delta = rrand(1, 3) * 0.5;
		"Will wait ".post; delta.postln;
		delta.yield;
	}
});

r.next;

TempoClock.default.sched(0, r);

r.stop;

(
SynthDef(\singrain, { |freq = 440, amp = 0.2, sustain = 1|
	var sig;
	sig = SinOsc.ar(freq, 0, amp) * EnvGen.kr(Env.perc(0.01, sustain), doneAction: Done.freeSelf);
	Out.ar(0, sig ! 2);
}).add;

r = Routine({
	var delta;
	loop {
		delta = rrand(1, 3) * 0.5;
		Synth(\singrain, [freq: exprand(200, 800), amp: rrand(0.1, 0.5), sustain: delta * 0.8]);
		delta.yield;
	}
});
)

r.play;
r.stop;

(
t = Task({
	loop {
		[60, 62, 64, 65, 67, 69, 71, 72].do({ |midi|
			Synth(\singrain, [freq: midi.midicps, amp: 0.2, sustain: 0.1]);
			0.125.wait;
		})
	}
}).play;
)

t.stop;
t.play;
t.stop;

(
f = {
	Task({
		loop {
			[60, 62, 64, 65, 67, 69, 71, 72].do({|midi|
				Synth(\singrain, [freq: midi.midicps, amp: 0.2, sustain: 0.1]);
				0.25.wait;
			});
		}
	});
}
)

t = f.value.play(quant: 4)
u = f.value.play(quant: [4, 0.5])
t.stop; u.stop;

(
var midi, dur;
midi = Routine({
	[60, 72, 71, 67, 69, 71, 72, 60, 69, 67].do({ |midi| midi.yield});
});

dur = Routine({
	[2, 2, 1, 0.5, 0.5, 1, 1, 2, 2, 3].do({ |dur| dur.yield});
});

SynthDef(\smooth, { |freq = 440, sustain = 1, amp = 0.5|
	var sig;
	sig = SinOsc.ar(freq, 0, amp) * EnvGen.kr(Env.linen(0.05, sustain, 0.1), doneAction: Done.freeSelf);
	Out.ar(0, sig ! 2)
}).add;

r = Task({
	var delta;
	while {
		delta = dur.next;
		delta.notNil;
	} {
		Synth(\smooth, [freq: midi.next.midicps, sustain: delta]);
		delta.yield;
	}
}).play(quant: TempoClock.default.beats + 1.0);
)

(
t = Task({
	loop {
		s.makeBundle(s.latency, {
			Synth(\singrain, [freq: exprand(400, 1200), sustain:0.08]);
		});
		0.1.wait;
	}
}).play;
)

//16 Sequencing with Patterns

r = Routine({
	[60, 72, 71, 67, 69, 71, 72, 60, 69, 67].do({ |midi| midi.yield});
});
while { (m = r.next).notNil} {m.postln};

p = Pseq([60, 72, 71, 67, 69, 71, 72, 60, 69, 67], 1);
r = p.asStream;
while { (m = r.next).notNil} {m.postln};

(
var midi, dur;
midi = Pseq([60, 72, 71, 67, 69, 71, 72, 60, 69, 67], 1).asStream;
dur = Pseq([2, 2, 1, 0.5, 0.5, 1, 1, 2, 2, 3], 1).asStream;

SynthDef(\smooth, {|out, freq = 440, sustain = 1, amp = 0.5|
	var sig;
	sig = SinOsc.ar(freq, 0, amp) * EnvGen.kr(Env.linen(0.05, sustain, 0.1), doneAction: Done.freeSelf);
	Out.ar(out, sig ! 2)
}).add;

r = Task({
	var delta;
	while {
		delta = dur.next;
		delta.notNil
	} {
		Synth(\smooth, [freq: midi.next.midicps, sustain: delta]);
		delta.yield;
	}
}).play(quant: TempoClock.default.beats + 1.0);
)

p = Pwhite(0.0, Pseries(0.01, 0.01, inf), 100).asStream;
p.all.plot;

p = Pn(Pshuf([1, 2, 3, 4, 5], 1), inf).asStream;
p.nextN(15);

(
SynthDef(\smooth, { |out, freq = 440, sustain = 1, amp = 0.5|
	var sig;
	sig = SinOsc.ar(freq, 0, amp) * EnvGen.kr(Env.linen(0.05, sustain, 0.1), doneAction: Done.freeSelf);
	Out.ar(out, sig ! 2)
	}).add;
)

(
p = Pbind(
	\instrument, \smooth,
	\midinote, Pseq([60, 72, 71, 67, 69, 71, 72, 60, 69, 67], 1),
	\dur, Pseq([2, 2, 1, 0.5, 0.5, 1, 1, 2, 2, 3], 1)
).play;
)

(
SynthDef(\bass, { |out, freq = 440, gate = 1, amp = 0.5, slideTime = 0.17, ffreq = 1100, width = 0.15, detune = 1.005, preamp = 4|
	var sig,
	env = Env.adsr(0.01, 0.3, 0.4, 0.1);
	freq = Lag.kr(freq, slideTime);
	sig = Mix(VarSaw.ar([freq, freq * detune], 0, width, preamp)).distort * amp * EnvGen.kr(env, gate, doneAction: Done.freeSelf);
	sig = LPF.ar(sig, ffreq);
	Out.ar(out, sig ! 2)
}).add;

TempoClock.default.tempo = 132/60;

p = Pxrand([
	Pbind(
		\instrument, \bass,
		\midinote, 36,
		\dur, Pseg([0.75, 0.25, 0.25, 0.25, 0.5], 1),
		\legato, Pseg([0.9, 0.3, 0.3, 0.3, 0.3], 1),
		\amp, 0.5, \detune, 1.005
	),

	Pmono(\bass,
		\midinote, Pseg([36, 48, 36], 1),
		\dur, Pseg([0.25, 0.25, 0.5], 1),
		\amp, 0.5, \detune,1.005
	),
	Pmono(\bass,
		\midinote, Pseg([36, 42, 41, 33], 1),
		\dur, Pseg([0.25, 0.25, 0.25, 0.75], 1),
		\amp, 0.5, \detune, 1.005
	),
	Pmono(\bass,
		\midinote, Pseg([36, 39, 36, 42], 1),
		\dur, Pseg([0.25, 0.5, 0.25, 0.5], 1),
		\amp, 0.5, \detunr, 1.005
	)
], inf).play(quant: 1);
)

(
SynthDef(\kik, { |out, preamp = 1, amp = 1|
	var freq = EnvGen.kr(Env([400, 66], [0.08],-3)),
	sig = SinOsc.ar(freq, 0.5pi, preamp).distort * amp * EnvGen.kr(Env([0, 1, 0.8, 0], [0.01, 0.1, 0.2]), doneAction: Done.freeSelf);
	Out.ar(out, sig ! 2);
}).add;

k = Pbind(\instrument, \kik, \delta, 1, \preamp, 4.5, \amp, 0.32).play(quant: 1);
)

p.stop;
k.stop;

