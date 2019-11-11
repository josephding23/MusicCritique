(
p = Ppatlace([Pwhite(1, 5, 5), Pgeom(10, 1.01, 10)], inf);
x = p.asStream;
x.all;
)

(
SynthDef(\help_sinegrain,
	{ arg out=0, freq=440, sustain=0.05;
		var env;
		env = EnvGen.kr(Env.perc(0.01, sustain, 0.2), doneAction: Done.freeSelf);
		Out.ar(out, SinOsc.ar(freq, 0, env))
	}
).add;
)

(
var c = Ppatlace([
	Pseq([0, 0, 0, 0, 8, 0, 8], inf),
	Pseries(1, 1, 32)
], inf) + 67;
x = c.asStream;

Routine({
	loop({
		Synth(\help_sinegrain, [\freq, x.next.midicps, \dur, 0.2]);
		0.17.wait;
	})
}).play;
)

(
c = Ppatlace([
	Pxrand([
		Pseq(#[0, -2, -3, -5, -7], 1), Pwhite(-12, 4, 3), Pshuf(#[0, -2, -3, -5, -7], 1)
	], inf),
	Pxrand([
		Pseq(#[0, 2, 4, 5, 7], 1), Pwhite(-4, 12, 3), Pshuf(#[0, 2, 4, 5, 7], 1)
	], inf)
], inf) + 67;
x = c.asStream;

Routine({
	loop({
		Synth(\help_sinegrain, [\freq, x.next.midicps, \dur, 0.2]);
		0.17.wait;
	})
}).play;
)