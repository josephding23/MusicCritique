Env.linen(0.1, 0.2, 0.1, 0.6).test.plot;
Env.linen(1, 2, 3, 0.6).test.plot;
Env.linen(1, 2, 3, 0.6, \sine).test.plot;
Env.linen(1, 2, 3, 0.6, \welch).test.plot;
Env.linen(1, 2, 3, 0.6, -3).test.plot;
Env.linen(1, 2, 3, 0.6, [[\sine, \welch, \lin, \exp]]).plot;

Env.triangle(1, 1).test.plot;

Env.sine(1, 1).test.plot;

Env.perc(0.05, 1, 1, -4).test.plot;
Env.perc(0.001, 1, 1, -4).test.plot;
Env.perc(0.001, 1, 1, -8).test.plot;
Env.perc(1, 0.01, 1, 4).test.plot;

Env.pairs([[0, 1], [2.1, 0.5], [3, 1.4]], \exp).plot;
Env.pairs([[0, 1], [3, 1.4], [2.1, 0.5], [3, 4]], \exp).plot;
Env.pairs({ {1.0.rand} ! 2 } ! 16, \exp).plot;

(
{
	var env = Env.step([0, 3, 5, 2, 7, 3, 0, 3, 4, 0], [0.5, 0.1, 0.2, 1.0, 1.5, 2, 0.2, 0.1, 0.2, 0.1]);
	var envgen = EnvGen.kr(env);
	var freq = (envgen + 60).midicps;
	SinOsc.ar(freq) * 0.1
}.play;
)

Env.adsr(0.02, 0.2, 0.25, 1, 1, -4).test(2).plot;
Env.adsr(0.001, 0.2, 0.25, 1, 1, -4).test(2).plot;
Env.adsr(0.001, 0.2, 0.25, 1, 1, -4).test(0.45).plot;

Env.dadsr().plot;

Env.cutoff(1, 1).test(2).plot;
Env.cutoff(1, 1, 4).test(2).plot;
Env.cutoff(1, 1, \sine).test(2).plot;

{ SinOsc.ar(EnvGen.kr(Env.circle([00, 1, 0], [0.01, 0.5, 0.2])) * 440 + 200) * 0.2}.play;
{ SinOsc.ar(EnvGen.kr(Env.circle([0, 1, 0, 2, 0, 1, 0], [0.01, 0.3])) * 440 + 200)* 0.2 }.play;
{ SinOsc.ar(EnvGen.kr(Env.circle([0, 1, 0, (2..4), 0, (1..3), 0], [0.01, 0.3])) * 440 + 200).sum * 0.2}.play;
