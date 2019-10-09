(
"Hello World!".postln;
"Hello".postln;
)

{[SinOsc.ar(440, 0, 0.2), SinOsc.ar(442, 0, 0.2)]}.play;

f = {3.0.rand;};
f.value

f = {arg a; a.value + 3};
f.value(3);

g = {3.0.rand;};
f.value(g);

(
f = {|a, b| a / b;};
f.value(a: 10, b: 2)
)

{SinOsc.ar([440, 660], 0, 0.2)}.play;

(
{ var ampOsc;
	ampOsc = SinOsc.kr(0.5, 1.5pi, 0.5, 0.5);
	SinOsc.ar(440, 0, ampOsc);
}.play;
)

(
{ var freq;
	freq = [[660, 880], [440, 660], 1320, 880].choose;
	SinOsc.ar(freq, 0, 0.2);
}.play;
)

{Pan2.ar(PinkNoise.ar(0.2), -0.3)}.play;

{PinkNoise.ar(0.2) + SinOsc.ar(440, 0, 0.2) + Saw.ar(660, 0.2)}.play;

{Mix.new([SinOsc.ar(440, 0, 0.2), Saw.ar(660, 0.2)]).postln}.play;

(
{
	var a, b;
	a = [SinOsc.ar(440, 0, 0.2), Saw.ar(662, 0.2)];
	b = [SinOsc.ar(442, 0, 0.2), Saw.ar(660, 0.2)];
	Mix([a, b]).postln;
}.play;
)

(
var n = 8;
{Mix.fill(n, {SinOsc.ar(500 + 500.0.rand, 0, 1 / n)})}.play;
)

(
var n = 8;
{
	Mix.fill(n, {arg index;
		var freq;
		index.postln;
		freq = 440 + index;
		freq.postln;
		SinOsc.ar(freq, 0, 1/n)
	})
}.play;
)

{PinkNoise.ar(0.2) + SinOsc.ar(440, 0, 0.2) + Saw.ar(660, 0.2)}.plot;

{PinkNoise.ar(0.2) + SinOsc.ar(440, 0, 0.2) + Saw.ar(660, 0.2)}.scope;

{[SinOsc.ar(440, 0, 0.2), SinOsc.ar(442, 0, 0.2)]}.scope(zoom: 10);


s.scope;
