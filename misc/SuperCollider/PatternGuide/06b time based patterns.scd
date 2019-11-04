(
p =  Pbind(
	\degree, Pif(Ptime(inf) < 4.0, Pwhite(-4, 11, inf)),
	\dur, 0.25
).play;
)
//Ptime(repeats): returns the amount of time elapsed since embedding, can be used to stop a value stream/pattern after a certain amount of time

Env(#[0, 1, 0], #[1,  1], 5).plot;
(
p = Pbind(
	\note, Pseg(
		Pwhite(-7, 19, inf),
		Pwhite(1, 4, inf) * 0.5,
	5, inf),
	\dur, 0.125
).play;
)
//Pseg(levels, durs, curves, repeats): similar to Pstep, bubt interpolates to the next value instead of stepping abruptly at the end of duration. Interpolation is linear by default, but any envelope segment curve can be used.

(
e = Env.linen(1, 1, 1);
e.at(2);
e.at(2.5);

// print envelope values
r = fork {
	var stream = e.asStream;
	12.do({
		stream.next.postln;
		0.25.wait;
	});
};
)

(
p = Pbind(
	\degree, Pwhite(-4, 11, 32),
	\pan, Env(#[-1, 1, -1], #[2, 2], \sin),
	\dur, 0.125
).play;
)
// Use an envelope to pan notes from left to right and back

(
p = Pbind(
	\degree, Pwhite(-4, 11, inf),
	\pan, Pif(Ptime(inf) <= 4.0, Env(#[-1, 1, -1], #[2, 2], \sin)),
	\dur, 0.125
).play;
)
// When the envelope ends, the stream will hold the final level indefinitely.

(
p = Pbind(
	\degree, Pwhite(-4, 11, inf),
	\pan, Pn(Pif(Ptime(inf) <= 4.0, Env(#[-1, 1, -1], #[2, 2], \sin)), inf),
	\dur, 0.125
).play;
)
// To keep looping the envelope, wrap Pif inside Pn

