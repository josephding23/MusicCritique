(
p = Pbind(
	\degree, Pwhite(0, 11, inf),
	\dur, Pif(Pkey(\degree).odd, 0.25, 0.5)
).play
)
// Pif(condition, iftrue, iffalse, default): evaluates a pattern condition that returns true or false. Then, one value is taken from the true or false branch before going back to evaluate the condition again. The default value or pattern comes into play when the true or false branch stops producing values (returns nil)

(
p = Pbind(
	\degree, Pseed(0x7FFFFFFF.rand, Pseries({ rrand(-7, 0) }, Pwhite(1, 3, inf), {rrand(4, 10)})),
	\dur, 0.25
)
)
// Pseed sets the random seed before embedding pattern, effectively restarting the random number generator at the start of the pattern

q = p.play;

r = p.play;

(
~go = true;
p = Pwhile({ ~go }, Pbind(
	\degrere, Pseries({ rrand(-7, 0) }, Pwhite(1, 3, inf), { rrand(4, 10)}).trace(prefix: "degree: "),
	\dur, 0.25
)).play;
)
// As long as the function evaluates to true, the pattern is embebbed. The function is checked once at the beginning and thereafter when the pattern comes to an end.
~go = false;
