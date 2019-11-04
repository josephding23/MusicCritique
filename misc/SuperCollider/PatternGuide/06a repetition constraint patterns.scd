(
p = Pbind(
	\n, Pwhite(3, 10, inf), // making 'n' a separate stream so that degree and dur can share it
	\degree, Pstutter(Pkey(\n), Pwhite(-4, 11, inf)),
	\dur, Pstutter(Pkey(\n), Pwhite(0.1, 0.4, inf)),
	\legato, 0.3
).play;
)
//Pstutter(n, pattern): each value from pattern is repeated n times, n is also a pattern

(
p = Pn(
	Plazy {
		Pbind(
			\degree, Pfin(rrand(3, 10), rrand(-4, 11)),
			\dur, rrand(0.1, 0.4)
		)
	}, inf
).play
)
// Pn(pattern, repeats): embeds the source pattern repeats times
// Plazy buils a new Pbind for each iteration
// Pfin cuts off the Pbind when it's time for a new value

(
p = Pbind(
	\degree, Pstutter(Pwhite(3, 10, inf), Pwhite(-4, 11, inf)),
	\dur, Pclutch(Pwhite(0.1, 0.4, inf), Pdiff(Pkey(\degree)).abs > 0),
	\legato, 0.3
).play;
)
// Pclutch(pattern, connected): if connected pattern is true, Pclutch returns the next value from pattern. if connected is false, the previous pattern value is repeated.
// if Pdiff returns 0, degree has not changed

(
p = Pn(Pbind(
	\degree, Pseq([-7, Pwhite(0, 11, inf)], 1),
	\dur, Pconst(4, Pwhite(1, 4, inf) * 0.25
)), inf).play;
)
// Pconst(sum, pattern, tolerence): Output numbers until the sum goes over a predefined limit. The last output value is adjusted so that the sum matches the limit exactly

(
p = Pn(Pfindur(4, Pbind(
	\degree, Pseq([-7, Pwhite(0, 11, inf)], 1),
	\dur, Pconst(4, Pwhite(1, 4, inf) * 0.25)
)), inf).play;
)
// Pfindur(dur, pattern, tolerence): applying the "constrain" behavior to the event's rhythmic values. The source pattern runs up to the specified duration, then stops.
//Pconst and Pfindur both can ensure the total rhythm doesn't go above 4.0, and create 4-beat segments with randomized rhythm

(
var startTime;
p = (Psync(Pbind(
	\dur, 0.25,
	\time, Pfunc {startTime = startTime ? (thisThread.clock.beats.debug("time"))}
), 1, 4) ++ Pfuncn({
	thisThread.clock.beats.debug("finish time");
	(thisThread.clock.beats - startTime).debug("elapsed");
	nil
}, 1)).play;
)

(
var startTime;
p = (Psync(Pbind(
	\dur, Pn(0.25, 5),
	\time, Pfunc{ startTime = startTime ? (thisThread.clock.beats.debug("time"))}
), 1, 4) ++ Pfuncn({
	thisThread.clock.beats.debug("finish time");
	(thisThread.clock.beats - startTime).debug("elapsed");
	nil
}, 1)).play;
)

//Psync(pattern, quant, maxdur, tolerence): like Pfindur, but does not have a fixed duration limit. Instead, it plays until either it reaches maxdur, or the pattern stops early and the last event is rounded up to the next integer multiple of quant