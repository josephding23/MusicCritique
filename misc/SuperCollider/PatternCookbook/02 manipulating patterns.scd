(
var melodies = (
	lowMelody: Pseries(4, Prand(#[-2, -1, 1, 2], inf), inf).fold(-7, 11),
	highMelody: Pseries(14, Prand(#[-3, -2, 2, 3], inf), inf).fold(7, 18)
);
p = Pbind(
	\degree, Pnsym1(Pwrand(#[lowMelody, highMelody], [0.7, 0.3], inf), melodies),
	// Generate lower range pitches 70% of the time, and higher range the other 30%.
	\dur, Pwrand(#[0.25, 0.5], #[0.4, 0.6], inf)
).play;
)

(
~degree = PatternProxy(Pn(Pseries(0, 1, 8), inf));
~dur = PatternProxy(Pn(0.25, inf));
// The PatternProxy is a single object that creates a single stream within Pbind, but it looks for its values to the pattern and stream contained inside the proxy.

p = Pbind(
	\degree, ~degree,
	\dur, ~dur
).play;
)

~degree.source = (Pexprand(1, 8, inf) -1).round;
~dur.source = Pwrand(#[0.25, 0.5, 0.75], #[0.5, 0.3, 0.2], inf);
// Changing the proxy's pattern replaces the stream, without having to touch the Pbind's closed box.

(
Pdefn(\degree, Pn(Pseries(0, 1, 8), inf));
Pdefn(\dur, Pn(0.25, inf));

p = Pbind(
	\degree, Pdefn(\degree),
	\dur, Pdefn(\dur)
).play;
)

Pdefn(\degree, (Pexprand(1, 8, inf) - 1).round);
Pdefn(\dur, Pwrand(#[0.25, 0.5, 0.75], #[0.5, 0.3, 0.2], inf));
// Another way is to use Pdefn, which is a global namespace of proxies for value patterns.