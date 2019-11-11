f = { |x| x + 1};
g = { |x| x * 2};

h = (g <> f);
//means to evaluate f first, then pass its result to g.

h.value(1);

(f <> g).value(1);

g.value(f.value(1));
// g · f == g(f(x)) -- f is evaluated first, and its reselt is passed to g

(
~rhythm = Pbind(
	\dur, Pwrand(#[0.125, 0.25, 0.5], #[0.3, 0.5, 0.2], inf),
	\legato, Pwrand(#[0.1, 0.6, 1.01], #[0.1, 0.3, 0.6], inf)
);
~melody = Pbind(
	\degree, Pwhite(-4, 11, inf)
);

p = Pchain(~melody, ~rhythm).play
)
//Chains separate Pbind-style patterns together, so that all their key-value pairs to into th same event

(
~rhythm = EventPatternProxy(Pbind(
	\dur, Pwrand(#[0.125, 0.25, 0.5], #[0.3, 0.5, 0.2], inf),
	\legato, Pwrand(#[0.1, 0.6, 1.01], #[0.1, 0.3, 0.6], inf)
));

~melody = EventPatternProxy(Pbind(
	\degree, Pwhite(-4, 11, inf)
));

p = Pchain(~melody, ~rhythm).play;
)
//EventPatternProxy makes it possible to swap different melody and rhythm components in and out on the fly, with no interruption.

~melody.source = PmonoArtic(\default, \degree, Pseries(4, Prand(#[-1, 1], inf), inf).fold(-4, 11));

~melody.source = Pbind(\degree, Pseries(4, Pwrand(#[-2, -1, -1, 2], #[0.3, 0.2, 0.2, 0.3], inf), inf).fold(-4, 11));

