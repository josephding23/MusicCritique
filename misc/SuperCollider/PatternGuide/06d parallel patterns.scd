(
p = Pspawner({ |sp| // sp = the Spawner object
	loop {
		sp.par(Pbind(
			\degree, Pseries(rrand(-5, 7), #[-1, 1].choose, rrand(4, 7)),
			\pan, rrand(-1.0, 1.0),
			\dur, rrand(0.1, 0.3)
		));
		// tell the Spawner eto wait a bit before the next pattern goes
		sp.wait(rrand(1, 4) * 0.25);
	}
}).play;
)
// Pspawner is run in a Routine. A Spawner object gets passed into this Routine, and this object is used to add or remove stream to/from the parallel stream.


(
p = Pspawn(Pbind(
	\method, \par,
	\pattern , Pfunc { // generate the subpattern in a Pfunc, Pfunc returns the pattern without rendering the stream
		Pbind(
			\degree, Pseries(rrand(-5, 7), #[-1, 1].choose, rrand(4, 7)),
			\pan, rrand(-1.0, 1.0),
			\dur, rrand(0.1, 0.3)
		)
	},
	\delta, Pwhite(1, 4, inf) * 0.25
)).play;
)
// Pspawn supports most of the features of Pspawner, but uses a pattern to control the Spawner object instead of a Routine function