Pseq(#[1, 2, 3], 4).asStream.all;
//Play through the entire list repeats times

Pser(#[1, 2, 3], 4).asStream.all;
//Only repeats items

(
p = Pbind(
	\degree, Prand([0, 1, 2, 4, 5], inf),
	\dur, 0.25
).play;
)
//Prand: Choose items from the list randomly

(
p = Pbind(
	\degree, Pshuf([0, 1, 2, 4, 5], inf),
	\dur, 0.25
).play;
)
//Pshuf: Shuffle the list in random order, and use the same random order repeats times

Place([0, [1, 2], [3, 4, 5]], 3).asStream.all;
//Take one from each item in the main array items in succession

(
p = Pbind(
	\degree, Ppatlace([
		Pseries(0, 1, 8),
		Pseries(2, 1, 7)
	], inf),
	\dur, 0.25
).play;
)
//Take one value from each sub-pattern in order

(
p = Pbind(
	\degree, Pseries(-7, 1, 15),
	\dur, Pgeom(0.5, 0.89140193218427, 15)
).play;
)
//Pgeom: geometric series, successively multiplying the current value by grow




