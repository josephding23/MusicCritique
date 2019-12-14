// Reverbs

(
{
	var x;
	x = Klank.ar(`[[200, 671, 1153, 1723], nil, [1, 1, 1, 1]], Dust2.ar(2, 0.1));
	x = Pan2.ar(x, -0.2);
	x = [x[0], DelayC.ar(x[1], 0.01, 0.01)];
	FreeVerb.ar(x, 0.75, 0.9, 0.4);
	/*
	arguments: in,
	mix(dry/wet balance. range 0..1),
	room(room size. range 0..1),
	damp(Reverb HF damp. range 0..1), mul, add
	*/
}.scope;
)

(
{
	var x;
	x = Klank.ar(`[[200, 671, 1153, 1723], nil, [1, 1, 1, 1]], Dust.ar(2, 0.1));
	GVerb.ar(x, 105, 5, 0.7, 0.8, 60, 0.1, 0.5, 0.4) + x;
	/*
	arguments: in,
	roomsize(in squared meters),
	revtime(in seconds),
	damping(0 to 1, high frequency folloff, 0 damps the reverb signal completely, 1 not at all),
	inputbw(0 to 1, same as damping control, but on the input signal),
	spread(a control on the stereo spread and diffusion of the reverb signal),
	drylevel(amount of dry signal),
	earlyreflevel(amount of early reflection level),
	taillevel(amount of tail level),
	maxroomsize(to set the size of the delay lines. defaults to +1)
	*/
}.scope;
)