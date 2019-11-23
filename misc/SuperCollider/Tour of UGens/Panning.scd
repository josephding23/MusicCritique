// Panning

(
s.quit;
s.options.numOutputBusChannels = 8;
s.options.numInputBusChannels = 8;
s.boot;
)

{ Pan2.ar(BrownNoise.ar, MouseX.kr(-1, 1), 0.3) }.scope(2);
{ Pan2.ar(BrownNoise.ar, SinOsc.kr(0.2), 0.3) }.scope(2);
// equal power stereo pan a mono source
// arguments: in, pan position, level

{ LinPan2.ar(BrownNoise.ar, MouseX.kr(-1, 1), 0.3) }.scope(2);
{ LinPan2.ar(BrownNoise.ar, SinOsc.kr(0.2), 0.3) }.scope(2);
// linear pan a mono source (not equal power)
// arguments: in, pan position, level

{ Balance2.ar(BrownNoise.ar, BrownNoise.ar, MouseX.kr(-1, 1), 0.3) }.scope(2);
// balance a stereo source
// arguments: in, pan position, level

{ Pan4.ar(BrownNoise.ar, MouseX.kr(-1, 1), MouseY.kr(1, -1), 0.3) }.scope(4);
// Four channel equal power panner. Outputs are in order LeftFront, RightFront, LeftBack, RightBack
// arguments: in, xpos, ypos, level

{ PanAz.ar(5, BrownNoise.ar, MouseX.kr(-1, 1), 0.3, 2) }.scope(5);
{ PanAz.ar(5, BrownNoise.ar, MouseX.kr(-1, 1), 0.3, 3) }.scope(5);
// Multichannel equal power panner
/*
arguments:
numChans, in,
level(A control rate level input),
width(the width of the pannnig envelope),
orientation(Should be 0 if the front is a vertex of the spanning polygon. The first speaker will be directly in front. Should be 0.5 if the front bisects a side of the spanning polygon. Then the first speaker will be the one left of center.)
*/

{ XFade2.ar(BrownNoise.ar, SinOsc.ar(500), MouseX.kr(-1, 1), 0.3) }.scope(1);
// equal power cross fade between two inputs
// arguments: in1, in2, pan(crossfadee position from -1 to +1), level

(
{
	var w, x, y, p, lf, rf, rr, lr;

	p = BrownNoise.ar;  // source

	#w, x, y = PanB2.ar(p, MouseX.kr(-1, 1), 0.3);
	// Encodes a mono signal to 2-dimensional ambisonic B-format
	// arguments: in,
	// azimuth(position around the circle from -1 to +1), gain(amplitude control)


	#lf, rf, rr, lr = DecodeB2.ar(4, w, x, y);
	// Decodes a two dimensional ambisonic B-format signal to a set of speakers in a regular polygon. The outputs will be in clockwise order.

	[lr, rf, lr, rr]
}.scope(4);
)

(
{
	var x, y;
	x = Mix.fill(4, { LFSaw.ar(200 + 2.0.rand2, 0, 0.1) });
	y = WhiteNoise.ar * LFPulse.kr(3, 0, 0.7, 0.2);
	#x, y = Rotate2.ar(x, y, MouseX.kr(0.2));
	[x, y]
}.scope(2);
)
// Rotate2 can be used for rotating an ambisonic B-format sound field around an axis
// arguments: x, y, pos(angle to rotate around the circle from -1 to +1)
// xout = cos(angle) * xin + sin(angle) * yin
// yout = cos(angle) * yin - sin(angle) * xin