// Distortion

{ SinOsc.ar(300, 0, 0.2) }.scope(1);
{ SinOsc.ar(300, 0, 0.2).abs }.scope(1);
{ SinOsc.ar(300, 0, 0.2).max(0) }.scope(1);
{ SinOsc.ar(300, 0).squared * 0.2 }.scope(1);
{ SinOsc.ar(300, 0).cubed * 0.2 }.scope(1);

{ SinOsc.ar(300, 0, MouseX.kr(0.1, 80, 1)).distort * 0.2 }.scope(1);
{ SinOsc.ar(300, 0, MouseX.kr(0.1, 80, 1)).softclip * 0.2 }.scope(1);
{ SinOsc.ar(300, 0, MouseX.kr(0.1, 80, 1)).clip(1) * 0.2 }.scope(1);
{ SinOsc.ar(300, 0, MouseX.kr(0.1, 80, 1)).fold2(1) * 0.2 }.scope(1);
{ SinOsc.ar(300, 0, MouseX.kr(0.1, 80, 1)).wrap2(1) * 0.2 }.scope(1);

{ SinOsc.ar(200, 0, 0.2).scaleneg(MouseX.kr(-1, 1)) }.scope(1);

(
{
	var in;
	in = SinOsc.ar(300, 0, MouseX.kr(0.1, 8pi, 1));
	SinOsc.ar(0, in, 0.2);
}.scope(1);
)
// waveshaping by phase modulating a 0 Hz sine oscillator

(
s.sendMsg(\b_alloc, 80, 1024); // allocate table
// fill with chebyshevs
s.listSendMsg([\b_gen, 80, \cheby, 7] ++ {1.0.rand2.squared}.dup(6));

{ Shaper.ar(80, SinOsc.ar(600, 0, MouseX.kr(0.1))) * 0.3; }.scope(1);

s.listSendMsg([\b_gen, 80, \cheby, 7] ++ {1.0.rand2.squared}.dup(6));
s.listSendMsg([\b_gen, 80, \cheby, 7] ++ {1.0.rand2.squared}.dup(6));
)
// Shaper - input is used to look up a value in a table

(
s.boot;

b = Buffer.alloc(s, 512, 1, { |buf| buf.chebyMsg([1,0,1,1,0,1])});
b.plot;
(
{
    Shaper.ar(
        b,
        SinOsc.ar(300, 0, Line.kr(0,1,6)),
        0.5
    )
}.scope;
)
)
