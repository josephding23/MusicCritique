(
SynthDef(\volumeCtl, { |out, amp = 1, gate = 1|
	var sig = In.ar(out, 2) * amp;
	sig = sig * EnvGen.kr(Env(#[1, 1, 0], #[1, 1], -3, releaseNode: 1), gate, doneAction: Done.freeSelf);
	ReplaceOut.ar(out, sig)
}).add;

~vbus1 = Bus.control(s, 1).set(0.5);
~vbus2 = Bus.control(s, 1).set(0.5);

~window = Window.new("mixers", Rect(10, 100, 320, 60));
~window.view.decorator = FlowLayout(~window.view.bounds, 2@2);
EZSlider(~window, 310@20, "low part", \amp, { |ez| ~vbus1.set(ez.value) }, 0.5);
~window.view.decorator.nextLine;
EZSlider(~window, 310@20, "high part", \amp, { |ez| ~vbus2.set(ez.value)}, 0.5);
~window.front.onClose_({ ~vbus1.free; ~vbus2.free});
)

(
p = Ppar([
	Pfxb(Pbind(
		\degree, Pseq([0, 7, 4, 3, 9, 5, 1, 4], inf),
		\octave, 4,
		\dur, 0.5
	), \volumeCtl, \amp, ~vbus1.asMap),
	Pfxb(Pbind(
		\degree, Pwhite(0, 11, inf),
		\dur, 0.25,
	), \volumeCtl, \amp, ~vbus2.asMap)
]).play;
)
//Pfxb(pattern, fxname, pairs): read from the bus identified by the out argument, and write he processed signal onto the same bus usin either ReplaceOut or XOut. Pfx uses whatever bus and group are specified in the incoming event. Pxfb allocates a separate bus and group for the effect and the pattern.

(
SynthDef(\playbuf, { |bufnum, start, dur = 1, amp = 0.2, out|
	var sig = PlayBuf.ar(1, bufnum, BufRateScale.ir(bufnum), 0, start);
	sig = sig * amp * EnvGen.kr(Env.linen(0.01, dur, 0.01), doneAction: Done.freeSelf);
	Out.ar(out, sig !  2)
}).add;
)

(
TempoClock.default.tempo = 1;
p = Pproto({
	~buf = (type: \allocRead, path: Platform.resourceDir +/+ "sounds/a11wlk01.wav").yield;
}, Pbind(
	\instrument, \playbuf,
	\bufnum, Pkey(\buf),
	\dur, Pwhite(1, 4, inf) * 0.25,
	\start, Pwhite(0, 188893 - (Pkey(\dur) * 44100), inf)
)).play;
)
//load sound file or wavetable buffers or play synths as part of the preparation to run a Pbind-style pattern. When the Pbind stops, those resources would be removed automatically from the server.