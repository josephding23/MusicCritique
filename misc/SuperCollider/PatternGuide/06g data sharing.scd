p = Pkey(\a).asStream;
//Read the key in the input event, outputs values until the input event doesn't contain the key(i.e., the value is nil). There is no repeats argument.
p.next((a: 2, b: 3));

p = (Pkey(\a) * Pkey(\b)).asStream;
p.next((a: 2, b: 3));

(
p = Pbind(
	\degree, Pseq([Pseries(-7, 1, 14), Pseries(7, -1, 14)], inf),
	\dur, 0.25,
	//Pkey can only look backward to keys stated earlier in the Pbind definition.
	\legato, Pkey(\degree).linexp(-7, 7, 2.0, 0.05)
).play;
)

(
SynthDef(\playbuf, { |bufnum, start, dur = 1, amp = 0.2, out|
	var sig = PlayBuf.ar(1, bufnum, BufRateScale.ir(bufnum), 0, start);
	sig = sig * amp * EnvGen.kr(Env.linen(0.01, dur, 0.01), doneAction: Done.freeSelf);
	Out.ar(out, sig ! 2)
}).add;
)

(
TempoClock.default.tempo = 1;
p = Pfset({
    ~buf = Buffer.read(s, Platform.resourceDir +/+ "sounds/a11wlk01.wav");
    0.2.yield;
    ~bufFrames = ~buf.numFrames;
}, Pbind(
    \instrument, \playbuf,
    \bufnum, Pkey(\buf),
    \dur, Pwhite(1, 4, inf) * 0.25,
    \start, Pwhite(0, Pkey(\bufFrames) - (Pkey(\dur) * 44100), inf)
), { defer(inEnvir { "freeing buffer".postln; ~buf.free }, 1.1) }).play;
)
//When embedded, Pfset creates an environment and populates it using environment variable assignments variable assignments in the provided function. For every next call, the value in the preset environment are inserted into the event prototype before evaluating the child pattern. It could also be used to load objects on the serverm although this takes some care because the object would be reloader every time the Pfset is played and you are responsible for freeing objects created this way in the cleanupFunc
s.bufferAllocator.debug;

p.stop;