e = (freq: 440, dur: 0.5);

e.at(\freq)
e[\freq]
e.freq

e.put(\freq, 880);
e[\freq] = 660;
e.freq = 220;

e.put(\amp, 0.6);
e.put(\dur, nil);
//Pbind is defined by a list of pairs: keys associated with the pattern that will supply the values for the events

(
p = Pbind(
	\degree, Pseq(#[0, 0, 4, 4, 5, 5, 4], 1),
	\dur, Pseq(#[0.5, 0.5, 0.5, 0.5, 0.5, 1], 1)
).asStream;
)

p.next(Event.new);

('degree': 0, 'dur': 0.5).play;

p = Pbind(\degree, Pwhite(0, 7, inf), \dur, 0.25, \legato, 1).play;
p.stop;

p = Pmono(\default, \degree, Pwhite(0, 7, inf), \dur, 0.25).play
//More monophonic
p.stop;

(
SynthDef(\default, { arg out=0, freq=440, amp=0.1, pan=0, gate=1;
    var z;
    z = LPF.ar(
            Mix.new(VarSaw.ar(freq + [0, Rand(-0.4,0.0), Rand(0.0,0.4)], 0, 0.3)),
            XLine.kr(Rand(4000,5000), Rand(2500,3200), 1)
        ) * Linen.kr(gate, 0.01, 0.7, 0.3, 2);
    OffsetOut.ar(out, Pan2.ar(z, pan, amp));
}, [\ir]);
)

(freq: 293.333, amp: 0.2, pan: -0.7).play;

(freq: 440, amp: 0.1, pan: 0.7).play;

(
SynthDef(\harpsi,  { |outbus = 0, freq = 440, amp = 0.1, gate = 1|
	var out;
	out = EnvGen.ar(Env.adsr, gate, doneAction: Done.freeSelf) * amp * Pulse.ar(freq, 0.25, 0.75);
	Out.ar(outbus, out ! 2);
}).add;

p = Pbind(
	\instrument, \harpsi,
	\degree, Pseries(0, 1, 8),
	\dur, 0.25
).play;
)