//10 SynthDefs and Synths

{SinOsc.ar(440, 0, 0.2)}.play;

SynthDef.new("tutorial-SinOsc", {Out.ar(0, SinOsc.ar(440, 0, 0.2))}).play;

(
SynthDef.new("tutotial-SinOsc-stereo", {var outArray;
	outArray = [SinOsc.ar(440, 0, 0.2), SinOsc.ar(442, 0, 0.2)];
	Out.ar(0, outArray)
}).play;
)

x = {SinOsc.ar(660, 0, 0.2)}.play;
y = SynthDef.new("tutorial-SinOsc", {Out.ar(0, SinOsc.ar(440, 0, 0.2))}).play
x.free
y.free

SynthDef.new("tutorial-PinkNoise", {Out.ar(0, PinkNoise.ar(0.3))}).add;

x = Synth.new("tutorial-PinkNoise");
x.free

SynthDef("tutorial-Rand", {Out.ar(0, SinOsc.ar(Rand(440, 660), 0, 0.2))}).add;
x = Synth("tutorial-Rand");
y = Synth("tutorial-Rand");

(
SynthDef("tutorial-args", {arg freq = 440, out = 0;
	Out.ar(out, SinOsc.ar(freq, 0, 0.2));
}).add;
)

x = Synth("tutorial-args");
y = Synth("tutorial-args", ["freq", 660]);
z = Synth("tutorial-args", ["freq", 880, "out", 1])

"a String" === "a String"
\aSymbol === 'aSymbol'


//11 Busses

In.ar(0, 1);
In.ar(0, 4);

{Out.kr(0, SinOsc.ar)}.scope;

b = Bus.control(s, 2);
c = Bus.audio(s);

(
SynthDef("tutorial-args", { arg freq = 440, out = 0;
	Out.ar(out, SinOsc.ar(freq, 0, 0.2));
}).add;
)
x = Synth("tutorial-args", [\out, 1, \freq, 660]);
y = Synth("tutorial-args", [\out, 1, \freq, 770]);


(
SynthDef.new("tutorial-SinOsc-stereo", { var outArray;
	outArray = [SinOsc.ar(440, 0, 0.2), SinOsc.ar(442, 0, 0.2)];
	Out.ar(0, outArray);
}).play;
)

s.reboot;

b = Bus.control(s, 2);
b.index;
b.numChannels
c = Bus.control(s);
c.numChannels
c.index;

(
SynthDef("tutorial-Infreq", {arg bus, freqOffset = 0;
	Out.ar(0, SinOsc.ar(In.kr(bus) + freqOffset, 0, 0.5));
}).add;

SynthDef("tutorial-Outfreq", {arg freq = 400, bus;
	Out.kr(bus, SinOsc.kr(1, 0, freq/40, freq));
}).add;

b = Bus.control(s, 1);
)

(
x = Synth.new("tutorial-Outfreq", [\bus, b]);
y = Synth.after(x, "tutorial-Infreq", [\bus, b]);
z = Synth.after(x, "tutorial-Infreq", [\bus, b, \freqOffset, 200]);
)

(
SynthDef("tutorial-DecayPink", { arg outBus = 0, effectBus, direct = 0.5;
	var source;
	source = Decay2.ar(Impulse.ar(1, 0.25), 0.01, 0.2, PinkNoise.ar);
	Out.ar(outBus, source * direct);
	Out.ar(effectBus, source * (1 - direct));
}).add;

SynthDef("tutorial-DecaySin", { arg outBus = 0, effectBus, direct = 0.5;
	var source;
	source = Decay2.ar(Impulse.ar(0.3, 0.25), 0.3, 1, SinOsc.ar(SinOsc.kr(0.2, 0, 110, 440)));
	Out.ar(outBus, source * direct);
	Out.ar(effectBus, source * (1 - direct));
}).add;

SynthDef("tutorial-Reverb", { arg outBus = 0, inBus;
	var input;
	input = In.ar(inBus, 1);
	16.do({input = AllpassC.ar(input, 0.04, {Rand(0.001, 0.04)}.dup, 3)});
	Out.ar(outBus, input);
}).add;

b = Bus.audio(s, 1);
)

(
x = Synth.new("tutorial-Reverb", [\inBus, b]);
y = Synth.before(x, "tutorial-DecayPink", [\effectBus, b]);
z = Synth.before(x, "tutorial-DecaySin", [\effectBus, b, \outBus, 1]);
)

y.set(\direct, 1);
z.set(\direct, 1);
y.set(\direct, 0);
z.set(\direct, 0);
x.free; y.free; z.free; b.free;

(
b = Bus.control(s, 1); b.set(880);
c = Bus.control(s, 1); c.set(884);
x = SynthDef("tutorial-map", { arg freq1 = 440, freq2 = 440;
	Out.ar(0, SinOsc.ar([freq1, freq2], 0, 0.1));
}).play(s);
)

x.map(\freq1, b, \freq2, c);

y = {Out.kr(b, SinOsc.kr(1, 0, 50, 880))}.play(addAction: \addToHead);
y.free;

b.get({ arg val; val.postln; f = val;});

x.set(\freq2, f / 2);
c.set(200);

x.free; b.free; c.free;

b = Bus.control(s, 1); b.set(880);
(
f = nil;
b.get({ arg val; f = val;});
f.postln;
)

f.postln;

x = Synth("default", [\freq, 300]);
y = Synth("default", [\freq, 450], x, \addAfter);
z = Synth("default", [\freq, 600], x, \addAfter)
x.free; y.free;
z.free

//12 Groups

(
SynthDef(\tutorial_DecaySin2, {arg outBus = 0, effectBus, direct = 0.5, freq = 440;
	var source;
	source = Pan2.ar(Decay2.ar(Impulse.ar(Rand(0.3, 1), 0, 0.125), 0.3, 1, SinOsc.ar(SinOsc.kr(0.2, 0, 110, freq))), Rand(-1.0, 1.0));
	Out.ar(outBus, source * direct);
	Out.ar(effectBus, source * (1 - direct));
}).add;

SynthDef(\tutorial_Reverb2, {arg outBus = 0, inBus;
	var input;
	input = In.ar(inBus, 2);
	16.do({input = AllpassC.ar(input, 0.04, Rand(0.001, 0.04), 3)});
	Out.ar(outBus, input);
}).add;
)

(
~sources = Group.new;
~effects = Group.after(~sources);
~bus = Bus.audio(s, 2);
)

(
x = Synth(\tutorial_Reverb2, [\inBus, ~bus], ~effects);
y = Synth(\tutorial_DecaySin2, [\effectBus, ~bus, \outBus, 0], ~sources);
z = Synth(\tutorial_DecaySin2, [\effectBus, ~bus, \outBus, 0, \freq, 660], ~sources);
)

~sources.free; ~effects.free;
~bus.free;

currentEnvironment.clear;
Environment.new.push;
currentEnvironment.pop;

s.boot;
a = Synth.new(\default);
a.group;

{SinOsc.ar(mul: 0.2)}.scope(1);
s.queryAllNodes;

SinOsc.superclass;
PureUGen.superclass;
UGen.superclass;
UGen.help;

//13 Buffer

s.boot;
b = Buffer.alloc(s, 100, 2);
b.free;

b = Buffer.alloc(s, s.sampleRate * 8.0, 2);
b.free;

b = Buffer.read(s, "D:/PycharmProjects/MusicCritique/misc/SuperCollider/okay-2.wav");

(
x = SynthDef("tutorial-PlayBuf", { arg out = 0, bufnum;
	Out.ar( out, PlayBuf.ar(1, bufnum, BufRateScale.kr(bufnum)))
}).play(s, [\bufnum, b]);
)

x.free; b.free;

(
SynthDef("tutorial-Buffer-cue", {arg out = 0, bufnum;
	Out.ar(out, DiskIn.ar(1, bufnum))
}).add;
)

b = Buffer.cueSoundFile(s, "D:/PycharmProjects/MusicCritique/misc/SuperCollider/okay-2.wav", 0, 2);
y = Synth.new("tutorial-Buffer-cue", [\bufnum, b], s);
b.free; y.free;

b = Buffer.read(s, "D:/PycharmProjects/MusicCritique/misc/SuperCollider/okay-2.wav")
b.bufnum;
b.numFrames;
b.numChannels;
b.sampleRate;
b.free;

(
b = Buffer.read(s, "D:/PycharmProjects/MusicCritique/misc/SuperCollider/okay-2.wav", action: {arg buffer;
	("numFrames after update:" + buffer.numFrames).postln;
	x = {PlayBuf.ar(1, buffer, BufRateScale.kr(buffer))}.play;
}
);
("numFrames before update:" + b.numFrames).postln;
)
x.free; b.free;

b = Buffer.alloc(s, s.sampleRate * 5, 1);

(
x = SynthDef("tutorial-RecordBuf", { arg out=0, bufnum=0;
	var noise;
	noise = PinkNoise.ar(0.3);
	RecordBuf.ar(noise, bufnum);
}).play(s, [\out, 0, \bufnum, b])
)

x.free;

(
SynthDef("tutorial-playback", {arg out=0, bufnum=0;
	var playbuf;
	playbuf = PlayBuf.ar(1, bufnum);
	FreeSelfWhenDone.kr(playbuf);
	Out.ar(out, playbuf);
})
)
b.free;

b = Buffer.alloc(s, 8, 1);
b.set(7, 0.5);
b.get(7, {|msg| msg.postln});
b.free;
b.postln;

b = Buffer.alloc(s, 16);
b.setn(0, [1, 2, 3]);
b.getn(0, 3, {|msg| msg.postln});
b.setn(0, Array.fill(b.numFrames, {1.0.rand}));
b.getn(0, b.numFrames, {|msg| msg.postln});
b.free;

(
v = FloatArray.fill(44100, {1.0.rand2});
b = Buffer.alloc(s, 44100);
)
(
b.loadCollection(v, action: {|buf|
	x = {PlayBuf.ar(buf.numChannels, buf, BufRateScale.kr(buf), loop: 1) * 0.2}.play;
});
)
x.free;

b.loadToFloatArray(0, -1, {|floatArray| (floatArray == v).postln});
b.free;

b = Buffer.read(s, "D:/PycharmProjects/MusicCritique/misc/SuperCollider/okay-2.wav");
b.plot;

b.play;
x = b.play(true);
x.free; b.free;

b = Buffer.alloc(s, 512, 1);
b.cheby([1, 0, 1, 1, 0, 1]);
(
x = play({
	Shaper.ar(
		b,
		SinOsc.ar(300, 0, Line.kr(0, 1, 6)),
		0.5
	)
});
)
x.free; b.free;