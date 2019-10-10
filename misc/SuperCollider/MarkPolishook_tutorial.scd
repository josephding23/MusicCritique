s = Server("aServer", NetAddr("127.0.0.1", 56789));
s.boot;

SynthDef("sine", {Out.ar(0, SinOsc.ar(4400, 0, 0.2))}).send(s);

s.sendMsg("s_new", "sine", n = s.nextNodeID, 0, 1);
s.sendMsg("/n_free", n);

s.quit;

{Out.ar(0, [SinOsc.ar(500, 0, 0.5).distort, SinOsc.ar(500, 0, 0.5)], 0, 1)}.scope;

/////////////////////////////////////////////


//Scalling = multiplication
{BrownNoise.ar(0.1) * SinOsc.kr(1, 1)}.scope;

(
{
	(WhiteNoise.ar(0.1) * SinOsc.kr(1, 1))
	+
	(BrownNoise.ar(0.1) * SinOsc.kr(1, 1))
}.scope;
)


//Addictive Synthesis
({
	var n = 12;
	Mix.arFill(
		n,
		{
			SinOsc.ar(
				[67.0.rrand(2000), 67.0.rrand(2000)],
				0,
				n.reciprocal * 0.75
			)
		}
	)
	*
	EnvGen.kr(Env.perc(11, 6), doneAction: Done.freeSelf)
}.scope
)


//Envelopes
(
{
	var n = 12;
	Mix.arFill(
		n,
		{
			SinOsc.ar(
				[67.rrand(2000), 67.0.rrand(2000)],
				0,
				n.reciprocal
			)
			*
			EnvGen.kr(
				Env.sine(2.0.rrand(17)),
				doneAction: Done.none
			)
		}
	)
	*
	EnvGen.kr(
		Env.perc(11, 6),
		doneAction: Done.freeSelf,
		levelScale: 0.75
	)
}.scope;
)


//Ring modulation
{SinOsc.ar(440, 0, 0.571) * SinOsc.kr(880)}.scope;

(
{
	SinOsc.ar(440, 0, 0.571)
	*
	(SinOsc.kr(880) *
		SinOsc.kr([6.99, 8.01].reciprocal))
}.scope;
)


//Amplitude modulation
(
{
	SinOsc.ar(440, 0, 0.571)
	*
	(SinOsc.kr(880).abs *
		SinOsc.kr([6.99, 8.01].reciprocal)
	)
}.scope;
)

//////////////////////////////////////

(
SynthDef("subtractive", { arg out;
	Out.ar(
		out,
		LPF.ar(
			Pulse.ar(440, 0.5, 0.1),
			Line.kr(8000, 660, 6)
		)
	)
})
)

Synth("subtractive")
(
SynthDef("passLowFreqs2", {arg out;
	Out.ar(
		out,
		RLPF.ar(
			Saw.ar([220, 221] + LFNoise0.kr(1, 100, 200), 0.2),
			[LFNoise0.kr(4, 600, 2400), LFNoise0.kr(3, 600, 2400)],
			0.1
		)
	)
}).add;
)

Synth("passLowFreqs2")

(
SynthDef("noiseToPitch", {arg out = 0, mul = 1;
	Out.ar(
		out,
		Resonz.ar(
			WhiteNoise.ar(mul),
			LFNoise0.kr(4, 110, 660),
			[0.005, 0.005]
		)
	)
}).add;
)

(
Synth("noiseToPitch", [\out, 0, \mul, 1]);
Synth("noiseToPitch", [\out, 1, \mul, 1]);
)

///////////////////////////////

//Playbuf

(
b = Buffer.read(s, Platform.resourceDir +"/"+ "sounds/a11wlk01.wav");
SynthDef("aMonoSamplePlayer", { arg bus = 0, bufnum = 0, rateScale = 1;
	Out.ar(
		bus,
		PlayBuf.ar(
			1,
			bufnum,
			BufRateScale.kr(bufnum) * rateScale
		)
		*
		EnvGen.kr(Env.sine(BufDur.kr(bufnum)))
	)
}).add;
)

(
Synth("aMonoSamplePlayer", [\bus, 0, \bufNum, b, \rateScale, 0.99]);
Synth("aMonoSamplePlayer", [\bus, 1, \bufNum, b, \rateScale, 1.01]);
)

////////////////////////////////

//Time-based filters

(
SynthDef("someGrains", {arg centerFreq = 777, freqDev = 200, grainFreq = 2;
	var gate;
	gate = Impulse.kr(grainFreq);
	Out.ar(
		0,
		SinOsc.ar(
			LFNoise0.kr(4, freqDev, centerFreq),
			0,
			EnvGen.kr(Env.sine(0.1), gate, 0.1)
		)
	)
}).add;

SynthDef("aDelay", {arg delay = 0.25;
	Out.ar(
		1,
		DelayN.ar(
			In.ar(0, 1),
			delay,
			delay,
		)
	)
}).add;
)

Synth("someGrains");
Synth("aDelay");

(
~source = Group.head(s);
~effects = Group.tail(s);
)

(
Synth.head(~source, "someGrains");
Synth.head(~effects, "aDelay");
)

//Feedback filters

(
{
	CombN.ar(
		SinOsc.ar(500.rrand(1000), 0, 0.2) * Line.kr(1, 0, 0.1),
		0.3,
		0.25,
		6
	)
}.scope
)

(
{
	AllpassN.ar(
		SinOsc.ar(500.rrand(1000), 0, 0.2) * Line.kr(1, 0, 0.1),
		0.3,
		0.25,
		6
	)
}.scope;
)

//Reverberation

(
{
	var s, z, y;
	s = Mix.ar(Array.fill(10, {Resonz.ar(Dust.ar(0.2, 50), 200 + 3000.0.rand, 0.003)}));
	z = DelayN.ar(s, 0.048);
	y = Mix.ar(Array.fill(7, {CombL.ar(z, 0.1, LFNoise1.kr(0.1.rand, 0.04, 0.05), 15)}));
	4.do({y = AllpassN.ar(y, 0.050, [0.050.rand, 0.050.rand], 1)});
	s + (0.2*y)
}.scope
)


//Carriers and modulators

(
SynthDef("fm1", { arg bus = 0, freq = 440, carPartial = 1, modPartial = 1, index = 3, mul = 0.05;

	//index values usually are between 0 and 24
	//carPartial :: modPartial => car/mod ratio

	var mod;
	var car;

	mod = SinOsc.ar(
		freq * modPartial,
		0,
		freq * index * LFNoise1.kr(5.reciprocal).abs
	);

	car = SinOsc.ar(
		(freq * carPartial) + mod;
		0,
		mul
	);

	Out.ar(
		bus,
		car
	)
}).add;
)

(
Synth("fm1", [\bus, 0, \freq, 440, \carPartial, 1, \modPartial, 2.4, \index, 10], );
Synth("fm1", [\bus, 1, \freq, 442, \carPartial, 1, \modPartial, 2.401]);
)