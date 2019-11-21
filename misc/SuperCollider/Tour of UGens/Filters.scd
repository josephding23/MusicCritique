// Filters

// Low Pass, High Pass
// arguments: in, freq(Cutoff frequency in Hertz), mul, add
{ LPF.ar(Saw.ar(200, 0.1), SinOsc.kr(XLine.kr(0.7, 300, 20), 0, 3600, 4000)) }.scope(1);
// A second order low pass filter
{ HPF.ar(Saw.ar(200, 0.1), FSinOsc.kr(XLine.kr(0.7, 300, 20), 0, 3600, 4000), 5)}.sco


// Band Pass, Band Cut
// arguments: in, freq, rq, mul, add
// rq： the reciprocal of Q, Q is conventionally defined as cutoffFreq / bandwidth, meaning rq = (bandwidth / cutoffFreq)。
{ BPF.ar(Saw.ar(200, 0.5), FSinOsc.kr(XLine.kr(0.7, 300, 20), 0, 3600, 4000), 0.3) }.scope(1);
// A second order band pass filter
{ BRF.ar(Saw.ar(200, 0.5), FSinOsc.kr(XLine.kr(0.7, 300, 20), 0, 3800, 4000), 0.3) }.scope(1);
// A second order band reject filter


// Resonant Low Pass, High Pass, Band Pass
// 12 dB / octave
// arguments: in, freq, rq, mul, add
{ RLPF.ar(Saw.ar(200, 0.1), SinOsc.ar(XLine.kr(0.7, 300, 20), 0, 3600, 4000), 0.2) }.scope(1);
// a resonant low pass filter
{ RHPF.ar(Saw.ar(200, 0.1), FSinOsc.kr(XLine.kr(0.7, 300, 20), 0, 3600, 4000), 0.2) }.scope(1);
// a resonant high pass filter

// Resonz
// resonant band pass filter with uniform amplitude
// argument: in, frequency, rq, mul, add
{ Resonz.ar(WhiteNoise.ar(0.5), 2000, 0.1) }.scope(1);
// same as Ringz, except that it has a constant gain at 0 dB instead of being constant skirt
{ Resonz.ar(WhiteNoise.ar(0.5), XLine.kr(1000, 8000, 10), 0.05)}.scope(1);
{ Resonz.ar(WhiteNoise.ar(0.5), 2000, XLine.kr(1, 0.001, 8))}.scope(1);

// Ringz
// internally it is the same as Resonz but the bandwidth is expressed as ring time
// arguments: in, frequency, ring time(decaytime, the 60 dB decay time of the filter), mul, add
{ Ringz.ar(Dust.ar(3, 0.3), 2000, 2) }.scope(1);
{ Ringz.ar(WhiteNoise.ar(0.005), 2000, 0.5) }.scope(1);
// This is the same as Resonz , except that it is a constant skirt gain filter, meaning that the peak gain depends on the value of Q. Also, instead of the resonance parameter in Resonz, the bandwidth is specified in a 60dB ring decay time. One Ringz is equivalent to one component of the Klank UGen.
{ Ringz.ar(WhiteNoise.ar(0.005), XLine.kr(100, 3000, 10), 0.5) }.scope(1);
{ Ringz.ar(Impulse.ar(6, 0.3), 2000, XLine.kr(4, 0.04, 8)) }.scope(1);


// Simpler Filters
// arguments: in, coef, mul, add
{ OnePole.ar(WhiteNoise.ar(0.5), 0.95) }.scope(1);
{ OnePole.ar(WhiteNoise.ar(0.5), -0.95) }.scope(1);
// coef: Feedback coefficient. Should be between -1 and +1
// A one pole filter. Implements the formula:
// out(i) = ((1 - abs(coef)) * in(i)) + (coef * out(i-1)).
{ OnePole.ar(WhiteNoise.ar(0.5), Line.kr(-0.99, 0.99, 10)) }.scope(1);

{ OneZero.ar(WhiteNoise.ar(0.5), 0.5) }.scope(1);
{ OneZero.ar(WhiteNoise.ar(0.5), -0.5) }.scope(1);
// coef: Feed forward coefficient.
//+0.5 makes a two point averaging filter (see also LPZ1 ).
//-0.5 makes a differentiator (see also HPZ1 ).
//+1 makes a single sample delay (see also Delay1 ).
//-1 makes an inverted single sample delay.
// A one zero filter. Implements the formula:
// out(i) = ((1 - abs(coef)) * in(i)) + (coef * in(i-1)).
{ OneZero.ar(WhiteNoise.ar(0.5), Line.kr(-0.5, 0.5, 10)) }.scope(1);


// NonLinear Filters
{ Saw.ar(500, 0.1) + Dust2.ar(100, 0.9) }.scope(1);
{ Median.ar(3, Saw.ar(500, 0.1) + Dust2.ar(100, 0.9))}.scope(1);
{ Saw.ar(500, 0.1) + LPZ1.ar(Dust2.ar(100, 0.9)) }.scope(1);
{ Median.ar(5, Saw.ar(500, 0.1) + LPZ1.ar(Dust2.ar(100, 0.9)))}.scope(1);
// returns the median of the last length input points. This non-linear filter is good at reducing impulse noise from a signal
// arguments: length(Number of input points in which to find the median. Must be an odd number from 1 to 31), in, mul, add
(
{
	x = SinOsc.ar(1000, 0, 0.2);
	[x, Median.ar(31, x)];
}.scope(1);
)
// long Median filters begin chopping off the peaks of the waveform

{ Saw.ar(500, 0.1) + Dust2.ar(100, 0.9) }.scope(1);
{ Slew.ar(Saw.ar(500, 0.1) + Dust2.ar(100, 0.9), 1000, 1000) }.scope(1);
// Limits the slope of an input signal. The slope is expressed in units per second.
// arguments: in, up(Maximum upward slope in units per second), dn(Maximum downward slope in units per second), mul, add
(
{
	z = LFPulse.ar(800);
	[z, Slew.ar(z, 4000, 4000)]
}.plot
)


// Formant Filter
{ Formlet.ar(Impulse.ar(20, 0.5), 1000, 0.01, 0.1) }.scope(1);
{ Formlet.ar(Blip.ar(XLine.kr(10, 400, 8), 1000, 0.1), 1000, 0.01, 0.1) }.scope(1);
// a resonant filter whose impulse response is like that of a sine wave with a Decay2 envelope over it. It is possible to control the attacktime and decaytime.
// Formlet is equivalent to:
// Ringz(in, freq, decaytime) - Ringz(in, freq, attacktime)
// arguments: in, freq(resonant frequency in Hertz,
// attacktime: 60 dB attack time in seconds, decaytime: 60 dB decay time in seconds)
(
{
	var in;
	in = Blip.ar(SinOsc.kr(5, 0.20, 300), 1000, 0.1);
	Formlet.ar(in, XLine.kr(1500, 700, 8), 0.005, 0.04);
}.scope(1);
)
// modulating formant frequency

{ Klank.ar(`[[800, 1071, 1153, 1723], nil, [1, 1, 1, 1]], Impulse.ar(2, 0, 0.1)) }.scope(1);
{ Klank.ar(`[[800, 1071, 1153, 1723], nil, [1, 1, 1, 1]], Dust.ar(8, 0.1)) }.scope(1);
{ Klank.ar(`[[800, 1071, 1153, 1723], nil, [1, 1, 1, 1]], PinkNoise.ar(0.007)) }.scope(1);
{ Klank.ar(`[[800, 1071, 1153, 1723], nil, [1, 1, 1, 1]], PinkNoise.ar([0.007, 0.007])) }.scope(1);
// Klank is a bank of fixed frequency resonators which can be used to simulated the resonant modes of an object. Each mode is given a ring time, which is the time for the mode to decay by 60 dB.
/*
arguments:
specificationArrayRef(frequencies: an array of filter frequencies, amplitudes: an array of filter amplitudes, if nil then default ti 1.0, ring times: an Array of 60 dB decay times in seconds for the filters),
input,
freqscale(a scale factor multiplied by all frequencies at initialization time),
freqoffset(an offset added to all frequencies at initialization),
decayscale(a scale factor multiplied by all ring times at initialization time)
*/






