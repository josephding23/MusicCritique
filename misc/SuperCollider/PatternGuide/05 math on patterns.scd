p = Pwhite(1, 5, inf) * 2；
p.operator
p.a
p.b
// Created a binary operator pattern

Pwhite(1, 5, inf).asStream.nextN(10);
(Pwhite(1, 5, inf) * 2).asStream.nextN(10);
(Pwhite(1, 5, inf) * 0.25).asStream.nextN(10);
(Pwhite(1, 5, inf) * Prand(#[-1, 1], inf)).asStream.nextN(10);

(Pseq([10, 9, 8], 1) + Pseq([1, 2], 1)).do { |x| x.postln};
//Every time a value is requested from the resulting stream, both of the component streams are asked for a value, and the operator applies to those results. If either stream ends, the binary operator stream also ends.

(
p = Pbind(
	\midinote, Pwhite(48, 72, inf) + .x Pseq(#[0, 4, 7, 11], 1),
	\dur, 0.125
).play;
)
//.x is like a nested loop: in streamA + .x streamB, the first value of streamA is added to every value of streamB in succession, then the second value of streamA is added to every streamB value, and so on.

Pwhite(0, 255, 20).collect({ |x| x.asHexString(2) }).do {|x| x.postln};
//Applies the function to each return value from the pattern.
Pwhite(0, 255, 20).select({ |x| x.odd}).do {|x| x.postln};
//Preserve values from the output stream that pass the Boolean test; discard the rest
Pwhite(0, 255, 20).reject({ |x| x.odd}).do {|x| x.postln};
//Discard values from the output stream that pass the test; return the rest

Pwhite(0, 255, 20).clump(4).do{ |x| x.postln};
// clump gets n values from the pattern at once and returns all of them as an array.
Array.fill(5, {Array.fill(4, {rrand(1, 5)})});
// Return a two-dimensional array
p = Pseq(Array.fill(5, {Array.fill(4, {rrand(1, 5)})}), 1);
// A pattern reading that array in sequence
p.do {|x| x.postln};
p.flatten.do{ |x| x.postln};
// the reverse of clump, returns a one-dimensional stream of numbers

Pseries(1, 1, 20).drop(5).do{ |x| x.postln};
// Discard the first n values, and return whatever is left.

Array.geom(20, 1, 1.01).differentiate;
Pgeom(1, 1.01, 20).differentiate.do{ |x| x.postln};
// Return the difference between succesive values



