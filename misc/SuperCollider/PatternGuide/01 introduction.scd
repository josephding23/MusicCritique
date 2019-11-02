a = Routine {
	var i = 0;
	loop {
		i.yield;
		i = i + 1;
	}
};

a.nextN(10);

a = Routine {
	(0..).do { |i|
		i.yield;
	};
};

a.nextN(10);

a = Pseries(start: 0, step: 1, length: inf).asStream;
a.nextN(10);

p = Pseries(0, 1, 10);
p.next;

q = p.asStream;
q.next;


