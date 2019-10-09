<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1

instr 1
kamp = 24
kfreq expseg p4, p3/2, 50*p4, p3/2, p4
iloopnum = p5
alyd1 init 0
alyd2 init 0
seed 0
kfreqmult oscili 1, 2, 1
kosc oscili 1, 2.1, 1
ktone randomh 0.5, 2, 0.2
icount = 1

loop:
kfreq = kfreqmult * kfreq
atal oscili 1, 0.5, 1
apart oscili 1, icount * exp(atal*ktone), 1
anum = apart*kfreq*kosc

asig1 oscili kamp, anum, 1
asig2 oscili kamp, 1.5*anum, 1
asig3 oscili kamp, 2*anum, 1
asig4 oscili kamp, 2.5*anum, 1

alyd1 = (alyd1 + asig1 + asig4)/icount
alyd2 = (alyd2 + asig2 + asig3)/icount

loop_lt icount, 1, iloopnum, loop
outs alyd1, alyd2
endin

</CsInstruments>
<CsScore>
f1 0 128 10 1
i1 0 60 110 50
e
</CsScore>
</CsoundSynthesizer>
<bsbPanel>
 <label>Widgets</label>
 <objectName/>
 <x>100</x>
 <y>100</y>
 <width>320</width>
 <height>240</height>
 <visible>true</visible>
 <uuid/>
 <bgcolor mode="nobackground">
  <r>255</r>
  <g>255</g>
  <b>255</b>
 </bgcolor>
</bsbPanel>
<bsbPresets>
</bsbPresets>
