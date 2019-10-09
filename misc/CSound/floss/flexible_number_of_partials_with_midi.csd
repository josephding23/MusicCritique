<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1

giSine    ftgen     0, 0, 2^10, 10, 1
          massign   0, 1 ;all midi channels to instr 1

instr 1
ibasfreq cpsmidi
iampmid ampmidi 20
inumparts = int(iampmid) + 1
ipart = 1
loop:
ifreq = ibasfreq * ipart
iamp = 1/ipart/inumparts
	event_i "i", 10, 0, 1, ifreq, iamp
	loop_le ipart, 1, inumparts, loop
endin

instr 10
ifreq = p4
iamp = p5
aenv transeg 0, .01, 0, iamp, p3-.01, -3, 0
apart poscil aenv, ifreq, giSine
outs apart/3, apart/3
endin

</CsInstruments>
<CsScore>
f 0 3600
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
