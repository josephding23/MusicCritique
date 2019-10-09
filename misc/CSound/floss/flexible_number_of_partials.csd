<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1

giSine ftgen 0, 0, 2^10, 10, 1

instr 1
inumparts = p4
ibasfreq = 200
ipart = 1
loop:
ifreq = ibasfreq * ipart
iamp = 1/ipart/inumparts
event_i  "i", 10, 0, p3, ifreq, iamp
loop_le ipart, 1, inumparts, loop
endin

instr 10
ifreq = p4
iamp = p5
aenv transeg 0, .01, 0, iamp, p3-0.1, -10, 0
apart poscil aenv, ifreq, giSine
outs apart, apart
endin

</CsInstruments>
<CsScore>
;         number of partials
i 1 0 3   10
i 1 3 3   20
i 1 6 3   2
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
