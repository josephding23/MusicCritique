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
kcps = 220
knh = p4
klh = p5
kmul line 0, p3, 1

asig gbuzz .6, kcps, knh, klh, kmul, 1
outs asig, asig
endin

</CsInstruments>
<CsScore>
f 1 0 16384 11 1

i 1 0 3 3 1
i 1 + 3 30 1
i 1 + 3 3 2
i 1 + 3 30 2
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
