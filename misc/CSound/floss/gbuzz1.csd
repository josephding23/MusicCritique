<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1

gicos ftgen 0, 0, 2^10, 11, 1

instr 1
knh line 1, p3, 20
klh = 1
kmul = 1
asig gbuzz 1, 100, knh, klh, kmul, gicos
outs asig, asig
endin

</CsInstruments>
<CsScore>
i 1 0 8
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
