<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 441
nchnls = 2
0dbfs = 1

instr 1
iAmp = p4
iFreq = p5
kPan line  0, p3, 1
aNote oscils iAmp, iFreq, 0
aL, aR pan2 aNote, kPan
outs aL, aR
endin

instr 2
iAmp = p4
iFreq = p5
kPan line  0, p3, 1
aNote oscils iAmp, iFreq, 0
aR, aL pan2 aNote, kPan
outs aL, aR
endin

</CsInstruments>
<CsScore>
i 1 0 2 0.8 441
i 2 2 2 0.8 441
i 1 4 2 0.8 441
i 2 6 2 0.8 441

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
