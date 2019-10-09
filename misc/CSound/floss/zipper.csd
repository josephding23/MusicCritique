<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
nchnls = 2
0dbfs = 1

instr 1
aSine oscils .5, 800, 0
kEnv transeg 0, .1, 5, 1, .1, -5, 0
aOut = aSine * kEnv
outs aOut, aOut
endin

instr 2
aSine oscils .5, 800, 0
aEnv transeg 0, .1, 5, 1, .1, -5, 0
aOut = aSine * aEnv
outs aOut, aOut
endin

</CsInstruments>
<CsScore>
r 5
i 1 0 1
s 
r 5
i 2 0 1
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
