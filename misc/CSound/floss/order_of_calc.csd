<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 4410
nchnls = 2
0dbfs = 1

instr 1
gkcount init 0
gkcount = gkcount + 1
endin

instr 10
printk 0, gkcount
endin

instr 100

gkcount init 0
gkcount = gkcount +1
endin


</CsInstruments>
<CsScore>
i 1 01
i 10 0 1
i 100 1 1 
i 10 1 1

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
