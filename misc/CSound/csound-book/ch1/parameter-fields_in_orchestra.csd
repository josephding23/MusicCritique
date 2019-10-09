<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
kr = 4410
ksmps = 10
nchnls = 1

instr 107
aOs oscil p4, p5, p6
out aOs
endin

instr 108
aFos foscil p4, p5, p6, p7, p8, p9
out aFos
endin

</CsInstruments>
<CsScore>

; P1 					P2 										P3         P4				P5				P6
; INS # 		START-TIME 		DURATION		 AMP			FREQ		WAVESHAPE
i 107 0 1 10000 440 1
i 107 1.5 1 20000 220 2
i 107 3 3 10000 110 2
i 107 4 2 5000 329.6 2
i 107 4.5 1.5 6000 440 2

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
