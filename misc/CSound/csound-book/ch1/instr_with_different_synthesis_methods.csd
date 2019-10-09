<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 10
nchnls = 2
0dbfs = 1

instr 101			; A simple oscilator
a1 oscil 10000, 440, 1	
; ares oscil xamp, xcps [, ifn, iphs]
out a1
endin

instr 102			; A basic frequency modulated oscillator
a1 foscil 10000, 440, 1, 2, 3, 1
; ares foscil xamp, kcps, xcar, xmod, kndx, ifn [, iphs]
out a1
endin

instr 103 ; Output is a set of harmonically related sine partials. 
a1 buzz 10000, 440, 10, 1
; ares buzz xamp, xcps, knh, ifn [, iphs]
out a1
endin

instr 104 ; Produces a natually decaying plucked string or drum sound
a1 pluck 10000, 440, 440, 2, 1
; ares pluck kamp, kcps, icps, ifn, imeth [, iparm1] [, iparm2]
out a1
endin

instr 105 ; Generates granular synthesis textures
a1 grain 10000, 440, 55, 10000, 10, .05, 1, 3, 1
; ares grain xamp, xpitch, xdens, kampoff, kpitchoff, kgdur, igfn, iwfn, imgdur [, igrnd]
out a1
endin

instr 106 ; Read ampled sound from a table
a1 loscil 10000, 440, 4
; ar1 [,ar2] loscil xamp, kcps, ifn [, ibas] [, imod1] [, ibeg1] [, iend1] [, imod2] [, ibeg2] [, iend2]
out a1
endin

</CsInstruments>
<CsScore>

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
