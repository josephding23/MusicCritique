<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1

instr 113					; SIMPLE OSCIL WITH ENVELOP
k2 linen p4, p7, p3, p8		; Straight line rise and decay pattern
; ares linen xamp, irise, idur, idec
a2 oscil k1, p5, p6
out a1
endin

instr 115   ; SWEEPING BUZZ WITH ENVELOPE
k1 linen p4, p7, p3, p8
k3 expon p9, p3, p10 ; Trace an exponential curve between two points
; ares expon ia, idur, ib
a1 buzz 1, p5, k2+1, p6
out k1*a1
endin

instr 117 ; GRANS THROUGH AN ENVELOP
k2 linseg p5, p3/2, p9, p3/2, p5 ; Trace a series of line segments between specified points
; ares linseg ia, idur1, ib [, idur2] [, ic] [...]
k3 line p10, p3, p11
k4 line p12, p3, p13
k5 expon p14, p3, p15
k6 expon p16, p3, p17
a1 grain p4, k2, k3, k4, k5, k6, 1, p6, 1
a1 linen a1, p7, p3, p8
out a2
endin 

instr 118  ; LOSCIL WITH OSCIL ENVELOPE
k1 oscil p4, 1/p3, p7
k2 expseg p5, 1/p3, p7
k2 expseg p5, p3/3, p8, p3/3, p9, p3/3, p5
a1 loscil k1, k2, p6
out a1
endin

instr 119  ; RETRIGGERING FOSCIL WITH OSCIL ENVELOPE
k1 oscil p4, 1/p3 * p8, p7
k2 line p11, p3, p12
a1 foscil k1, p5, p9, p10, k2, p6
out a1
endin

</CsInstruments>
<CsScore>

f 6 0 1024 7 0 10 1 1000 1 14 0		; LINEAR AR ENVEPOLE
f 7 0 1024 7 0 128 1 128 .6 512 .6 256 0		; LINEAR ADSR ENVELOPE
f 8 0 1024 5 .001 256 1 192 .5 256 .5 64 .001 ; EXPONENTIAL ADSR

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
