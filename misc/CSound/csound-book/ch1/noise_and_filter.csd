<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1

instr 128 ; BANDPASS-FILTERED NOISE
idur = p3
iamp = p4
ifrq = p5
iatk = p6
irel = p7
icf1 = p8
icf2 = p9
ibw1 = p10
ibw2 = p11
kenv expseg .01, iatk, iamp, idur/6, iamp*.4, idur(iatk+irel+idur/6), iamp*.6, irel, .01
anoise rand ifrq ; Generates a controlled random number series
; ares rand xamp [, iseed] [, isel] [, ioffset]
kcf expon icf1, idur, icf2
kbw line ibw1, idur, ibw2
afilt reson anoise, kcf, kbw, 2 ; A second-order resonant filter
; ares reson asig, xcf, xbw [, iscl] [, iskip]
out afilt*kenv
display kenv, idur
endin

instr 133
idur = p3
iamp = ampdb(p4)
ifrq = p5
icut1 = p6
icut2 = p7
iresgn = p8
kcut expon icut1, idur, icut2
aplk pluck iamp, ifrq, ifrq, 0, 1
abpf butterbp aplk, kcut, kcut*.2 ; A band-pass Butterworth filter
; ares butterbp asig, xfreq, xband [, iskip]
alpf butterlp aplk, kcut ; A low-pass Butterworth filter
; ares butterlp asig, kfreq [, iskip]
amix = alpf + (abpf*iresgn)
out amix
dispfft amix, idur, 1024
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
