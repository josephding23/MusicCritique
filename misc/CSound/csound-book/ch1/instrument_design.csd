<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1

instr	120  ; SIMPLE CHORUSING
idur = p3
iamp = ampdb(p4) ; returns the amplitude equivalent of the decibel value x
ifrq = cpspch(p5) ; converts a pitch-class value to cycles-per-second
ifun = p6
iatk = p7
irel = p8
iatkfun = p9
kenv envlpx iamp, iatk, irel, iatkfun, .7, .01 ; Applies an envelope consisting of 3 segments
; ares envlpx xamp, irise, idur, idec, ifn, iatss, iatdec [, ixmod]
asig3 oscil kenv, ifrq*.99, ifun ; SYNTHESIS BLOCK
asig2 oscil kenv, ifrq*1.01, ifun
asig1 oscil kenv, ifrq, ifun
amix = asig1 + asig2 + asig3
out amix
display kenv, idur
endin

instr 122 ; SIMPLE SPECTUAL FUSION
idur = p3
iamp = ampdb(p4)
ifrq = cpspch(p5)
ifun = p6
iatk = p7
irel = p8
iatkfun = p9
index1 = p10
index2 = p11
kenv envlpx iamp, iatk, idur, irel, iatkfun, .7, .01
kmodswp expon index1, idur, index2
kbuzswp expon 20, idur, 1
asig3 foscil kenv, ifrq, 1, 1, kmodswp, ifun
asig2 buzz kenv, ifrq*.99, kbuzswp+1, ifun
asig1 pluck iamp, ifrq+.5, ifrq, ifun
amix = asig1 + asig2 + asig3
out amix
dispfft amix, .25, 1024
endin

instr 124 ; SWEEPING AMPLITUDE MODULATION	
idur = p3
iamp = ampdb(p4)
ifrq = cps2pch(p5)
ifun = p6
iatk = p7
irel = p8
iatkfun = p9
imodp1 = p10
imodp2 = p11
imodfr1 = p12
imodfr2 - p13
imodfun = p14
kenv envlpx iamp, iatk, idur, irel, iatkfun, .7, .01
kmodpth expon imodp1, idur, imodp2
kmodfrq line cpspch(imodfr1), idur, cpspch(imodfr2)
alfo oscil kmodpth, kmodfrq, imodfun
asig oscil alfo, ifrq, ifun
out asig*kenv
endin

instr 126 ; SIMPLE DELAYED VIBRATO
idur = p3
iamp = ampdb(p4)
ifrq = cpspch(p5)
iatk = p6
irel = p7
ivibdel = p8
imoddpt = p9
imodfrq = p10
ihram = p11
kenv linen iamp, iatk, idur, irel
kvibenv linseg 0, ivibdel, 1, idur-ivibdel, .3
klfo oscil kvibenv*imoddpt, imodfrq, 1
asig buzz kenv, ifrq+klfo, iharm, 1
out asig
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
