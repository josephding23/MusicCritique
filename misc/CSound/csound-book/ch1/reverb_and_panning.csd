<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1

gacmb init 0
garvb init 0

instr 137  ; GLOBAL COMB/VERB LOSCIL
idur = p3
iamp = ampdb(p4)
ifrq = cpspch(p5)
ifun = p6
iatk = p7
irel = p8
irvbsnd = p9
icmbsnd = p10
kenv linen iamp, iatk, idur, irel
asig loscil kenv, ifrq, ifun
out asig
garvb = garvb + (asig*irvbsnd)
gacmb = gacmb + (asig*icmbsnd)
endin

instr 198  ; GLOBAL ECHO
idur = p3
itime = p4
iloop = p5
kenv linen 1, .01, idur, .01
acomb comb gacmb, itime, iloop, 0
out acomb*kenv
gacmb = 0
endin

instr 199  ; GLOBAL REVERB
idur = p3
irvbtim = p4
ihiatn = p5
arvb nreverb garvb, irvbtim, ihiatn
out arvb
garvb = 0
endin

instr 138  ; SWEEPING FM WITH VIBRATO & DISCRETE PAN
idur = p3
iamp = ampdb(p4)
ifrq = cpspch(p5)
ifc = p6
ifm = p7
iatk = p8
irel = p9
indx1 = p10
indx2 = p11
indxtim = p12
ilfodep = p13
ilfofrq = p14
ipan = p15
irvbsnd = p16
kampenv expseg .01, iatk, iamp, idur/9, iampp*.6, idur/(iatk+irel+idur/9), iamp*.7, irel, .01
klfo oscil ilfodep, ilfofrq, indx2
asig foscil kampenv, ifrq+klfo, ifc, ifm, kindex, 1
outs asig*ipan, asig*(1-ipan)
garvb = garvb + (asig*irvbsnd)
endin

instr 141  ; AMPLITUDE MODULATION LFO PANNER
idur = p3
iamp = ampdb(p4)
ifrq = cpspch(p5)
ifun = p6
iatk = p7
irel = p8
iatkfun = p9
imodpth = p10
imodfrq = p11
imodfun = p12
ipanfrq = p13
irvbsnd = p14
kenv envlpx iamp, iatk, idur, irel, iatkfun, .7, .01
kpan oscil .5, ipanfrq, 1
klfo oscil imodpth, imodfrq, imodfun
asig oscil klfo*kenv, ifrq, ifun
kpanlfo = kpan+.5
outs asig*kpanlfo, asig*(1-kpanlfo)
garvb = garvb + (asig*irvbsnd)
endin


</CsInstruments>
<CsScore>

; INS  STRT  DUR  RVBTIME     HFROLL
i 199  0     12   4.6         .8

; INS  STRT  DUR  TIME        LOOPT  
i 198  0     6    10          .8
i 198  0     6    10          .3
i 198  0     6    10          .5

; INS  STRT  DUR  AMP   FRQ1  SAMPLE  ATK  REL  RVBSND  CMBSND 
i 137  0     2.1  70    8.09  5       .01  .01  .3      .6
i 137  1     2.1  70    8.09  5       .01  .01  .5

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
