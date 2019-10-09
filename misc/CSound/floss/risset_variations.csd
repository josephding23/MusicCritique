<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1

giFqs     ftgen     0, 0, -11,-2,.56,.563,.92, .923,1.19,1.7,2,2.74, \
                     3,3.74,4.07
giAmps    ftgen     0, 0, -11, -2, 1, 2/3, 1, 1.8, 8/3, 1.46, 4/3, 4/3, 1, 4/3
giSine    ftgen     0, 0, 2^10, 10, 1
          seed      0
          
instr 1
ibasfreq = 400
ifqdev = p4
iampdev = p5
idurdev = p6
indx = 0
loop:
ifqmult tab_i indx,  giFqs
ifreq = ibasfreq * ifqmult
iampmult tab_i indx, giAmps
iamp = iampmult / 20
event_i "i", 10, 0, p3, ifreq, iamp, ifqdev, iampdev, idurdev
loop_lt indx, 1, 11, loop
endin

instr 10
ifreqnorm = p4
iampnorm = p5
ifqdev = p6
iampdev = p7
idurdev = p8

icent random  -ifqdev, ifqdev
ifreq = ifreqnorm * cent(icent)

idb random -iampdev, iampdev
iamp = iampnorm * cent(idb)

idurperc random -idurdev, idurdev
iptdur = p3 * 2^(idurperc/100)
p3 = iptdur

aenv transeg 0, .01, 0, iamp, p3-.01, -10, 0
apart poscil aenv, ifreq, giSine
outs apart, apart
endin
</CsInstruments>
<CsScore>
;         frequency   amplitude   duration
;         deviation   deviation   deviation
;         in cent     in dB       in %
;;unchanged sound (twice)
r 2
i 1 0 5   0           0           0
s
;;slight variations in frequency
r 4
i 1 0 5   25          0           0
;;slight variations in amplitude
r 4
i 1 0 5   0           6           0
;;slight variations in duration
r 4
i 1 0 5   0           0           30
;;slight variations combined
r 6
i 1 0 5   25          6           30
;;heavy variations
r 6
i 1 0 5   50          9           100
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
