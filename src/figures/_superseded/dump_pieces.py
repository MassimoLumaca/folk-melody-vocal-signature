#!/usr/bin/env python3
"""Reproduce the SymbTr per-piece transferred scores (identical logic to symbtr.py) and dump to CSV."""
import os, csv, json
import numpy as np
from collections import defaultdict

VOCAL={"sarki","turku","ilahi","beste","yuruksemai","agirsemai","fantezi",
"rumeliturkusu","nakis","murabba","kar","kanto","nefes","divan","popsarkisi",
"ninni","mersiye","durak","destan","cocuksarkisi","bozlak","kosma","kalenderi",
"kar_i_natik","kar_i_nev","karce","sugul","tevsihilahi","tesbih","tekbir",
"salatuselam","salatiummiye","miraciye","selam","guvende"}
INSTR={"pesrev","sazsemaisi","longa","sirto","sazeseri","oyunhavasi","medhal",
"mandra","kasaphavasi","kurthavasi"}
FLAGGED={"aranagme","zeybek","kocekce","mars","mehter","karsilama","tavsanca","etud"}
EXCL={"seyir","kupe","ornek_oz"}

d="symdata/symbtr_txt"
pieces=[]
for fn in sorted(os.listdir(d)):
    form=fn.split('--')[1]
    if form in EXCL: continue
    grp='vocal' if form in VOCAL else 'instrumental' if form in INSTR else \
        'flagged' if form in FLAGGED else None
    if grp is None: continue
    pitch,onset,dur=[],[],[]; t=0.0
    try:
        for r in csv.DictReader(open(os.path.join(d,fn),encoding='utf-8'),delimiter='\t'):
            if r['Kod']=='51': continue
            try: koma=int(r['Koma53'])
            except: continue
            try:
                b=4*float(r['Pay'])/float(r['Payda']) if r['Payda'] not in ('','0') else 0.0
            except: b=0.0
            if koma==-1 or r['Nota53']=='Es':
                t+=b; continue
            pitch.append(koma); onset.append(t); dur.append(b); t+=b
    except Exception:
        continue
    if len(pitch)<3: continue
    p=np.array(pitch,float); o=np.array(onset); du=np.array(dur)
    ivl=np.diff(p); ttrans=len(p)-1; rest=o[1:]-(o[:-1]+du[:-1])
    pieces.append(dict(form=form,group=grp,n=len(p),
        rep=float((ivl==0).sum()/ttrans),
        rng=float((p.max()-p.min())*12/53),
        gap=float((rest>=0.5-1e-9).sum()/ttrans)))
for k in ['rep','rng','gap']:
    v=np.array([x[k] for x in pieces]); mu,sd=v.mean(),v.std(ddof=1)
    for x in pieces: x['z'+k]=(x[k]-mu)/sd
for x in pieces:
    x['core']=x['zrep']-x['zrng']; x['aug']=x['core']+x['zgap']

with open("symbtr_pieces.csv","w",newline='') as f:
    w=csv.DictWriter(f,fieldnames=['form','group','n','rep','rng','gap','zrep','zrng','zgap','core','aug'])
    w.writeheader()
    for x in pieces: w.writerow(x)

# sanity: reproduce piece-level continuity AUC
def auc(a,b):
    a=np.asarray(a)[:,None]; b=np.asarray(b)[None,:]
    return float(((a>b).sum()+0.5*(a==b).sum())/(a.size*b.size))
pv=[x for x in pieces if x['group']=='vocal']; pi=[x for x in pieces if x['group']=='instrumental']
print("n_pieces",len(pieces),"vocal",len(pv),"instr",len(pi))
print("piece auc continuity(zgap)", round(auc([x['zgap'] for x in pv],[x['zgap'] for x in pi]),4))
print("piece auc core", round(auc([x['core'] for x in pv],[x['core'] for x in pi]),4))
