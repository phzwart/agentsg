
import sys, time, csv
import numpy as np
from agentsg.cell.metric import UnitCell, params_from_metric
from agentsg.cell.reduction import niggli_gk
from agentsg.cell.canonical import reindexing_via_canonical
from agentsg.lattice_symmetry import tolerance_metric_symmetry

def cell_to_cart(cell): return np.array(UnitCell(*cell).orthogonalization_matrix()).T
def perturb(cell,mag,rng):
    B=cell_to_cart(cell).copy()
    for i in range(3):
        v=rng.normal(size=3); v=v/np.linalg.norm(v)*mag; B[i]+=v
    return params_from_metric((B@B.T).tolist())
def rlcob(rng,base,me=2,el=5.0):
    while True:
        M=rng.integers(-me,me+1,size=(3,3)); d=round(np.linalg.det(M))
        if d not in (1,-1): continue
        Bp=M.T@np.array(UnitCell(*base).metric_tensor())@M
        try: p=params_from_metric(Bp.tolist())
        except: continue
        if max(p[:3])/min(p[:3])>el or any(a<20 or a>160 for a in p[3:]): continue
        return M.astype(int)
def idet(M):
    (a,b,c),(d,e,f),(g,h,i)=M; return a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)

def route_selling(co,cr,boundary_rel=1e-2,verify_rel=0.06):
    ops=reindexing_via_canonical(co,cr,boundary_rel=boundary_rel,verify_rel=verify_rel)
    return [np.array(P,dtype=int) for P in ops] if ops else []

def route_lepage(co,cr,delta_deg=3.0,len_tol=4.0):
    red_o,Mo=niggli_gk(co); red_r,Mr=niggli_gk(cr)
    Mo=np.array(Mo,dtype=float); Mr=np.array(Mr,dtype=float)
    Gro=Mo.T@np.array(UnitCell(*co).metric_tensor())@Mo; Mri=np.linalg.inv(Mr)
    Hops=tolerance_metric_symmetry(red_o,length_tol_pct=len_tol,angle_tol_deg=delta_deg)
    H=[np.eye(3,dtype=int)]+[np.array([[int(round(float(op.W.rows[i][j]))) for j in range(3)] for i in range(3)]) for op in Hops]
    def close(p,q):
        dl=max(abs(p[i]-q[i])/q[i]*100 for i in range(3)); da=max(abs(p[3+i]-q[3+i]) for i in range(3))
        return dl<=len_tol and da<=delta_deg
    seen=set(); Ps=[]
    for B in H:
        Gb=B.T@Gro@B
        if close(params_from_metric(Gb.tolist()),red_r):
            P=Mo@B@Mri; Pi=np.round(P).astype(int)
            if np.allclose(P,Pi,atol=1e-6) and abs(idet(Pi.tolist()))==1:
                k=tuple(Pi.flatten())
                if k not in seen: seen.add(k); Ps.append(Pi)
    return Ps

parent=(8.0,6.0,11.0,90.0,90.3,90.0)
G_parent=np.array(UnitCell(*parent).metric_tensor())
def is_correct(P,Msc,tol=0.09):
    Q=Msc@P; Gt=Q.T@G_parent@Q
    return np.abs(Gt-G_parent).sum()/np.abs(G_parent).sum()<tol

MAGS=[0.02,0.05,0.1,0.2,0.5]; N=300
rows=[]
t0=time.time()
for mag in MAGS:
    rng=np.random.default_rng(hash(("v2",mag))%(2**32))
    sc=sw=sm=lc=lw=lm=0
    for i in range(N):
        cr=perturb(parent,mag,rng); co=perturb(parent,mag,rng)
        Msc=rlcob(rng,co); cos=params_from_metric((Msc.T@np.array(UnitCell(*co).metric_tensor())@Msc).tolist())
        sP=route_selling(cos,cr); lP=route_lepage(cos,cr)
        if sP:
            if any(is_correct(P,Msc) for P in sP): sc+=1
            else: sw+=1
        else: sm+=1
        if lP:
            if any(is_correct(P,Msc) for P in lP): lc+=1
            else: lw+=1
        else: lm+=1
    rows.append(dict(mag=mag,n=N,
                     sell_correct=sc,sell_wrong=sw,sell_miss=sm,
                     lep_correct=lc,lep_wrong=lw,lep_miss=lm))
    with open("selling_vs_lepage.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"mag={mag}: SELL {sc}/{N} (wrong {sw}, miss {sm}) | LEP {lc}/{N} (wrong {lw}, miss {lm}) [{time.time()-t0:.0f}s]",flush=True)
print("DONE",time.time()-t0)
