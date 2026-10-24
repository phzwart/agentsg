"""An uncertified lookup can miss an operator: for the 100 x 100.1 x 5 A plate (r_1 = 5 %,
below the 7.7 % that certifies a 7.4 % band), a new cell is built on a member of the
widened set outside the first layer; its true operator costs about 6.1 %, the exact
enumeration (indexing_classes) finds it and the lookup of U between the reduced cells
does not. Adapted from a mock referee's probe.

    PYTHONPATH=agentsg/src python manuscript/benchmark/lookup_plate_case.py
"""
import cctbx
import numpy as np, math, time, itertools
from agentsg.cell.metric import UnitCell, params_from_metric
from agentsg.cell import strain_closure as sc

def G_of(cell): return np.array(UnitCell(*cell).metric_tensor(), float)
def cell_of(G): return tuple(params_from_metric(np.asarray(G).tolist()))

# --- Selling reduction implemented from the paper alone (Table 1 flips) ---
F = {'ab': np.array([[-1,0,1],[0,1,0],[0,0,1]]),
     'ac': np.array([[-1,1,0],[0,1,0],[0,0,1]]),
     'bc': np.array([[1,0,0],[1,-1,0],[0,0,1]]),
     'da': np.array([[1,-1,-1],[0,0,-1],[0,-1,0]]),
     'db': np.array([[0,0,-1],[-1,1,-1],[-1,0,0]]),
     'dc': np.array([[0,-1,0],[-1,0,0],[-1,-1,1]])}
def conorms_basis(G):
    a,b,c = np.eye(3)
    d = -a-b-c
    v = {'d':d,'a':a,'b':b,'c':c}
    return {k: -(v[k[0]]@G@v[k[1]]) for k in ('da','db','dc','ab','ac','bc')}
def paper_reduce(G, tol=0.0, maxit=1000):
    B = np.eye(3, dtype=int)
    for _ in range(maxit):
        Gc = B.T@G@B
        p = conorms_basis(Gc)
        k = min(p, key=p.get)
        if p[k] >= -tol*abs(Gc).max(): return B
        B = B@F[k]
    raise RuntimeError("no convergence")

def price(GN, Gpred):
    w,V = np.linalg.eigh(GN); Gm = V@np.diag(w**-.5)@V.T
    l = np.log(np.linalg.eigvalsh(Gm@Gpred@Gm)); m=l.mean()
    return float(np.linalg.norm(l-m)), math.exp(-m)

U = sc.universal_operators()
def lookup(GR, GN, band=0.074, BR=None, BN=None, transpose=False, scale_lim=0.1):
    BR = paper_reduce(GR) if BR is None else BR
    BN = paper_reduce(GN) if BN is None else BN
    W = U.transpose(0,2,1) if transpose else U
    Ps = np.rint(np.einsum('ij,njk,kl->nil', BR, W, np.linalg.inv(BN))).astype(int)
    Ps = Ps[np.round(np.linalg.det(Ps))==1]
    out = {}
    for P in Ps:
        pr, s = price(GN, P.T@GR@P)
        if pr <= band and abs(math.sqrt(s)-1) <= scale_lim:
            out[tuple(P.flatten())] = pr
    return out

def rand_unimodular(rng, r=2, proper=True):
    while True:
        P = rng.integers(-r, r+1, (3,3))
        d = round(np.linalg.det(P))
        if d == 1 or (not proper and d == -1): return P

def perturb(G, strain, rng):
    w,V = np.linalg.eigh(G); Gh = V@np.diag(w**.5)@V.T
    X = rng.normal(size=(3,3)); X=(X+X.T)/2; X *= strain/np.linalg.norm(X)
    return Gh@(np.eye(3)+X)@Gh

def gram(p):  # conorms da,db,dc,ab,ac,bc
    return np.array([[p[0]+p[3]+p[4], -p[3], -p[4]],[-p[3], p[1]+p[3]+p[5], -p[5]],[-p[4],-p[5],p[2]+p[4]+p[5]]], float)

from scipy.optimize import nnls
import warnings; warnings.simplefilter("ignore")
_,L1,L2,K1,K2 = sc._data()
def nearest_obtuse_metric(G, S, push=1e-4):
    S=np.asarray(S,float)
    q=sc.conorms(S,G)
    Gf=S[1:]@G@S[1:].T
    M=sc._metric_M(Gf[None])[0]; R=np.linalg.cholesky(M).T
    x,_=nnls(R,R@q)
    x = x + push*np.abs(q).max()*(x<=1e-12)   # push into the interior
    Gp_frame = sc._gram(x)              # metric in the frame of S basis
    B=S[1:].T                           # columns: basis vectors of S in ref reduced coords
    Binv=np.linalg.inv(B)
    return Binv.T@Gp_frame@Binv, B

cell=(100,100.1,5,90.03,89.98,90.05)
red=sc.reduce_cell(cell); W=sc.widened_closure(red)
outside=[m for m in W.members if sc._key(m.superbase) not in K1]
m=outside[0]
Gp,B=nearest_obtuse_metric(red.G,m.superbase)
BR=np.array([[float(x) for x in r] for r in red.basis.rows])
GN=B.T@Gp@B; newcell=cell_of(GN)
Ptrue=np.rint(BR@B).astype(int)
if round(np.linalg.det(Ptrue))!=1: Ptrue=-Ptrue
print("new cell", [round(x,3) for x in newcell])
print("true P", Ptrue.tolist(), "price", round(price(GN, Ptrue.T@G_of(cell)@Ptrue)[0],4))
r=sc.indexing_classes(W,newcell,"P 1")
for i,c in enumerate(r.classes):
    ops=[np.array([[float(x) for x in rr] for rr in o.rows]).astype(int) for o in c.operators]
    tag = "TRUE" if any((o==Ptrue).all() for o in ops) else ""
    print(i, round(100*c.representative.price,2), ops[0].tolist(), tag)
lk=lookup(G_of(cell), GN, BR=BR, BN=np.array([[float(x) for x in rr] for rr in sc.reduce_cell(newcell).basis.rows]))
print("lookup finds true:", tuple(Ptrue.flatten()) in lk, " lookup n:", len(lk), " r1 of reference:", 0.0501, " layer", W.layer)
print("strain of the new cell relative to reference (geodesic, scale-fitted) = true price; Table 8 r1=5.0% < 7.7%")
