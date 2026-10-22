# The per-slot theorem: what is proved, what is refuted, what is open

Companion to Appendix A of *The Selling closure*; checked by
`manuscript/appendix_z/test_z57_perslot_bands.py`.

## 1. Setting

Per-slot bands are frame-covariant. A superbase $F$ with vectors $u_0,\dots,u_3$ has on each slot the band

$$\tau_{ij}(F)=\sigma\,|u_i|\,|u_j|,$$

and the widened set is

$$S_\sigma(G)=\{F:\ p_{ij}(F;G)\ge-\sigma|u_i||u_j|\ \ \forall ij\},$$

the superbases all of whose pairwise angles are at least $90^\circ-\delta$ with $\sigma=\sin\delta$ ("$\delta$-obtuse"). The exact closure is the case $\delta=0$.

Construction B (Appendix B, `test_z49`): with $\ell_k=|v_k|$ the lengths of the reduced superbase $F_0$, idealise the slots $Z_0=\{kl:\ p^0_{kl}\le 4\sigma\ell_k\ell_l\}$, enumerate $\mathrm{Cl}(\Lambda^*)$, keep the members with $p_{ij}(F;G)\ge-\sigma|u_i||u_j|$. Stated condition: $\lambda_1>32\,\sigma\max_{kl\in Z_0}\ell_k\ell_l$.

## 2. Refuted: construction B under its stated condition

**Proposition 1.** Construction B is not exact under its stated condition.

*Witness.* The needle $a=5.58$, $b=5.21$, $c=131.97$ Å, $\gamma=91.12^\circ$, i.e. conorms $(p_{da},p_{db},p_{dc},p_{ab},p_{ac},p_{bc})=(30.61,26.58,17414.7,0.568,0,0)$ Å², $\lambda_1=27.15$ Å². Take $\sigma$ with $\lambda_1/(\sigma\ell_a\ell_c)=33$ ($\delta=0.064^\circ$). Then $Z_0=\{ac,bc\}$, the condition holds, construction B returns 6 superbases, and the brute-force $S_\sigma$ over the window has 12. The six missed members contain $a+c$ and $b$; their slot $(a+c,b)$ has conorm $-(a+c)\cdot b=p^0_{ab}+p^0_{bc}=0.568$ under $\Lambda^*$ but $-0.568$ under $G$ for the sign choice that is missed, and band $\sigma|a+c||b|\approx 25\,\sigma\ell_a\ell_b$. The slot $ab$ of $F_0$ survives idealisation because its *own* band $\sigma\ell_a\ell_b=0.034$ is small, but the same dot product is read in a member frame with a band 25 times larger. Exactness returns only when $\sigma|a+c||b|<p^0_{ab}$, i.e. $\lambda_1/(\sigma\ell_a\ell_c)\gtrsim45$. $\square$

*Mechanism.* The uniform theorem never meets this because a single $\tau$ means the same thing in every frame. With per-slot bands the idealisation threshold of an $F_0$ slot must be the largest band that the dot product it represents inherits in any member frame, not the band of the slot in $F_0$'s frame. Appendix B's sentence "returned the brute-force set in all 44 cases" is a sample, not a theorem, and the construction behind it is wrong on cells with a small surviving short–short conorm next to a long axis.

## 3. Proved: the per-slot set through a uniform hub

**Theorem A6′.** Let $F_0$ be obtuse for $G$ and let $\hat\tau$ be any number such that

(a) $\lambda_1>32\hat\tau$, and
(b) every negative slot of every $F\in S_\sigma(G)$ has band $\sigma|u_i||u_j|\le\hat\tau$.

Let $\Lambda^*$ be $F_0$ idealised at $4\hat\tau$ (uniformly). Then

$$S_\sigma(G)=\{F\in\mathrm{Cl}(\Lambda^*):\ p_{ij}(F;G)\ge-\sigma|u_i||u_j|\ \forall ij\}.$$

*Proof.* By (b), every $F\in S_\sigma(G)$ has all conorms $\ge-\hat\tau$, so $S_\sigma(G)\subseteq S_{\hat\tau}(G)$. By (a) and Theorem A6, $S_{\hat\tau}(G)=\{F\in\mathrm{Cl}(\Lambda^*):p_{ij}(F;G)\ge-\hat\tau\}$. Hence $S_\sigma(G)=S_\sigma(G)\cap S_{\hat\tau}(G)=\{F\in\mathrm{Cl}(\Lambda^*):\ p\ge-\hat\tau\text{ and }p\ge-\sigma|u_i||u_j|\}$. For $F\in\mathrm{Cl}(\Lambda^*)$ passing the per-slot filter, $F\in S_\sigma(G)$, so by (b) its negative slots have band at most $\hat\tau$ and the condition $p\ge-\hat\tau$ is automatic. The converse inclusion is the definition. $\square$

This is rigorous and settles the *filter*: once the idealisation is right, filtering per slot is exact. It does not settle needles, because (a) and (b) together ask for a uniform $\hat\tau$ above the band of every slot that can be negative, and for a needle that band is $\sigma\ell_{\rm long}\ell_{\rm short}$, which exceeds $\lambda_1/32=\ell_{\rm short}^2/32$ as soon as $\sigma>\ell_{\rm short}/(32\,\ell_{\rm long})$. On 160 random cells the hypothesis held in 31 (all exact).

## 4. The construction that is never wrong: frame-consistent idealisation

**Definition (fixpoint).** Start from $Z_0$. Given $Z$, let $\Lambda^*_Z$ be the idealised lattice and $\mathrm{Cl}(\Lambda^*_Z)$ its closure. For every member $F$ and every slot $ij$ of $F$ with

$$0<p_{ij}(F;\Lambda^*_Z)\le 4\,\sigma|u_i||u_j|,$$

add to $Z$ every surviving slot $kl$ of $F_0$ that feeds that conorm, i.e. with $\Delta_{kl}(u_i)\Delta_{kl}(u_j)\ne0$ in superbase coordinates, and $p^0_{kl}\le4\sigma|u_i||u_j|$. Repeat until nothing is added. $Z$ only grows, so this stops in at most six rounds. Refuse if $\Lambda^*_Z$ becomes singular. Then filter per slot.

At the fixpoint, **in every member frame every surviving conorm exceeds four times the band of the slot it is read in** — the frame-covariant form of the uniform theorem's "$m^*>4\tau$".

**Checked** (`z57`; all re-checked against brute force over coordinates in $[-3,3]$): on the needle it adds $ab$ at round 2 and returns 12 of 12 at every $\sigma$ tried. On 240 random cells (needles, plates, isotropic, monoclinic; $\delta=0.1$–$3^\circ$) it is never wrong: exact in 124, refused in 116 (plates and large $\delta$, where $S_\sigma$ has up to 86 members and is not a sub-closure of any single lattice); construction B is wrong in 9 of the 240.

**Conjecture A6″.** If the fixpoint terminates with $\Lambda^*_Z$ positive definite and $\lambda_1>32\,\sigma\max_{kl\in Z}\ell_k\ell_l$, the consistent construction is exact. (48/48 cases with the condition satisfied were exact; 76 further cases were exact beyond it; no failure.)

## 5. What a proof of A6″ needs

Follow Theorem A6. Let $F\in S_\sigma(G)\setminus\mathrm{Cl}(\Lambda^*)$. Two facts carry over unchanged:

* $e_c(F;\Lambda^*)\le e_c(F;G)$ in every parity class (Lemma A2 in $F_0$'s frame, dropping non-negative terms).
* In $F$'s own frame, with $N$ its negative slots and $P$ its positive slots and $\gamma'$ the integers of Lemma A2,

$$e_c(F;G)\le U_c(F):=4\Big[\sigma\sum_{N}|u_i||u_j|\,\gamma'_{ij}-\sum_{P}p_{ij}(F)\,\gamma'_{ij}\Big].$$

The lower bound is where the frames enter. In the uniform proof one reads Lemma A3 in $F_0$'s frame: some class has a surviving $\gamma_{kl}\ge1$, so $e_c(F;\Lambda^*)\ge4p^0_{kl}>16\tau$. With per-slot bands $p^0_{kl}$ may be tiny (the needle's $p^0_{ab}$), and the chain $U_c<4p^0_{kl}$ fails — that is Proposition 1 in proof form (`reduction_check.py` evaluates both sides; whenever $U_c<4\min_{\rm surv}p^0$ held, the construction was exact, and it failed exactly on the counterexample).

The fixpoint supplies the missing ingredient: Lemma A3 may be read in the frame of *any* member $F'\in\mathrm{Cl}(\Lambda^*)$, since $F'$ is obtuse for $\Lambda^*$, and there the surviving conorm with $\gamma^{F'}_{kl}\ge1$ satisfies $p_{kl}(F';\Lambda^*)>4\sigma|u'_k||u'_l|$. The inequality to prove is therefore

$$\sigma\sum_{N}|u_i||u_j|\gamma'_{ij}-\sum_{P}p_{ij}(F)\gamma'_{ij}\ <\ \sigma\,|u'_k||u'_l| \cdot 4 \qquad(\ast\ast')$$

for a suitable member $F'$ and the slot $kl$ of $F'$ that $F$ violates — a comparison of vector lengths between adjacent frames, with $\sum_N\gamma'\le4$ from Lemma A5(i) once the large $\gamma'$ are shown to vanish. The natural choice is the member $F'$ sharing the two vectors of $F$'s most negative slot. This is the weighted form of Lemma A5; it is a finite statement over the window (Lemma A0 gives $|k_i|\le3$ for $F$ once $(\ast\ast')$ controls the excess, as in the uniform case), with the lengths entering as continuous parameters, so it is an optimisation per window pattern rather than an enumeration. Proving it, or finding the configuration where it fails, is the open item.

## 6. Consequences for the paper and the pipeline

1. Appendix B must state construction B's status correctly: a sample of exact cases, with the needle above as the counterexample, and point to the consistent construction.
2. The pipeline (`members_by('slot', …)` in the benchmark, item 7 of the census fixes) should use the consistent idealisation. It refuses more often than B, but it is never wrong in any case tried, which is the property the paper claims.
3. Theorem A6′ (Section 3) can go into Appendix A as proved; Conjecture A6″ is stated as checked, with `z57`.

## 7. Window and the 3° rule

* At a 3° idealisation ($\sigma=\sin3^\circ/4$) on 300 random cells: exact on every cell up to 3:1; refused on 25 of 56 at 5:1 and 37 of 41 at 10:1; never wrong (`z57`).
* Against $[-3,3]$ brute force, the 362 non-refused cases of both sets were exact; one needed a coordinate of 3.
* The $[-2,2]$ window is complete for exact closures (Lemma A0) and within the hypothesis of Theorem A6, but not beyond it: a refused 7.73 × 47.54 × 77.30 Å cell with β = 87.0° has 11 δ-obtuse superbases in $[-2,2]$ and 12 in $[-3,3]$ (`z57`).
* Ties never change membership (Proposition A9); they enter at selection, $|\Delta p|\le2\varepsilon\max(p)$ on exchanged slots.

## 8. Zeroing on the closure, and the zeroing filtration

The fixpoint of Section 4 is now done during the enumeration
(`closure_construct` in `z57`): at every member reached, a slot with
$0<p_{ij}(F;\Lambda^*)\le4\sigma|u_i||u_j|$ zeroes the surviving $F_0$ slots that
feed it; $\Lambda^*$ is updated and the members found so far are examined again.
By Lemma A8 nothing found under fewer zeros is lost. On 543 cells it returns the
same sets as the rounds (Z differed only in two refused cases).

In cosine units the threshold is one number $t$ for every frame, so the zeroing is
a filtration (`zero_filtration`): events in increasing $t$, the closure constant
between them, up to the **geometric limit** at which $\Lambda^*$ would become
singular. Results on the 240 random cells of z57:

* stopped at $t=4\sigma$, never wrong; the canonical order refuses 6 cases that
  the unordered version answers (order dependence through the subtraction);
* enumerated once up to the limit and filtered at $\sigma$: equal to brute force
  at every band with $4\sigma$ below the limit;
* beyond the limit, filtering the top of the filtration misses members in 79 of
  178 refused cases (checked against $[-3,3]$), so the limit is a real refusal
  boundary, not a conservative one;
* geometric limit over the 543 cells: tenth percentile $0.3^\circ$, median
  $6.4^\circ$, ninetieth percentile $26^\circ$.
