# gen_tabular_no_preset.py
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd

def _stdz(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-9
    return (X - mu) / sd

# ---------- Dense-Hard：中低维（D<=1000） ----------
def make_tabular_hard(n=96000, d=200, anom_ratio=0.03, noise=0.02, seed=42):
    r = np.random.default_rng(seed)
    n_anom = max(1, int(n * anom_ratio)); n_norm = n - n_anom

    # 两条更紧的同心环（前2维）
    a1 = r.uniform(0, 2*np.pi, size=n_norm//2)
    a2 = r.uniform(0, 2*np.pi, size=n_norm - a1.size)
    r_inner, gap = 2.0, 2.2
    r_outer = r_inner + gap
    r1 = r.normal(r_inner, 0.025, size=a1.size)
    r2 = r.normal(r_outer, 0.035, size=a2.size)
    ring1 = np.stack([r1*np.cos(a1), r1*np.sin(a1)], axis=1)
    ring2 = np.stack([r2*np.cos(a2), r2*np.sin(a2)], axis=1)
    X2 = np.vstack([ring1, ring2])

    # 其余维：更紧的双高斯
    rest = max(0, d-2)
    if rest>0:
        cov = np.diag(r.uniform(0.18, 0.6, size=rest))
        g1 = r.multivariate_normal(np.zeros(rest), cov, size=X2.shape[0]//2)
        g2 = r.multivariate_normal(np.ones(rest)*1.0, cov, size=X2.shape[0]-g1.shape[0])
        Xn = np.hstack([X2, np.vstack([g1, g2])])
    else:
        Xn = X2
    Xn += r.normal(0, noise, size=Xn.shape)

    # 异常：子空间尖峰 / 窄角半环 / 稀疏高值云（6:2:2）
    n_anom = int(n * anom_ratio)
    frac = [0.6, 0.2, 0.2]
    cnt = [int(n_anom*frac[0]), int(n_anom*frac[1])]
    cnt.append(n_anom - sum(cnt))
    k_low, k_high = max(2, int(d*0.35)), max(3, int(d*0.5))
    anom_lo, anom_hi = 8.5, 11.0

    A1=[]
    for _ in range(cnt[0]):
        x = r.normal(0,1,size=d)
        k = r.integers(k_low, k_high+1); idx = r.choice(d, size=k, replace=False)
        x[idx] += r.uniform(anom_lo, anom_hi, size=k)
        A1.append(x)
    A1 = np.vstack(A1) if A1 else np.empty((0,d))

    A2=[]
    half_span = np.pi/6; mid_rad = 2.0 + 2.2/2
    for _ in range(cnt[1]):
        t = r.uniform(-half_span/2, half_span/2)
        rad = r.normal(mid_rad, 0.015)
        x = np.zeros(d); x[0]=rad*np.cos(t); x[1]=rad*np.sin(t)
        if d>2: x[2:] = r.normal(0.35,0.4,size=d-2)
        A2.append(x)
    A2 = np.vstack(A2) if A2 else np.empty((0,d))

    A3=[]
    for _ in range(cnt[2]):
        x = r.normal(0,1,size=d)
        k = r.integers(k_low, k_high+1); idx = r.choice(d, size=k, replace=False)
        x[idx] += r.normal(anom_hi,0.35,size=k)
        A3.append(x)
    A3 = np.vstack(A3) if A3 else np.empty((0,d))

    X = np.vstack([Xn, np.vstack([A1,A2,A3])])
    y = np.hstack([np.zeros(len(Xn),int), np.ones(len(X)-len(Xn),int)])
    return _stdz(X), y

# ---------- Sparse-HighD：超高维稀疏（D>1000） ----------
def make_tabular_sparse_highD(n=4000, d=9500, anom_ratio=0.0128, sparsity=0.995, seed=42):
    r = np.random.default_rng(seed)
    n_anom = max(1, int(n*anom_ratio)); n_norm = n - n_anom

    ranks = np.arange(1, d+1)
    p = 1.0 / (ranks ** 1.1); p = p / p.sum()
    gate = np.clip((1.0 - sparsity), 1e-5, 0.2)
    p = p * gate * d

    Xn = (r.random((n_norm, d)) < p).astype(float)
    Xn += r.normal(0, 0.02, size=Xn.shape)

    cnt1 = int(n_anom*0.7); cnt2 = n_anom - cnt1
    Xa1 = (r.random((cnt1, d)) < p).astype(float)
    K_low, K_high = max(5, int(d*0.002)), max(6, int(d*0.01))
    for i in range(cnt1):
        K = r.integers(K_low, K_high+1)
        idx = r.choice(d, size=K, replace=False)
        Xa1[i, idx] = 1.0 + r.uniform(2.0, 4.0, size=K)
    Xa2 = np.zeros((cnt2, d))
    for i in range(cnt2):
        K = r.integers(K_low, K_high+1)
        idx = r.choice(d, size=K, replace=False)
        Xa2[i, idx] = r.uniform(3.0, 6.0, size=K)

    X = np.vstack([Xn, Xa1, Xa2])
    y = np.hstack([np.zeros(n_norm, int), np.ones(n_anom, int)])
    return _stdz(X), y

# ---------- 自动路由 ----------
def make_tabular_auto(n, d, anom_ratio, seed=42, **kw):
    if d <= 1000:
        return make_tabular_hard(n=n, d=d, anom_ratio=anom_ratio, seed=seed,
                                 noise=kw.get("noise", 0.02))
    else:
        sparsity = kw.get('sparsity', 0.995 if d<20000 else 0.999)
        return make_tabular_sparse_highD(n=n, d=d, anom_ratio=anom_ratio,
                                         sparsity=sparsity, seed=seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="tabular.csv", help="输出CSV路径")
    ap.add_argument("--n", type=int, required=True, help="样本总数")
    ap.add_argument("--d", type=int, required=True, help="特征维数")
    ap.add_argument("--anom_ratio", type=float, required=True, help="异常占比")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--sparsity", type=float, default=None, help="仅对稀疏高维生成器有效")
    ap.add_argument("--noise", type=float, default=None, help="仅对Dense-Hard有效")
    args = ap.parse_args()

    X, y = make_tabular_auto(
        n=args.n, d=args.d, anom_ratio=args.anom_ratio, seed=args.seed,
        sparsity=args.sparsity if args.sparsity is not None else (0.995 if args.d<=1000 else (0.995 if args.d<20000 else 0.999)),
        noise=0.02 if args.noise is None else args.noise
    )

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] saved: {args.out}  shape={X.shape}  positives={y.sum()}  ratio={y.mean():.4f}")

if __name__ == "__main__":
    main()
