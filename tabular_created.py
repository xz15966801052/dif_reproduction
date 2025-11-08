# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd

def _stdz(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-9
    return (X - mu) / sd

def make_tabular_hard(n=20000, d=32, anom_ratio=0.03, noise=0.03, seed=42):
    """
    Normal: The first two dimensions consist of two concentric rings, while the remaining dimensions are double Gaussian clusters (multimodal + light noise)
    Anomaly: Three mixtures - subspace spikes/inner half-rings/sparse high-value clouds (linearly unsplictable)
    """
    r = np.random.default_rng(seed)
    n_anom = max(1, int(n * anom_ratio))
    n_norm = n - n_anom

    # ---- Normal: Concentric ring (first 2 dimensions) ----
    a1 = r.uniform(0, 2*np.pi, size=n_norm//2)
    a2 = r.uniform(0, 2*np.pi, size=n_norm - a1.size)
    r1 = r.normal(2.0, 0.05, size=a1.size)
    r2 = r.normal(4.0, 0.08, size=a2.size)
    ring1 = np.stack([r1*np.cos(a1), r1*np.sin(a1)], axis=1)
    ring2 = np.stack([r2*np.cos(a2), r2*np.sin(a2)], axis=1)
    X2 = np.vstack([ring1, ring2])

    # ---- Normal: The remaining dimensions of the double Gaussian cluster (multimodal)----
    rest = d - 2
    cov = np.diag(r.uniform(0.3, 1.1, size=rest))
    g1 = r.multivariate_normal(mean=np.zeros(rest),    cov=cov, size=X2.shape[0]//2)
    g2 = r.multivariate_normal(mean=np.ones(rest)*1.5, cov=cov, size=X2.shape[0]-g1.shape[0])
    Xn = np.hstack([X2, np.vstack([g1, g2])])
    Xn += r.normal(0, noise, size=Xn.shape)

    # ---- Anomaly: Three types of hard modes ----
    k1 = max(2, d//4)   # Subspace size 1
    k2 = max(2, d//3)   # Subspace size 2
    Xa = []
    for _ in range(n_anom):
        mode = r.integers(0, 3)
        if mode == 0:
            # Subspace spike: The overall rise on a random subspace
            x = r.normal(0, 1, size=d)
            idx = r.choice(d, size=k1, replace=False)
            x[idx] += r.normal(6, 0.6, size=k1)
        elif mode == 1:
            # The inner semicircle: The first two dimensions form a semicircle, which is embedded between the normal rings
            t = r.uniform(0, np.pi/2)
            rad = r.normal(3.0, 0.03)
            x = np.zeros(d); x[0] = rad*np.cos(t); x[1] = rad*np.sin(t)
            x[2:] = r.normal(0.5, 0.6, size=d-2)
        else:
            # Sparse high-value cloud: Multiple dimensions simultaneously shift to the rarefied region
            x = r.normal(0, 1, size=d)
            idx = r.choice(d, size=k2, replace=False)
            x[idx] += r.uniform(4, 9, size=k2)
        Xa.append(x)
    Xa = np.vstack(Xa)

    # ---- Merge + Standardization ----
    X = np.vstack([Xn, Xa])
    y = np.hstack([np.zeros(len(Xn), dtype=int), np.ones(len(Xa), dtype=int)])
    X = _stdz(X)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="tabular_created.csv", help="Output CSV path")
    ap.add_argument("--n", type=int, default=20000, help="Total number of samples ")
    ap.add_argument("--d", type=int, default=32, help="Feature dimension")
    ap.add_argument("--anom_ratio", type=float, default=0.03, help="Abnormal proportion")
    ap.add_argument("--noise", type=float, default=0.03, help="Normal sample Gaussian noise")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    X, y = make_tabular_hard(n=args.n, d=args.d, anom_ratio=args.anom_ratio,
                             noise=args.noise, seed=args.seed)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] saved: {args.out}  shape={X.shape}  positives={y.sum()}  ratio={y.mean():.4f}")

if __name__ == "__main__":
    main()
