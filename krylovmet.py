#!/usr/bin/env python3
"""
Tiny meteorology-flavoured Krylov demo:
3D-Var normal equations solved with Conjugate Gradient (CG).

- State x: 2D temperature field on a grid (flattened to 1D vector)
- Background term: B^{-1} ≈ (I - alpha * Laplacian) (Helmholtz-like), SPD
- Observation operator H: point sampling at m grid points
- R = sigma_r^2 I
- Solve (B^{-1} + H^T R^{-1} H) δx = H^T R^{-1}(y - H x_b)
- Analysis: x_a = x_b + δx
"""

import numpy as np

# ----------------------------
# Utilities
# ----------------------------
def build_laplacian_coeffs(nx, ny, dx=1.0, dy=1.0):
    """Return helper to apply 5-point Laplacian with homogeneous Neumann BCs."""
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    def lap(u):
        U = u.reshape(ny, nx)
        V = np.empty_like(U)

        # Neumann BC via one-sided differences at boundaries
        # interior
        V[1:-1,1:-1] = (
            (U[1:-1,2:] - 2*U[1:-1,1:-1] + U[1:-1,0:-2]) * inv_dx2 +
            (U[2:,1:-1] - 2*U[1:-1,1:-1] + U[0:-2,1:-1]) * inv_dy2
        )
        # edges: replicate nearest interior gradient (Neumann ~ zero normal gradient)
        # left/right
        V[:,0]     = (U[:,1]     - U[:,0])     * inv_dx2 + \
                     (np.roll(U, -1, axis=0)[:,0] - 2*U[:,0] + np.roll(U, 1, axis=0)[:,0]) * inv_dy2
        V[:,-1]    = (U[:,-2]    - U[:,-1])    * inv_dx2 + \
                     (np.roll(U, -1, axis=0)[:,-1] - 2*U[:,-1] + np.roll(U, 1, axis=0)[:,-1]) * inv_dy2
        # top/bottom
        V[0,:]     = (np.roll(U, -1, axis=1)[0,:] - 2*U[0,:] + np.roll(U, 1, axis=1)[0,:]) * inv_dx2 + \
                     (U[1,:]     - U[0,:])     * inv_dy2
        V[-1,:]    = (np.roll(U, -1, axis=1)[-1,:] - 2*U[-1,:] + np.roll(U, 1, axis=1)[-1,:]) * inv_dx2 + \
                     (U[-2,:]    - U[-1,:])    * inv_dy2

        # corners got included by rows above; good enough for demo.
        return V.ravel()

    return lap

def make_obs_operator(idx_obs, n_state):
    """Return H(v) and HT(w) closures for point-sampling observations."""
    idx_obs = np.asarray(idx_obs, dtype=int)
    m = idx_obs.size

    def H(v):
        return v[idx_obs]  # shape (m,)

    def HT(w):
        out = np.zeros(n_state, dtype=w.dtype)
        np.add.at(out, idx_obs, w)  # scatter add
        return out

    return H, HT

# ----------------------------
# Conjugate Gradient (Krylov)
# ----------------------------
def conjugate_gradient(A, b, M_inv=None, x0=None, tol=1e-6, maxiter=200, callback=None):
    """
    Solve A x = b for SPD A using preconditioned CG.
    A: function(v)->A v
    M_inv: function(r)->approximate M^{-1} r (preconditioner)
    """
    n = b.size
    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - A(x)
    z = M_inv(r) if M_inv else r
    p = z.copy()
    rz_old = np.dot(r, z)
    if callback:
        callback(0, np.linalg.norm(r))

    for k in range(1, maxiter+1):
        Ap = A(p)
        alpha = rz_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rn = np.linalg.norm(r)
        if callback:
            callback(k, rn)
        if rn <= tol:
            break
        z = M_inv(r) if M_inv else r
        rz_new = np.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    return x, k, rn

# ----------------------------
# Problem setup (toy 3D-Var)
# ----------------------------
def main():
    rng = np.random.default_rng(42)

    nx, ny = 80, 60
    n = nx * ny
    dx = dy = 1.0

    # "True" field (smooth waves + small-scale)
    y_coords, x_coords = np.mgrid[0:ny, 0:nx]
    truth = (
        10
        + 2.5*np.sin(2*np.pi*x_coords/nx) * np.cos(2*np.pi*y_coords/ny)
        + 0.8*np.sin(4*np.pi*x_coords/nx + 0.6) * np.sin(3*np.pi*y_coords/ny + 1.1)
    )
    truth += 0.3 * rng.standard_normal(size=(ny, nx))
    x_true = truth.ravel()

    # Background (first guess): truth + background noise
    sigma_b = 1.5
    x_b = x_true + sigma_b * rng.standard_normal(n)

    # Observations: pick m random grid points, add obs noise
    m = int(0.12 * n)  # ~12% of points observed
    idx_obs = rng.choice(n, size=m, replace=False)
    sigma_r = 0.8
    y_obs = x_true[idx_obs] + sigma_r * rng.standard_normal(m)

    # Operators
    lap = build_laplacian_coeffs(nx, ny, dx, dy)
    alpha = 0.15  # controls background smoothness (larger -> stronger B^{-1})
    def Binv(v):  # Helmholtz-like SPD operator: (I - alpha*Lap) v
        return v - alpha * lap(v)

    H, HT = make_obs_operator(idx_obs, n)

    # Left-hand side operator: A = B^{-1} + H^T R^{-1} H
    rinv = 1.0 / (sigma_r**2)
    def Aop(v):
        return Binv(v) + HT(rinv * H(v))

    # Right-hand side: H^T R^{-1} (y - H x_b)
    b = HT(rinv * (y_obs - H(x_b)))

    # Preconditioner: diagonal of A (cheap approx)
    # For Binv ≈ (1 + 2*alpha*(1/dx^2 + 1/dy^2)) on interior; tweak for demo
    diag_Binv = 1.0 + 2.0*alpha*((1.0/dx**2) + (1.0/dy**2))
    diag = np.full(n, diag_Binv)
    # Each observed location contributes +1/σ_r^2 to the diagonal
    np.add.at(diag, idx_obs, rinv)
    M_inv = lambda r: r / diag

    # Solve with CG (Krylov)
    history = []
    def cb(k, rn):
        history.append((k, rn))
        if k % 10 == 0:
            print(f"iter {k:3d} | residual {rn:.3e}")

    x0 = np.zeros(n)  # start from zero increment
    dx_sol, iters, final_res = conjugate_gradient(Aop, b, M_inv=M_inv, x0=x0, tol=1e-6, maxiter=500, callback=cb)

    x_analysis = x_b + dx_sol

    # Diagnostics
    def rmse(a, b):
        return np.sqrt(np.mean((a - b)**2))

    print("\n=== Results ===")
    print(f"CG iterations: {iters}, final residual: {final_res:.3e}")
    print(f"Background RMSE vs truth: {rmse(x_b, x_true):.3f}")
    print(f"Analysis   RMSE vs truth: {rmse(x_analysis, x_true):.3f}")

    # Optional quicklook (comment out if running headless)
    try:
        import matplotlib.pyplot as plt

        fig1 = plt.figure()
        plt.title("Background vs Analysis RMSE by iteration (residual norm)")
        it, res = np.array(history).T
        plt.semilogy(it, res)
        plt.xlabel("Iteration")
        plt.ylabel("||r_k||")

        fig2 = plt.figure()
        plt.title("Truth")
        plt.imshow(x_true.reshape(ny, nx), origin="lower")
        plt.colorbar()

        fig3 = plt.figure()
        plt.title("Background (x_b)")
        plt.imshow(x_b.reshape(ny, nx), origin="lower")
        plt.colorbar()

        fig4 = plt.figure()
        plt.title("Analysis (x_a)")
        plt.imshow(x_analysis.reshape(ny, nx), origin="lower")
        plt.colorbar()

        plt.show()
    except Exception as e:
        print(f"(Skipping plots: {e})")

if __name__ == "__main__":
    main()
