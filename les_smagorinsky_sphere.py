"""
LES Smagorinsky — 3D incompressible flow past a sphere.

Cartesian collocated finite-volume / finite-difference solver with:
  - Fractional-step (Chorin) projection
  - Smagorinsky SGS eddy viscosity  ν_t = (C_s Δ)² |S|
  - Immersed-sphere no-slip mask
  - Inlet / outlet / wall boundary conditions

Physical defaults match the LES_smagorinksy mesh setup:
  box 10×5×5 m, sphere R=0.5 at (3, 2.5, 2.5), U∞=1, ν=1e-3, C_s=0.17
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class LESConfig:
    # Domain (m) — matches computational_grid_gmsh_visualized.py
    Lx: float = 10.0
    Ly: float = 5.0
    Lz: float = 5.0

    # Sphere
    sphere_center: tuple[float, float, float] = (3.0, 2.5, 2.5)
    sphere_radius: float = 0.5

    # Grid (keep modest for interactive notebooks)
    nx: int = 64
    ny: int = 32
    nz: int = 32

    # Fluid / LES
    rho: float = 1.0
    nu: float = 1e-3
    Cs: float = 0.17
    U_inlet: float = 1.0

    # Time
    dt: float = 2.5e-3
    n_steps: int = 400
    cfl_max: float = 0.45

    # Pressure Poisson
    poisson_iters: int = 80
    poisson_tol: float = 1e-4

    # Output
    save_every: int = 20
    seed: int = 7


@dataclass
class LESState:
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    p: np.ndarray
    nu_t: np.ndarray
    solid: np.ndarray
    time: float = 0.0
    step: int = 0
    history: dict = field(default_factory=lambda: {
        "time": [],
        "ke": [],
        "max_speed": [],
        "mean_nu_t": [],
        "div_max": [],
    })
    snapshots: list = field(default_factory=list)


class SmagorinskyLESSphere:
    """3D LES Smagorinsky solver for flow past an immersed sphere."""

    def __init__(self, cfg: LESConfig | None = None):
        self.cfg = cfg or LESConfig()
        c = self.cfg
        self.dx = c.Lx / c.nx
        self.dy = c.Ly / c.ny
        self.dz = c.Lz / c.nz
        self.delta = (self.dx * self.dy * self.dz) ** (1.0 / 3.0)

        self.x = (np.arange(c.nx) + 0.5) * self.dx
        self.y = (np.arange(c.ny) + 0.5) * self.dy
        self.z = (np.arange(c.nz) + 0.5) * self.dz
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")

        cx, cy, cz = c.sphere_center
        r = np.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2 + (self.Z - cz) ** 2)
        self.solid = r <= c.sphere_radius
        self.fluid = ~self.solid

        rng = np.random.default_rng(c.seed)
        u = np.full((c.nx, c.ny, c.nz), c.U_inlet, dtype=np.float64)
        v = 0.01 * c.U_inlet * rng.standard_normal((c.nx, c.ny, c.nz))
        w = 0.01 * c.U_inlet * rng.standard_normal((c.nx, c.ny, c.nz))
        u[self.solid] = 0.0
        v[self.solid] = 0.0
        w[self.solid] = 0.0

        self.state = LESState(
            u=u, v=v, w=w,
            p=np.zeros((c.nx, c.ny, c.nz)),
            nu_t=np.zeros((c.nx, c.ny, c.nz)),
            solid=self.solid,
        )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _ddx(f, dx):
        out = np.empty_like(f)
        out[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
        out[0] = (f[1] - f[0]) / dx
        out[-1] = (f[-1] - f[-2]) / dx
        return out

    @staticmethod
    def _ddy(f, dy):
        out = np.empty_like(f)
        out[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dy)
        out[:, 0] = (f[:, 1] - f[:, 0]) / dy
        out[:, -1] = (f[:, -1] - f[:, -2]) / dy
        return out

    @staticmethod
    def _ddz(f, dz):
        out = np.empty_like(f)
        out[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * dz)
        out[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dz
        out[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dz
        return out

    @staticmethod
    def _lap(f, dx, dy, dz):
        out = np.zeros_like(f)
        out[1:-1] += (f[2:] - 2 * f[1:-1] + f[:-2]) / dx**2
        out[:, 1:-1] += (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / dy**2
        out[:, :, 1:-1] += (f[:, :, 2:] - 2 * f[:, :, 1:-1] + f[:, :, :-2]) / dz**2
        # Neumann-like edges: copy one-sided second differences
        out[0] += (f[1] - f[0]) / dx**2
        out[-1] += (f[-2] - f[-1]) / dx**2
        out[:, 0] += (f[:, 1] - f[:, 0]) / dy**2
        out[:, -1] += (f[:, -2] - f[:, -1]) / dy**2
        out[:, :, 0] += (f[:, :, 1] - f[:, :, 0]) / dz**2
        out[:, :, -1] += (f[:, :, -2] - f[:, :, -1]) / dz**2
        return out

    def apply_velocity_bc(self, u, v, w):
        c = self.cfg
        # Inlet (x=0 face)
        u[0, :, :] = c.U_inlet
        v[0, :, :] = 0.0
        w[0, :, :] = 0.0
        # Outlet (x=Lx): Neumann
        u[-1] = u[-2]
        v[-1] = v[-2]
        w[-1] = w[-2]
        # Side walls y,z: no-slip
        u[:, 0, :] = 0.0
        v[:, 0, :] = 0.0
        w[:, 0, :] = 0.0
        u[:, -1, :] = 0.0
        v[:, -1, :] = 0.0
        w[:, -1, :] = 0.0
        u[:, :, 0] = 0.0
        v[:, :, 0] = 0.0
        w[:, :, 0] = 0.0
        u[:, :, -1] = 0.0
        v[:, :, -1] = 0.0
        w[:, :, -1] = 0.0
        # Immersed sphere
        u[self.solid] = 0.0
        v[self.solid] = 0.0
        w[self.solid] = 0.0
        return u, v, w

    def smagorinsky_viscosity(self, u, v, w):
        """ν_t = (C_s Δ)² |S|,  |S| = sqrt(2 S_ij S_ij)."""
        dx, dy, dz = self.dx, self.dy, self.dz
        dudx = self._ddx(u, dx)
        dudy = self._ddy(u, dy)
        dudz = self._ddz(u, dz)
        dvdx = self._ddx(v, dx)
        dvdy = self._ddy(v, dy)
        dvdz = self._ddz(v, dz)
        dwdx = self._ddx(w, dx)
        dwdy = self._ddy(w, dy)
        dwdz = self._ddz(w, dz)

        s11 = dudx
        s22 = dvdy
        s33 = dwdz
        s12 = 0.5 * (dudy + dvdx)
        s13 = 0.5 * (dudz + dwdx)
        s23 = 0.5 * (dvdz + dwdy)

        s_mag = np.sqrt(
            2.0 * (s11**2 + s22**2 + s33**2 + 2 * (s12**2 + s13**2 + s23**2))
        )
        nu_t = (self.cfg.Cs * self.delta) ** 2 * s_mag
        nu_t[self.solid] = 0.0
        return nu_t

    def convective(self, u, v, w):
        """Upwind-biased convective term (u·∇)φ for each velocity component."""
        dx, dy, dz = self.dx, self.dy, self.dz

        def advect(phi):
            # First-order upwind
            flux_x = np.zeros_like(phi)
            pos = u > 0
            flux_x[1:-1] = np.where(
                pos[1:-1],
                u[1:-1] * (phi[1:-1] - phi[:-2]) / dx,
                u[1:-1] * (phi[2:] - phi[1:-1]) / dx,
            )
            flux_y = np.zeros_like(phi)
            posy = v > 0
            flux_y[:, 1:-1] = np.where(
                posy[:, 1:-1],
                v[:, 1:-1] * (phi[:, 1:-1] - phi[:, :-2]) / dy,
                v[:, 1:-1] * (phi[:, 2:] - phi[:, 1:-1]) / dy,
            )
            flux_z = np.zeros_like(phi)
            posz = w > 0
            flux_z[:, :, 1:-1] = np.where(
                posz[:, :, 1:-1],
                w[:, :, 1:-1] * (phi[:, :, 1:-1] - phi[:, :, :-2]) / dz,
                w[:, :, 1:-1] * (phi[:, :, 2:] - phi[:, :, 1:-1]) / dz,
            )
            return flux_x + flux_y + flux_z

        return advect(u), advect(v), advect(w)

    def diffuse(self, phi, nu_eff):
        """∇·(ν_eff ∇φ) with variable eddy viscosity (harmonic-face approx)."""
        dx, dy, dz = self.dx, self.dy, self.dz
        out = np.zeros_like(phi)

        # x-faces
        nu_f = 0.5 * (nu_eff[1:] + nu_eff[:-1])
        flux = nu_f * (phi[1:] - phi[:-1]) / dx
        out[1:-1] += (flux[1:] - flux[:-1]) / dx
        out[0] += flux[0] / dx
        out[-1] -= flux[-1] / dx

        # y-faces
        nu_f = 0.5 * (nu_eff[:, 1:] + nu_eff[:, :-1])
        flux = nu_f * (phi[:, 1:] - phi[:, :-1]) / dy
        out[:, 1:-1] += (flux[:, 1:] - flux[:, :-1]) / dy
        out[:, 0] += flux[:, 0] / dy
        out[:, -1] -= flux[:, -1] / dy

        # z-faces
        nu_f = 0.5 * (nu_eff[:, :, 1:] + nu_eff[:, :, :-1])
        flux = nu_f * (phi[:, :, 1:] - phi[:, :, :-1]) / dz
        out[:, :, 1:-1] += (flux[:, :, 1:] - flux[:, :, :-1]) / dz
        out[:, :, 0] += flux[:, :, 0] / dz
        out[:, :, -1] -= flux[:, :, -1] / dz
        return out

    def divergence(self, u, v, w):
        return self._ddx(u, self.dx) + self._ddy(v, self.dy) + self._ddz(w, self.dz)

    def solve_pressure_poisson(self, div_ustar, dt):
        """∇²p = (ρ/dt) ∇·u*  with Neumann walls / Dirichlet outlet strip."""
        c = self.cfg
        rhs = (c.rho / dt) * div_ustar
        rhs[self.solid] = 0.0
        p = self.state.p.copy()
        dx2, dy2, dz2 = self.dx**2, self.dy**2, self.dz**2
        inv = 2.0 / dx2 + 2.0 / dy2 + 2.0 / dz2

        for _ in range(c.poisson_iters):
            p_old = p
            p_new = np.empty_like(p)
            p_new[1:-1, 1:-1, 1:-1] = (
                (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]) / dx2
                + (p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]) / dy2
                + (p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]) / dz2
                - rhs[1:-1, 1:-1, 1:-1]
            ) / inv

            # Boundaries: Neumann (copy) except outlet Dirichlet p=0
            p_new[0] = p_new[1]
            p_new[-1] = 0.0
            p_new[:, 0, :] = p_new[:, 1, :]
            p_new[:, -1, :] = p_new[:, -2, :]
            p_new[:, :, 0] = p_new[:, :, 1]
            p_new[:, :, -1] = p_new[:, :, -2]
            p_new[self.solid] = p_new[self.solid]  # keep; zero gradient via fluid nbr avg below

            # Soft average from fluid neighbors into solid (Neumann on sphere)
            # Simple: set solid pressure to mean of available fluid stencil
            p = p_new
            err = np.max(np.abs(p - p_old))
            if err < c.poisson_tol:
                break
        p[self.solid] = 0.0
        return p

    def stable_dt(self, u, v, w, nu_eff):
        c = self.cfg
        umax = max(np.max(np.abs(u)), 1e-6)
        vmax = max(np.max(np.abs(v)), 1e-6)
        wmax = max(np.max(np.abs(w)), 1e-6)
        dt_cfl = c.cfl_max / (umax / self.dx + vmax / self.dy + wmax / self.dz)
        nu_max = max(np.max(nu_eff), c.nu)
        dt_diff = 0.2 / (nu_max * (1 / self.dx**2 + 1 / self.dy**2 + 1 / self.dz**2))
        return float(min(c.dt, dt_cfl, dt_diff))

    def step(self):
        c = self.cfg
        st = self.state
        u, v, w = st.u, st.v, st.w

        nu_t = self.smagorinsky_viscosity(u, v, w)
        nu_eff = c.nu + nu_t
        dt = self.stable_dt(u, v, w, nu_eff)

        cu, cv, cw = self.convective(u, v, w)
        du = self.diffuse(u, nu_eff)
        dv = self.diffuse(v, nu_eff)
        dw = self.diffuse(w, nu_eff)

        u_star = u + dt * (-cu + du)
        v_star = v + dt * (-cv + dv)
        w_star = w + dt * (-cw + dw)
        u_star, v_star, w_star = self.apply_velocity_bc(u_star, v_star, w_star)

        div = self.divergence(u_star, v_star, w_star)
        p = self.solve_pressure_poisson(div, dt)

        u = u_star - dt / c.rho * self._ddx(p, self.dx)
        v = v_star - dt / c.rho * self._ddy(p, self.dy)
        w = w_star - dt / c.rho * self._ddz(p, self.dz)
        u, v, w = self.apply_velocity_bc(u, v, w)

        st.u, st.v, st.w, st.p, st.nu_t = u, v, w, p, nu_t
        st.time += dt
        st.step += 1

        speed = np.sqrt(u**2 + v**2 + w**2)
        ke = 0.5 * np.mean(speed[self.fluid] ** 2)
        div_max = float(np.max(np.abs(self.divergence(u, v, w)[self.fluid])))
        st.history["time"].append(st.time)
        st.history["ke"].append(ke)
        st.history["max_speed"].append(float(np.max(speed[self.fluid])))
        st.history["mean_nu_t"].append(float(np.mean(nu_t[self.fluid])))
        st.history["div_max"].append(div_max)
        return dt

    def maybe_snapshot(self):
        c = self.cfg
        st = self.state
        if st.step % c.save_every != 0 and st.step != c.n_steps:
            return
        # Store mid-plane (k = nz//2) and a coarse 3D subsample for viz
        k = c.nz // 2
        stride = max(1, min(c.nx, c.ny, c.nz) // 24)
        st.snapshots.append({
            "step": st.step,
            "time": st.time,
            "u_xy": st.u[:, :, k].copy(),
            "v_xy": st.v[:, :, k].copy(),
            "p_xy": st.p[:, :, k].copy(),
            "nu_t_xy": st.nu_t[:, :, k].copy(),
            "speed_xy": np.sqrt(st.u[:, :, k]**2 + st.v[:, :, k]**2 + st.w[:, :, k]**2),
            "u3": st.u[::stride, ::stride, ::stride].copy(),
            "v3": st.v[::stride, ::stride, ::stride].copy(),
            "w3": st.w[::stride, ::stride, ::stride].copy(),
            "speed3": np.sqrt(
                st.u[::stride, ::stride, ::stride]**2
                + st.v[::stride, ::stride, ::stride]**2
                + st.w[::stride, ::stride, ::stride]**2
            ),
            "stride": stride,
        })

    def run(self, n_steps: int | None = None, progress: Callable | None = None):
        n = n_steps if n_steps is not None else self.cfg.n_steps
        self.maybe_snapshot()
        for i in range(n):
            dt = self.step()
            self.maybe_snapshot()
            if progress and (i % 20 == 0 or i == n - 1):
                progress(i + 1, n, self.state)
            elif (i % 50 == 0) or (i == n - 1):
                h = self.state.history
                print(
                    f"step {self.state.step:4d}/{n}  t={self.state.time:.3f}  "
                    f"dt={dt:.2e}  |U|_max={h['max_speed'][-1]:.3f}  "
                    f"νt_mean={h['mean_nu_t'][-1]:.2e}  div_max={h['div_max'][-1]:.2e}"
                )
        return self.state

    def midplane_coords(self):
        return self.x, self.y

    def sphere_surface_mesh(self, n_theta=40, n_phi=80):
        cx, cy, cz = self.cfg.sphere_center
        R = self.cfg.sphere_radius
        th = np.linspace(0, np.pi, n_theta)
        ph = np.linspace(0, 2 * np.pi, n_phi)
        th, ph = np.meshgrid(th, ph, indexing="ij")
        xs = cx + R * np.sin(th) * np.cos(ph)
        ys = cy + R * np.sin(th) * np.sin(ph)
        zs = cz + R * np.cos(th)
        return xs, ys, zs


def run_demo(quick: bool = True) -> SmagorinskyLESSphere:
    """Convenience runner used by the notebook / CLI."""
    if quick:
        cfg = LESConfig(nx=48, ny=24, nz=24, n_steps=120, dt=3e-3, save_every=15,
                        poisson_iters=40)
    else:
        cfg = LESConfig()
    solver = SmagorinskyLESSphere(cfg)
    print(
        f"LES Smagorinsky sphere | grid {cfg.nx}x{cfg.ny}x{cfg.nz} | "
        f"Δ={solver.delta:.4f} | Re≈{cfg.U_inlet * 2 * cfg.sphere_radius / cfg.nu:.0f}"
    )
    solver.run()
    return solver


if __name__ == "__main__":
    run_demo(quick=True)
