import numpy as np
from dataclasses import dataclass

from conic_ex1_formfinding import (
    ConicExample1Settings,
    build_conic_mesh,
    run_formfinding,
    build_plot,
)


@dataclass
class ConicExample2Settings(ConicExample1Settings):
    # Exercise 2 from Gosling et al. (2013), Fig. 3:
    # "As Exercise 1, except..."
    # - Circular head ring: 4 m above base, 5 m diameter, fixed
    # - Prestress: radial (warp) = circumferential (fill) = 4 kN/m
    ring_diameter: float = 5.0
    ring_height: float = 4.0
    target_sigma_radial: float = 4.0
    target_sigma_circ: float = 4.0


def main():
    settings = ConicExample2Settings()
    coords, triangles, fixed_nodes, center_xy = build_conic_mesh(settings)

    print("===== Conic Exercise 2 Input Summary =====")
    print(f"Geometry: {settings.base_size:.1f} m x {settings.base_size:.1f} m square, fixed edges")
    print(
        f"Head ring: diameter={settings.ring_diameter:.1f} m, "
        f"elevation={settings.ring_height:.1f} m, fixed"
    )
    print(
        f"Target prestress radial/circ: "
        f"{settings.target_sigma_radial:.2f} / {settings.target_sigma_circ:.2f} kN/m"
    )
    print("Material constants (for reporting):")
    print(
        f"  Ew={settings.warp_modulus:.1f} kN/m, Ef={settings.fill_modulus:.1f} kN/m, "
        f"G={settings.shear_modulus:.1f} kN/m"
    )
    print(
        f"  nu_wf={settings.poisson_wf:.2f}, nu_fw={settings.poisson_fw:.2f}"
    )
    print(f"Nodes={coords.shape[0]}, triangles={triangles.shape[0]}, fixed nodes={fixed_nodes.size}")

    results = run_formfinding(
        coords0=coords.copy(),
        triangles=triangles,
        fixed_nodes=fixed_nodes,
        center_xy=center_xy,
        settings=settings,
    )
    coords[:] = results["coords"]

    sigma_r = results["sigma_radial"]
    sigma_c = results["sigma_circ"]
    history = results["history"]
    print("\n===== Conic Exercise 2 Final Summary =====")
    if history:
        print(f"Outer iterations: {history[-1]['iteration']}")
        print(f"Final convergence metric: {history[-1]['conv']:.3e}")
    print(f"Mean sigma_radial (kN/m): {np.mean(sigma_r):.3f} | std: {np.std(sigma_r):.3f}")
    print(f"Mean sigma_circ   (kN/m): {np.mean(sigma_c):.3f} | std: {np.std(sigma_c):.3f}")

    build_plot(
        coords=coords,
        triangles=triangles,
        fixed_nodes=fixed_nodes,
        title="Conic Example 2 form-finding (equal prestress)",
        out_html="conic_ex2_formfound.html",
    )


if __name__ == "__main__":
    main()
