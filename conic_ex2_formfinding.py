import numpy as np
from dataclasses import dataclass

try:
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.spatial import Delaunay
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


@dataclass
class ConicExample2Settings:
    # Exercise 2 geometry from Gosling et al. (2013), Fig. 3
    # "As Exercise 1, except..."
    # Circular head ring: 4 m above base, 5 m diameter, fixed.
    base_size: float = 14.0
    ring_diameter: float = 5.0
    ring_height: float = 4.0

    # Mesh controls
    Nx: int = 80
    Ny: int = 80
    ring_points: int = 120

    # Prestress targets (kN/m): radial (warp) and circumferential (fill)
    target_sigma_radial: float = 4.0
    target_sigma_circ: float = 4.0

    # Form-finding controls
    max_outer_iterations: int = 40
    stress_tolerance: float = 1e-4
    distortion_limit: float = 1.1
    relax_distorted_elements: bool = True
    optimise_xyz: bool = False  # keep plan projection fixed for stability
    min_weight: float = 1e-12

    # Material constants used in the round-robin specification.
    warp_modulus: float = 600.0
    fill_modulus: float = 600.0
    poisson_wf: float = 0.4
    poisson_fw: float = 0.4
    shear_modulus: float = 30.0


def normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(vec)
    if nrm < 1e-14:
        return fallback / np.linalg.norm(fallback)
    return vec / nrm


def triangle_edges(tri: np.ndarray) -> list[tuple[int, int]]:
    a, b, c = tri
    return [(a, b), (b, c), (c, a)]


def triangle_frame_and_sides(coords: np.ndarray, tri: np.ndarray):
    p0, p1, p2 = coords[tri[0]], coords[tri[1]], coords[tri[2]]
    v1 = p1 - p0
    v2 = p2 - p0
    nvec = np.cross(v1, v2)
    area2 = np.linalg.norm(nvec)
    area = 0.5 * area2
    if area < 1e-14:
        raise ValueError("Degenerate triangle encountered in mesh.")

    normal = nvec / area2
    e1 = normalize(v1, fallback=np.array([1.0, 0.0, 0.0]))
    e2 = normalize(np.cross(normal, e1), fallback=np.array([0.0, 1.0, 0.0]))

    q0 = np.array([0.0, 0.0])
    q1 = np.array([np.dot(p1 - p0, e1), np.dot(p1 - p0, e2)])
    q2 = np.array([np.dot(p2 - p0, e1), np.dot(p2 - p0, e2)])
    q = [q0, q1, q2]

    edge_pairs = [(0, 1), (1, 2), (2, 0)]
    lengths = np.zeros(3, dtype=float)
    side_dirs_2d = np.zeros((3, 2), dtype=float)
    for j, (r, s) in enumerate(edge_pairs):
        dv = q[s] - q[r]
        lj = np.linalg.norm(dv)
        if lj < 1e-14:
            raise ValueError("Zero-length side encountered.")
        lengths[j] = lj
        side_dirs_2d[j] = dv / lj
    return area, lengths, side_dirs_2d, e1, e2, normal


def project_radial_circ_to_local(
    center_xy: np.ndarray,
    centroid_xyz: np.ndarray,
    normal: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
):
    radial3 = np.array(
        [
            centroid_xyz[0] - center_xy[0],
            centroid_xyz[1] - center_xy[1],
            0.0,
        ],
        dtype=float,
    )
    radial3 = radial3 - np.dot(radial3, normal) * normal
    radial3 = normalize(radial3, fallback=e1)

    circ3 = np.cross(normal, radial3)
    circ3 = normalize(circ3, fallback=np.cross(normal, e1))

    r2 = np.array([np.dot(radial3, e1), np.dot(radial3, e2)])
    c2 = np.array([np.dot(circ3, e1), np.dot(circ3, e2)])
    r2 = normalize(r2, fallback=np.array([1.0, 0.0]))
    c2 = normalize(c2, fallback=np.array([0.0, 1.0]))
    return r2, c2


def desired_local_stress_tensor(
    sigma_radial: float,
    sigma_circ: float,
    r2: np.ndarray,
    c2: np.ndarray,
) -> np.ndarray:
    return sigma_radial * np.outer(r2, r2) + sigma_circ * np.outer(c2, c2)


def natural_stress_from_tensor(
    tensor_2d: np.ndarray,
    side_dirs_2d: np.ndarray,
) -> np.ndarray:
    sig_n = np.zeros(3, dtype=float)
    for j in range(3):
        u = side_dirs_2d[j]
        sig_n[j] = float(u @ tensor_2d @ u)
    return sig_n


def compute_membrane_weights(
    coords_ref: np.ndarray,
    triangles: np.ndarray,
    center_xy: np.ndarray,
    sigma_radial_elem: np.ndarray,
    sigma_circ_elem: np.ndarray,
    min_weight: float,
) -> np.ndarray:
    n_elem = triangles.shape[0]
    W = np.zeros((n_elem, 3), dtype=float)
    for i, tri in enumerate(triangles):
        area, lengths, side_dirs, e1, e2, normal = triangle_frame_and_sides(coords_ref, tri)
        centroid = np.mean(coords_ref[tri], axis=0)
        r2, c2 = project_radial_circ_to_local(center_xy, centroid, normal, e1, e2)
        sigma_tensor = desired_local_stress_tensor(sigma_radial_elem[i], sigma_circ_elem[i], r2, c2)
        sigma_n = natural_stress_from_tensor(sigma_tensor, side_dirs)
        side_forces = area * sigma_n / lengths
        W_i = side_forces / (2.0 * lengths)
        W[i] = np.maximum(W_i, min_weight)
    return W


def assemble_weighted_laplacian(
    n_nodes: int,
    triangles: np.ndarray,
    membrane_weights: np.ndarray,
):
    rows = []
    cols = []
    data = []

    def add_edge(a: int, b: int, weight: float):
        rows.extend([a, b, a, b])
        cols.extend([a, b, b, a])
        data.extend([weight, weight, -weight, -weight])

    for i, tri in enumerate(triangles):
        for j, (a, b) in enumerate(triangle_edges(tri)):
            add_edge(int(a), int(b), float(membrane_weights[i, j]))

    if HAVE_SCIPY:
        return sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    L = np.zeros((n_nodes, n_nodes), dtype=float)
    for r, c, v in zip(rows, cols, data):
        L[r, c] += v
    return L


def solve_equilibrium(
    coords_seed: np.ndarray,
    fixed_nodes: np.ndarray,
    triangles: np.ndarray,
    membrane_weights: np.ndarray,
    optimise_xyz: bool,
) -> np.ndarray:
    n_nodes = coords_seed.shape[0]
    all_nodes = np.arange(n_nodes, dtype=int)
    fixed_nodes = np.unique(fixed_nodes.astype(int))
    free_nodes = np.setdiff1d(all_nodes, fixed_nodes)
    if free_nodes.size == 0:
        return coords_seed.copy()

    L = assemble_weighted_laplacian(n_nodes, triangles, membrane_weights)
    if HAVE_SCIPY:
        L_ff = L[free_nodes][:, free_nodes]
        L_fb = L[free_nodes][:, fixed_nodes]
    else:
        L_ff = L[np.ix_(free_nodes, free_nodes)]
        L_fb = L[np.ix_(free_nodes, fixed_nodes)]

    coords_new = coords_seed.copy()
    dims = [0, 1, 2] if optimise_xyz else [2]
    for d in dims:
        rhs = -L_fb @ coords_seed[fixed_nodes, d]
        if HAVE_SCIPY:
            coords_new[free_nodes, d] = spla.spsolve(L_ff, rhs)
        else:
            try:
                coords_new[free_nodes, d] = np.linalg.solve(L_ff, rhs)
            except np.linalg.LinAlgError:
                coords_new[free_nodes, d] = np.linalg.lstsq(L_ff, rhs, rcond=None)[0]
    return coords_new


def compute_equilibrium_stresses(
    coords: np.ndarray,
    triangles: np.ndarray,
    membrane_weights: np.ndarray,
    center_xy: np.ndarray,
):
    n_elem = triangles.shape[0]
    sigma_radial = np.zeros(n_elem, dtype=float)
    sigma_circ = np.zeros(n_elem, dtype=float)
    areas = np.zeros(n_elem, dtype=float)
    for i, tri in enumerate(triangles):
        area, lengths, side_dirs, e1, e2, normal = triangle_frame_and_sides(coords, tri)
        areas[i] = area
        sigma_n = (2.0 * membrane_weights[i] * lengths * lengths) / max(area, 1e-14)
        M = np.array(
            [
                [side_dirs[0, 0] ** 2, side_dirs[0, 1] ** 2, 2.0 * side_dirs[0, 0] * side_dirs[0, 1]],
                [side_dirs[1, 0] ** 2, side_dirs[1, 1] ** 2, 2.0 * side_dirs[1, 0] * side_dirs[1, 1]],
                [side_dirs[2, 0] ** 2, side_dirs[2, 1] ** 2, 2.0 * side_dirs[2, 0] * side_dirs[2, 1]],
            ],
            dtype=float,
        )
        sxx, syy, sxy = np.linalg.solve(M, sigma_n)
        stress_tensor = np.array([[sxx, sxy], [sxy, syy]], dtype=float)

        centroid = np.mean(coords[tri], axis=0)
        r2, c2 = project_radial_circ_to_local(center_xy, centroid, normal, e1, e2)
        sigma_radial[i] = float(r2 @ stress_tensor @ r2)
        sigma_circ[i] = float(c2 @ stress_tensor @ c2)
    return sigma_radial, sigma_circ, areas


def element_areas(coords: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    areas = np.zeros(triangles.shape[0], dtype=float)
    for i, tri in enumerate(triangles):
        p0, p1, p2 = coords[tri[0]], coords[tri[1]], coords[tri[2]]
        areas[i] = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
    return areas


def point_to_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-20:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / ab2)
    t = min(1.0, max(0.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(point - closest))


def build_conic_mesh(settings: ConicExample2Settings):
    if not HAVE_SCIPY:
        raise RuntimeError("SciPy is required to generate the conic mesh.")

    L = settings.base_size
    center = np.array([0.5 * L, 0.5 * L], dtype=float)
    ring_r = 0.5 * settings.ring_diameter

    xs = np.linspace(0.0, L, settings.Nx + 1)
    ys = np.linspace(0.0, L, settings.Ny + 1)
    pts = []
    for y in ys:
        for x in xs:
            if np.linalg.norm(np.array([x, y]) - center) >= ring_r + 1e-8:
                pts.append([x, y])

    theta = np.linspace(0.0, 2.0 * np.pi, settings.ring_points, endpoint=False)
    ring_pts = np.column_stack(
        [
            center[0] + ring_r * np.cos(theta),
            center[1] + ring_r * np.sin(theta),
        ]
    )
    pts.extend(ring_pts.tolist())
    pts = np.array(pts, dtype=float)

    tri_all = Delaunay(pts).simplices

    keep = np.ones(tri_all.shape[0], dtype=bool)
    for i, tri in enumerate(tri_all):
        tri_xy = pts[tri]
        centroid = np.mean(tri_xy, axis=0)
        if np.linalg.norm(centroid - center) < ring_r * 1.0001:
            keep[i] = False
            continue
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            if point_to_segment_distance(center, tri_xy[a], tri_xy[b]) < ring_r * 0.9999:
                keep[i] = False
                break
    triangles = tri_all[keep]

    coords = np.zeros((pts.shape[0], 3), dtype=float)
    coords[:, :2] = pts

    x = coords[:, 0]
    y = coords[:, 1]
    r = np.linalg.norm(coords[:, :2] - center[None, :], axis=1)
    outer = np.isclose(x, 0.0) | np.isclose(x, L) | np.isclose(y, 0.0) | np.isclose(y, L)
    ring = np.isclose(r, ring_r, atol=1e-8)
    fixed_nodes = np.where(outer | ring)[0]

    d_outer = np.minimum.reduce([x, L - x, y, L - y])
    d_ring = np.abs(r - ring_r)
    coords[:, 2] = settings.ring_height * d_outer / (d_outer + d_ring + 1e-12)
    coords[outer, 2] = 0.0
    coords[ring, 2] = settings.ring_height

    return coords, triangles.astype(int), fixed_nodes.astype(int), center


def run_formfinding(
    coords0: np.ndarray,
    triangles: np.ndarray,
    fixed_nodes: np.ndarray,
    center_xy: np.ndarray,
    settings: ConicExample2Settings,
):
    coords_ref = coords0.copy()
    coords_eq = coords0.copy()

    n_elem = triangles.shape[0]
    sigma_rad_target = np.full(n_elem, settings.target_sigma_radial, dtype=float)
    sigma_circ_target = np.full(n_elem, settings.target_sigma_circ, dtype=float)
    area0 = np.maximum(element_areas(coords0, triangles), 1e-14)

    history = []
    membrane_weights = np.zeros((n_elem, 3), dtype=float)

    for it in range(1, settings.max_outer_iterations + 1):
        membrane_weights = compute_membrane_weights(
            coords_ref=coords_ref,
            triangles=triangles,
            center_xy=center_xy,
            sigma_radial_elem=sigma_rad_target,
            sigma_circ_elem=sigma_circ_target,
            min_weight=settings.min_weight,
        )
        coords_eq = solve_equilibrium(
            coords_seed=coords_ref,
            fixed_nodes=fixed_nodes,
            triangles=triangles,
            membrane_weights=membrane_weights,
            optimise_xyz=settings.optimise_xyz,
        )

        new_areas = element_areas(coords_eq, triangles)
        if np.min(new_areas) < 1e-12:
            coords_eq = 0.5 * (coords_ref + coords_eq)
            new_areas = element_areas(coords_eq, triangles)
            if np.min(new_areas) < 1e-12:
                raise RuntimeError("Degenerate elements detected. Consider refining mesh or reducing targets.")

        sigma_rad_eq, sigma_circ_eq, area_eq = compute_equilibrium_stresses(
            coords=coords_eq,
            triangles=triangles,
            membrane_weights=membrane_weights,
            center_xy=center_xy,
        )

        if settings.relax_distorted_elements and settings.distortion_limit > 1.0:
            area_ratio = area_eq / area0
            inside = (
                (area_ratio >= 1.0 / settings.distortion_limit)
                & (area_ratio <= settings.distortion_limit)
            )
        else:
            inside = np.ones(n_elem, dtype=bool)
        conv_mask = inside if np.any(inside) else np.ones(n_elem, dtype=bool)

        rel_r = abs(np.mean(sigma_rad_eq[conv_mask]) - settings.target_sigma_radial) / max(
            abs(settings.target_sigma_radial), 1e-12
        )
        rel_c = abs(np.mean(sigma_circ_eq[conv_mask]) - settings.target_sigma_circ) / max(
            abs(settings.target_sigma_circ), 1e-12
        )
        conv = max(rel_r, rel_c)
        history.append(
            {
                "iteration": it,
                "conv": conv,
                "mean_radial": float(np.mean(sigma_rad_eq[conv_mask])),
                "mean_circ": float(np.mean(sigma_circ_eq[conv_mask])),
                "active_elements": int(np.sum(conv_mask)),
            }
        )
        print(
            f"[Conic Ex2] iter={it:02d} conv={conv:.3e} | "
            f"mean(radial)={np.mean(sigma_rad_eq[conv_mask]):.3f}, "
            f"mean(circ)={np.mean(sigma_circ_eq[conv_mask]):.3f}"
        )
        if conv < settings.stress_tolerance:
            break

        coords_ref = coords_eq.copy()
        if settings.relax_distorted_elements and settings.distortion_limit > 1.0:
            distorted = ~inside
            sigma_rad_target = np.full(n_elem, settings.target_sigma_radial, dtype=float)
            sigma_circ_target = np.full(n_elem, settings.target_sigma_circ, dtype=float)
            sigma_rad_target[distorted] = sigma_rad_eq[distorted]
            sigma_circ_target[distorted] = sigma_circ_eq[distorted]

    return {
        "coords": coords_eq,
        "history": history,
        "sigma_radial": sigma_rad_eq,
        "sigma_circ": sigma_circ_eq,
        "membrane_weights": membrane_weights,
    }


def build_plot(
    coords: np.ndarray,
    triangles: np.ndarray,
    fixed_nodes: np.ndarray,
    title: str,
    out_html: str = "conic_ex2_formfound.html",
):
    if not HAVE_PLOTLY:
        print("Plotly is not installed. Skipping interactive plot generation.")
        return

    all_nodes = np.arange(coords.shape[0], dtype=int)
    free_nodes = np.setdiff1d(all_nodes, fixed_nodes)

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            intensity=coords[:, 2],
            colorscale="Viridis",
            opacity=0.68,
            colorbar=dict(title="Z (m)", thickness=15),
            name="Conic membrane",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=coords[free_nodes, 0],
            y=coords[free_nodes, 1],
            z=coords[free_nodes, 2],
            mode="markers",
            marker=dict(size=2.6, color="steelblue"),
            name="Free nodes",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=coords[fixed_nodes, 0],
            y=coords[fixed_nodes, 1],
            z=coords[fixed_nodes, 2],
            mode="markers",
            marker=dict(size=4.5, color="black"),
            name="Fixed boundary/ring nodes",
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X (m)"),
            yaxis=dict(title="Y (m)"),
            zaxis=dict(title="Z (m)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.8, z=1.2)),
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        height=760,
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved interactive conic plot -> {out_html}")


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
