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
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


@dataclass
class TwinHyparSettings:
    # Geometry
    Lx: float = 12.0
    Ly: float = 6.0
    Nx: int = 80
    Ny: int = 40
    height_difference: float = 4.0  # high - low support level (m)

    # Prestress targets (kN/m)
    target_sigma_warp: float = 50.0
    target_sigma_fill: float = 5.0

    # Cable target force (kN). If None, estimated from cable material + strain.
    target_cable_force: float | None = None
    cable_prestress_strain: float = 0.002

    # UWM controls
    max_outer_iterations: int = 35
    stress_tolerance: float = 1e-4
    distortion_limit: float = 1.2
    relax_distorted_elements: bool = True
    optimise_xyz: bool = True
    min_weight: float = 1e-12

    # Material properties (stored and reported for traceability)
    warp_modulus: float = 1400.0   # kN/m
    fill_modulus: float = 800.0    # kN/m
    shear_modulus: float = 100.0   # kN/m
    poisson_ratio: float = 0.8
    cable_diameter_mm: float = 19.0
    cable_modulus_kN_per_mm2: float = 205.0


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
            raise ValueError("Zero-length pseudo-cable side encountered.")
        lengths[j] = lj
        side_dirs_2d[j] = dv / lj

    return area, lengths, side_dirs_2d, e1, e2, normal


def project_warp_fill_to_local(
    normal: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    warp_axis: np.ndarray,
    fill_axis: np.ndarray,
):
    w3 = warp_axis - np.dot(warp_axis, normal) * normal
    w3 = normalize(w3, fallback=e1)

    f3 = fill_axis - np.dot(fill_axis, normal) * normal
    f3 = f3 - np.dot(f3, w3) * w3
    f3 = normalize(f3, fallback=np.cross(normal, w3))

    w2 = np.array([np.dot(w3, e1), np.dot(w3, e2)])
    f2 = np.array([np.dot(f3, e1), np.dot(f3, e2)])
    w2 = normalize(w2, fallback=np.array([1.0, 0.0]))
    f2 = normalize(f2, fallback=np.array([0.0, 1.0]))
    return w2, f2


def desired_local_stress_tensor(
    sigma_fill: float,
    sigma_warp: float,
    w2: np.ndarray,
    f2: np.ndarray,
) -> np.ndarray:
    return sigma_fill * np.outer(f2, f2) + sigma_warp * np.outer(w2, w2)


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
    sigma_fill_elem: np.ndarray,
    sigma_warp_elem: np.ndarray,
    warp_axis: np.ndarray,
    fill_axis: np.ndarray,
    min_weight: float,
) -> np.ndarray:
    n_elem = triangles.shape[0]
    W = np.zeros((n_elem, 3), dtype=float)
    for i, tri in enumerate(triangles):
        area, lengths, side_dirs, e1, e2, normal = triangle_frame_and_sides(coords_ref, tri)
        w2, f2 = project_warp_fill_to_local(normal, e1, e2, warp_axis, fill_axis)
        sigma_tensor = desired_local_stress_tensor(sigma_fill_elem[i], sigma_warp_elem[i], w2, f2)
        sigma_n = natural_stress_from_tensor(sigma_tensor, side_dirs)
        side_forces = area * sigma_n / lengths
        W_i = side_forces / (2.0 * lengths)
        W[i] = np.maximum(W_i, min_weight)
    return W


def compute_cable_weights(
    coords_ref: np.ndarray,
    cables: np.ndarray,
    target_cable_force: float,
    min_weight: float,
) -> np.ndarray:
    if cables.size == 0 or target_cable_force <= 0.0:
        return np.zeros(cables.shape[0], dtype=float)
    p0 = coords_ref[cables[:, 0]]
    p1 = coords_ref[cables[:, 1]]
    lengths = np.linalg.norm(p1 - p0, axis=1)
    lengths = np.maximum(lengths, 1e-14)
    w = target_cable_force / (2.0 * lengths)
    return np.maximum(w, min_weight)


def assemble_weighted_laplacian(
    n_nodes: int,
    triangles: np.ndarray,
    membrane_weights: np.ndarray,
    cables: np.ndarray,
    cable_weights: np.ndarray,
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

    for i, (a, b) in enumerate(cables):
        if cable_weights.size == 0:
            break
        add_edge(int(a), int(b), float(cable_weights[i]))

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
    cables: np.ndarray,
    cable_weights: np.ndarray,
    optimise_xyz: bool,
) -> np.ndarray:
    n_nodes = coords_seed.shape[0]
    all_nodes = np.arange(n_nodes, dtype=int)
    fixed_nodes = np.unique(fixed_nodes.astype(int))
    free_nodes = np.setdiff1d(all_nodes, fixed_nodes)
    if free_nodes.size == 0:
        return coords_seed.copy()

    L = assemble_weighted_laplacian(n_nodes, triangles, membrane_weights, cables, cable_weights)
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
    warp_axis: np.ndarray,
    fill_axis: np.ndarray,
):
    n_elem = triangles.shape[0]
    sigma_fill = np.zeros(n_elem, dtype=float)
    sigma_warp = np.zeros(n_elem, dtype=float)
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

        w2, f2 = project_warp_fill_to_local(normal, e1, e2, warp_axis, fill_axis)
        sigma_warp[i] = float(w2 @ stress_tensor @ w2)
        sigma_fill[i] = float(f2 @ stress_tensor @ f2)
    return sigma_fill, sigma_warp, areas


def compute_cable_forces(coords: np.ndarray, cables: np.ndarray, cable_weights: np.ndarray) -> np.ndarray:
    if cables.size == 0 or cable_weights.size == 0:
        return np.zeros(0, dtype=float)
    p0 = coords[cables[:, 0]]
    p1 = coords[cables[:, 1]]
    lengths = np.linalg.norm(p1 - p0, axis=1)
    return 2.0 * cable_weights * lengths


def element_areas(coords: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    areas = np.zeros(triangles.shape[0], dtype=float)
    for i, tri in enumerate(triangles):
        p0, p1, p2 = coords[tri[0]], coords[tri[1]], coords[tri[2]]
        areas[i] = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
    return areas


def run_uwm(
    coords0: np.ndarray,
    triangles: np.ndarray,
    cables: np.ndarray,
    fixed_nodes: np.ndarray,
    settings: TwinHyparSettings,
    warp_axis: np.ndarray | None = None,
    fill_axis: np.ndarray | None = None,
):
    if warp_axis is None:
        warp_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    if fill_axis is None:
        fill_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    if settings.target_cable_force is None:
        raise ValueError("settings.target_cable_force must be set before calling run_uwm().")

    coords_ref = coords0.copy()
    coords_eq = coords0.copy()
    n_elem = triangles.shape[0]
    sigma_fill_target_elem = np.full(n_elem, settings.target_sigma_fill, dtype=float)
    sigma_warp_target_elem = np.full(n_elem, settings.target_sigma_warp, dtype=float)
    area0 = np.maximum(element_areas(coords0, triangles), 1e-14)

    history = []
    membrane_weights = np.zeros((n_elem, 3), dtype=float)
    cable_weights = np.zeros(cables.shape[0], dtype=float)

    for it in range(1, settings.max_outer_iterations + 1):
        membrane_weights = compute_membrane_weights(
            coords_ref=coords_ref,
            triangles=triangles,
            sigma_fill_elem=sigma_fill_target_elem,
            sigma_warp_elem=sigma_warp_target_elem,
            warp_axis=warp_axis,
            fill_axis=fill_axis,
            min_weight=settings.min_weight,
        )
        cable_weights = compute_cable_weights(
            coords_ref=coords_ref,
            cables=cables,
            target_cable_force=settings.target_cable_force,
            min_weight=settings.min_weight,
        )
        coords_eq = solve_equilibrium(
            coords_seed=coords_ref,
            fixed_nodes=fixed_nodes,
            triangles=triangles,
            membrane_weights=membrane_weights,
            cables=cables,
            cable_weights=cable_weights,
            optimise_xyz=settings.optimise_xyz,
        )

        sigma_fill_eq, sigma_warp_eq, area_eq = compute_equilibrium_stresses(
            coords=coords_eq,
            triangles=triangles,
            membrane_weights=membrane_weights,
            warp_axis=warp_axis,
            fill_axis=fill_axis,
        )
        cable_forces = compute_cable_forces(coords_eq, cables, cable_weights)

        if settings.relax_distorted_elements and settings.distortion_limit > 1.0:
            area_ratio = area_eq / area0
            inside = (
                (area_ratio >= 1.0 / settings.distortion_limit)
                & (area_ratio <= settings.distortion_limit)
            )
        else:
            inside = np.ones(n_elem, dtype=bool)
        conv_mask = inside if np.any(inside) else np.ones(n_elem, dtype=bool)

        rel_f = abs(np.mean(sigma_fill_eq[conv_mask]) - settings.target_sigma_fill) / max(
            abs(settings.target_sigma_fill), 1e-12
        )
        rel_w = abs(np.mean(sigma_warp_eq[conv_mask]) - settings.target_sigma_warp) / max(
            abs(settings.target_sigma_warp), 1e-12
        )
        conv = max(rel_f, rel_w)
        history.append(
            {
                "iteration": it,
                "conv": conv,
                "mean_fill": float(np.mean(sigma_fill_eq[conv_mask])),
                "mean_warp": float(np.mean(sigma_warp_eq[conv_mask])),
                "active_elements": int(np.sum(conv_mask)),
            }
        )
        print(
            f"[Twin-Hypar UWM] iter={it:02d} conv={conv:.3e} | "
            f"mean(fill)={np.mean(sigma_fill_eq[conv_mask]):.3f}, "
            f"mean(warp)={np.mean(sigma_warp_eq[conv_mask]):.3f}"
        )
        if conv < settings.stress_tolerance:
            break

        coords_ref = coords_eq.copy()
        if settings.relax_distorted_elements and settings.distortion_limit > 1.0:
            distorted = ~inside
            sigma_fill_target_elem = np.full(n_elem, settings.target_sigma_fill, dtype=float)
            sigma_warp_target_elem = np.full(n_elem, settings.target_sigma_warp, dtype=float)
            sigma_fill_target_elem[distorted] = sigma_fill_eq[distorted]
            sigma_warp_target_elem[distorted] = sigma_warp_eq[distorted]

    return {
        "coords": coords_eq,
        "history": history,
        "sigma_fill": sigma_fill_eq,
        "sigma_warp": sigma_warp_eq,
        "cable_forces": cable_forces,
        "membrane_weights": membrane_weights,
        "cable_weights": cable_weights,
    }


def build_structured_rectangle_mesh(Lx: float, Ly: float, Nx: int, Ny: int):
    xs = np.linspace(0.0, Lx, Nx + 1)
    ys = np.linspace(0.0, Ly, Ny + 1)

    def node_id(i: int, j: int) -> int:
        return j * (Nx + 1) + i

    coords = np.zeros(((Nx + 1) * (Ny + 1), 3), dtype=float)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            coords[node_id(i, j), :2] = [x, y]

    triangles = []
    for j in range(Ny):
        for i in range(Nx):
            n00 = node_id(i, j)
            n10 = node_id(i + 1, j)
            n01 = node_id(i, j + 1)
            n11 = node_id(i + 1, j + 1)
            triangles.append([n00, n10, n11])
            triangles.append([n00, n11, n01])
    triangles = np.array(triangles, dtype=int)

    cables = []
    for i in range(Nx):
        cables.append([node_id(i, 0), node_id(i + 1, 0)])
        cables.append([node_id(i, Ny), node_id(i + 1, Ny)])
    for j in range(Ny):
        cables.append([node_id(0, j), node_id(0, j + 1)])
        cables.append([node_id(Nx, j), node_id(Nx, j + 1)])
    cables = np.array(cables, dtype=int)
    return coords, triangles, cables


def locate_support_nodes(coords: np.ndarray, support_xy: np.ndarray) -> np.ndarray:
    ids = []
    xy_nodes = coords[:, :2]
    for xy in support_xy:
        dist2 = np.sum((xy_nodes - xy[None, :]) ** 2, axis=1)
        ids.append(int(np.argmin(dist2)))
    return np.unique(np.array(ids, dtype=int))


def idw_initial_surface(
    coords: np.ndarray,
    support_xy: np.ndarray,
    support_z: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    z = np.zeros(coords.shape[0], dtype=float)
    for i in range(coords.shape[0]):
        d = np.linalg.norm(support_xy - coords[i, :2], axis=1)
        exact = np.where(d < 1e-12)[0]
        if exact.size > 0:
            z[i] = support_z[exact[0]]
            continue
        w = 1.0 / np.power(d, power)
        z[i] = float(np.sum(w * support_z) / np.sum(w))
    return z


def estimate_cable_force_from_material(settings: TwinHyparSettings) -> float:
    # Assumes cable modulus in kN/mm^2 (steel-like value 205 kN/mm^2) and
    # force = E * A * strain.
    area_mm2 = 0.25 * np.pi * settings.cable_diameter_mm ** 2
    return settings.cable_modulus_kN_per_mm2 * area_mm2 * settings.cable_prestress_strain


def build_plot(
    coords: np.ndarray,
    triangles: np.ndarray,
    cables: np.ndarray,
    fixed_nodes: np.ndarray,
    title: str,
    out_html: str = "twin_hypar_formfound.html",
):
    if not HAVE_PLOTLY:
        raise RuntimeError("Plotly is required for interactive HTML plotting.")

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
            colorscale="RdBu",
            reversescale=True,
            opacity=0.65,
            colorbar=dict(title="Z (m)", thickness=15),
            name="Twin hypar membrane",
        )
    )

    cx, cy, cz = [], [], []
    for cab in cables:
        p1, p2 = coords[cab[0]], coords[cab[1]]
        cx += [p1[0], p2[0], None]
        cy += [p1[1], p2[1], None]
        cz += [p1[2], p2[2], None]
    fig.add_trace(
        go.Scatter3d(
            x=cx,
            y=cy,
            z=cz,
            mode="lines",
            line=dict(color="red", width=4),
            name="Boundary cables",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=coords[free_nodes, 0],
            y=coords[free_nodes, 1],
            z=coords[free_nodes, 2],
            mode="markers",
            marker=dict(size=2.8, color="steelblue"),
            name="Free nodes",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=coords[fixed_nodes, 0],
            y=coords[fixed_nodes, 1],
            z=coords[fixed_nodes, 2],
            mode="markers",
            marker=dict(size=7, color="black"),
            name="Support nodes (3H/3L)",
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X (m)"),
            yaxis=dict(title="Y (m)"),
            zaxis=dict(title="Z (m)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.7, z=1.1)),
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.75)"),
        height=760,
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved interactive twin-hypar plot -> {out_html}")


def main():
    settings = TwinHyparSettings()
    coords, triangles, cables = build_structured_rectangle_mesh(
        Lx=settings.Lx,
        Ly=settings.Ly,
        Nx=settings.Nx,
        Ny=settings.Ny,
    )

    # Six support points on perimeter, alternating 3 high and 3 low.
    support_xy = np.array(
        [
            [0.0, 0.0],
            [0.5 * settings.Lx, 0.0],
            [settings.Lx, 0.0],
            [settings.Lx, settings.Ly],
            [0.5 * settings.Lx, settings.Ly],
            [0.0, settings.Ly],
        ],
        dtype=float,
    )
    z_high = 0.5 * settings.height_difference
    z_low = -0.5 * settings.height_difference
    support_z = np.array([z_high, z_low, z_high, z_low, z_high, z_low], dtype=float)

    fixed_nodes = locate_support_nodes(coords, support_xy)
    coords[:, 2] = idw_initial_surface(coords, support_xy, support_z, power=2.0)
    coords[fixed_nodes, 2] = support_z

    if settings.target_cable_force is None:
        settings.target_cable_force = estimate_cable_force_from_material(settings)

    print("===== Twin Hypar Input Summary =====")
    print(f"Plan dimensions: {settings.Lx:.2f} m x {settings.Ly:.2f} m")
    print(f"Height difference (high-low): {settings.height_difference:.2f} m")
    print(f"Supports: {len(fixed_nodes)} nodes (3 high / 3 low)")
    print(f"Target prestress (warp/fill): {settings.target_sigma_warp:.2f} / {settings.target_sigma_fill:.2f} kN/m")
    print(f"Target cable force: {settings.target_cable_force:.2f} kN")
    print("Material properties:")
    print(f"  Warp modulus:  {settings.warp_modulus:.1f} kN/m")
    print(f"  Fill modulus:  {settings.fill_modulus:.1f} kN/m")
    print(f"  Shear modulus: {settings.shear_modulus:.1f} kN/m")
    print(f"  Poisson ratio: {settings.poisson_ratio:.2f}")
    print(
        f"  Cable: d={settings.cable_diameter_mm:.1f} mm, "
        f"E={settings.cable_modulus_kN_per_mm2:.1f} kN/mm^2"
    )

    results = run_uwm(
        coords0=coords.copy(),
        triangles=triangles,
        cables=cables,
        fixed_nodes=fixed_nodes,
        settings=settings,
        warp_axis=np.array([1.0, 0.0, 0.0]),
        fill_axis=np.array([0.0, 1.0, 0.0]),
    )
    coords[:] = results["coords"]

    sigma_f = results["sigma_fill"]
    sigma_w = results["sigma_warp"]
    cable_forces = results["cable_forces"]
    history = results["history"]
    print("\n===== Twin Hypar Final Summary =====")
    if history:
        print(f"Outer iterations: {history[-1]['iteration']}")
        print(f"Final convergence metric: {history[-1]['conv']:.3e}")
    print(f"Mean sigma_fill  (kN/m): {np.mean(sigma_f):.3f} | std: {np.std(sigma_f):.3f}")
    print(f"Mean sigma_warp  (kN/m): {np.mean(sigma_w):.3f} | std: {np.std(sigma_w):.3f}")
    if cable_forces.size > 0:
        print(f"Mean cable force (kN): {np.mean(cable_forces):.3f} | std: {np.std(cable_forces):.3f}")

    build_plot(
        coords=coords,
        triangles=triangles,
        cables=cables,
        fixed_nodes=fixed_nodes,
        title=(
            "Twin hypar UWM form-finding | "
            f"target(warp/fill)=({settings.target_sigma_warp:.1f}, {settings.target_sigma_fill:.1f}) kN/m"
        ),
        out_html="twin_hypar_formfound.html",
    )


if __name__ == "__main__":
    main()
