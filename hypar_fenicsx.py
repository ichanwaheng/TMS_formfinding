#!/usr/bin/env python3
"""Build and visualize a hypar membrane mesh with FEniCSx."""

from __future__ import annotations

import numpy as np
from mpi4py import MPI
from dolfinx import mesh
import plotly.graph_objects as go


def build_hypar(
    lx: float = 4.0,
    ly: float = 4.0,
    h: float = 2.0,
    nx: int = 40,
    ny: int = 40,
) -> tuple[mesh.Mesh, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Create a triangulated rectangle, embed in 3D, and warp to a hypar."""
    # Use 3D points so geometry has (x, y, z) coordinates.
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        points=((0.0, 0.0, 0.0), (lx, ly, 0.0)),
        n=(nx, ny),
        cell_type=mesh.CellType.triangle,
    )

    coords = domain.geometry.x
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    z_coords = h * (x_coords / lx) * (1.0 - y_coords / ly) + h * (
        1.0 - x_coords / lx
    ) * (y_coords / ly)
    coords[:, 2] = z_coords

    def on_boundary(x: np.ndarray) -> np.ndarray:
        return np.logical_or.reduce(
            (
                np.isclose(x[0], 0.0),
                np.isclose(x[0], lx),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], ly),
            )
        )

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, on_boundary)

    # Per-edge facet tags for downstream cable definitions.
    facets_bottom = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[1], 0.0)
    )
    facets_top = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[1], ly)
    )
    facets_left = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], 0.0)
    )
    facets_right = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], lx)
    )

    all_facets = np.concatenate([facets_bottom, facets_top, facets_left, facets_right])
    all_tags = np.concatenate(
        [
            np.full(len(facets_bottom), 1, dtype=np.int32),  # bottom
            np.full(len(facets_top), 2, dtype=np.int32),  # top
            np.full(len(facets_left), 3, dtype=np.int32),  # left
            np.full(len(facets_right), 4, dtype=np.int32),  # right
        ]
    )
    sort_idx = np.argsort(all_facets)
    mesh.meshtags(domain, fdim, all_facets[sort_idx], all_tags[sort_idx])

    corners = {
        "SW (0,0)": mesh.locate_entities(
            domain, 0, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
        ),
        "SE (Lx,0)": mesh.locate_entities(
            domain, 0, lambda x: np.isclose(x[0], lx) & np.isclose(x[1], 0.0)
        ),
        "NW (0,Ly)": mesh.locate_entities(
            domain, 0, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], ly)
        ),
        "NE (Lx,Ly)": mesh.locate_entities(
            domain, 0, lambda x: np.isclose(x[0], lx) & np.isclose(x[1], ly)
        ),
    }

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    cell_to_vertices = domain.topology.connectivity(tdim, 0)
    num_cells = domain.topology.index_map(tdim).size_local + domain.topology.index_map(
        tdim
    ).num_ghosts
    triangles = np.array([cell_to_vertices.links(i) for i in range(num_cells)])

    domain.topology.create_connectivity(fdim, 0)
    edge_to_vertices = domain.topology.connectivity(fdim, 0)
    cables = np.array([edge_to_vertices.links(i) for i in boundary_facets])

    return domain, triangles, cables, corners


def plot_hypar(
    domain: mesh.Mesh,
    triangles: np.ndarray,
    cables: np.ndarray,
    corners: dict[str, np.ndarray],
    output_html: str = "hypar_fenicsx.html",
) -> None:
    coords = domain.geometry.x

    lx = float(np.max(coords[:, 0]))
    ly = float(np.max(coords[:, 1]))
    h = float(np.max(coords[:, 2]))

    def on_boundary(x: np.ndarray) -> np.ndarray:
        return np.logical_or.reduce(
            (
                np.isclose(x[0], 0.0),
                np.isclose(x[0], lx),
                np.isclose(x[1], 0.0),
                np.isclose(x[1], ly),
            )
        )

    is_boundary = on_boundary(coords.T)
    free_idx = np.where(~is_boundary)[0]
    boundary_idx = np.where(is_boundary)[0]

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
            opacity=0.55,
            colorbar=dict(title="Z (m)", thickness=15),
            name="Membrane surface",
            showscale=True,
            flatshading=False,
        )
    )

    ex: list[float | None] = []
    ey: list[float | None] = []
    ez: list[float | None] = []
    for tri in triangles:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            p1, p2 = coords[tri[a]], coords[tri[b]]
            ex += [p1[0], p2[0], None]
            ey += [p1[1], p2[1], None]
            ez += [p1[2], p2[2], None]

    fig.add_trace(
        go.Scatter3d(
            x=ex,
            y=ey,
            z=ez,
            mode="lines",
            line=dict(color="rgba(60,60,60,0.35)", width=1),
            name="Mesh edges",
            hoverinfo="skip",
        )
    )

    cx: list[float | None] = []
    cy: list[float | None] = []
    cz: list[float | None] = []
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
            x=coords[free_idx, 0],
            y=coords[free_idx, 1],
            z=coords[free_idx, 2],
            mode="markers",
            marker=dict(size=4, color="steelblue", opacity=0.9),
            name="Free nodes",
            text=[
                (
                    f"Node {i}<br>({coords[i,0]:.3f}, "
                    f"{coords[i,1]:.3f}, {coords[i,2]:.3f})"
                )
                for i in free_idx
            ],
            hoverinfo="text",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=coords[boundary_idx, 0],
            y=coords[boundary_idx, 1],
            z=coords[boundary_idx, 2],
            mode="markers",
            marker=dict(size=5, color="red", opacity=1.0),
            name="Boundary nodes",
            text=[
                (
                    f"Node {i}<br>({coords[i,0]:.3f}, "
                    f"{coords[i,1]:.3f}, {coords[i,2]:.3f})"
                )
                for i in boundary_idx
            ],
            hoverinfo="text",
        )
    )

    corner_x: list[float] = []
    corner_y: list[float] = []
    corner_z: list[float] = []
    corner_text: list[str] = []
    for name, idx_arr in corners.items():
        if len(idx_arr) > 0:
            idx = idx_arr[0]
            corner_x.append(coords[idx, 0])
            corner_y.append(coords[idx, 1])
            corner_z.append(coords[idx, 2] + 0.15)
            corner_text.append(name.split()[0])

    fig.add_trace(
        go.Scatter3d(
            x=corner_x,
            y=corner_y,
            z=corner_z,
            mode="text",
            text=corner_text,
            textfont=dict(size=14, color="black"),
            name="Corners",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "Hypar - FEniCSx mesh<br>"
                f"<sup>{coords.shape[0]} nodes · "
                f"{len(triangles)} elements · "
                f"{len(cables)} boundary cables</sup>"
            ),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="X (m)"),
            yaxis=dict(title="Y (m)"),
            zaxis=dict(title="Z (m)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.8, z=1.3)),
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="gray",
            borderwidth=0.5,
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        height=680,
    )

    fig.write_html(output_html)
    print(f"Saved -> {output_html}")


def main() -> None:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Run this script with a single MPI process.")

    lx, ly = 4.0, 4.0
    h = 2.0
    nx, ny = 40, 40

    domain, triangles, cables, corners = build_hypar(
        lx=lx,
        ly=ly,
        h=h,
        nx=nx,
        ny=ny,
    )

    print(f"Mesh topology dimension: {domain.topology.dim}")
    print(f"Geometry dimension:      {domain.geometry.dim}")
    print(
        f"Number of cells (triangles): "
        f"{domain.topology.index_map(domain.topology.dim).size_global}"
    )
    print(f"Number of nodes:            {domain.topology.index_map(0).size_global}")

    coords = domain.geometry.x
    print("\nCorner node heights:")
    for name, idx_arr in corners.items():
        if len(idx_arr) > 0:
            idx = idx_arr[0]
            xyz = coords[idx]
            print(
                f"  {name} -> node {idx:3d}, "
                f"coords ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.3f}) m"
            )

    plot_hypar(domain, triangles, cables, corners, output_html="hypar_fenicsx.html")


if __name__ == "__main__":
    main()
