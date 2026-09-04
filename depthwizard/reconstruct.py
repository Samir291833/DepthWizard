"""Lightweight textured surface reconstruction for the future Three.js viewer.

The mesh is a regular decimated grid. Vertex ``source_row``/``source_col``
mapping is preserved in the sidecar, while UVs use the same source coordinates
and therefore remain stable for texture lookup and future point picking.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MeshData:
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray
    source_rows: np.ndarray
    source_cols: np.ndarray
    grid_shape: tuple[int, int]
    surface_kind: str


@dataclass(frozen=True)
class ReconstructionResult:
    glb_path: Path
    texture_path: Path
    metadata_path: Path
    vertices: int
    triangles: int
    source_shape: tuple[int, int]
    mesh_shape: tuple[int, int]


def _sample_indices(length: int, limit: int) -> np.ndarray:
    if length < 2:
        raise ValueError("surface dimensions must each be at least two pixels")
    count = min(length, max(2, int(limit)))
    return np.unique(np.rint(np.linspace(0, length - 1, count)).astype(np.int32))


def build_mesh(surface: np.ndarray, rgb: np.ndarray, *, max_side: int = 128,
               surface_kind: str = "relative") -> MeshData:
    """Build a decimated regular grid while retaining exact source indices.

    ``surface_kind`` is metadata, not a conversion: ``relative`` stays unitless
    and ``metric`` keeps the supplied elevation values in metres.
    """
    if surface.ndim != 2 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected surface (H,W) and RGB (H,W,3) arrays")
    if surface.shape != rgb.shape[:2]:
        raise ValueError(f"surface {surface.shape} and RGB {rgb.shape[:2]} must match")
    if surface_kind not in {"relative", "metric"}:
        raise ValueError("surface_kind must be 'relative' or 'metric'")

    rows = _sample_indices(surface.shape[0], max_side)
    cols = _sample_indices(surface.shape[1], max_side)
    sampled = np.nan_to_num(surface[np.ix_(rows, cols)], nan=0.0).astype(np.float32)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    height, width = surface.shape
    x = col_grid.astype(np.float32) / float(width - 1)
    y = -row_grid.astype(np.float32) / float(height - 1)
    positions = np.stack((x, y, sampled), axis=-1).reshape(-1, 3)
    u = col_grid.astype(np.float32) / float(width - 1)
    v = row_grid.astype(np.float32) / float(height - 1)
    uvs = np.stack((u, v), axis=-1).reshape(-1, 2)

    dz_drow, dz_dcol = np.gradient(sampled.astype(np.float32))
    normals = np.stack((dz_dcol, dz_drow, np.ones_like(sampled)), axis=-1).reshape(-1, 3)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)

    mesh_rows, mesh_cols = len(rows), len(cols)
    top_left = np.arange((mesh_rows - 1) * (mesh_cols - 1), dtype=np.uint32)
    top_left = top_left + (top_left // (mesh_cols - 1))
    col_offset = np.arange(mesh_cols - 1, dtype=np.uint32)
    top_left = (np.arange(mesh_rows - 1, dtype=np.uint32)[:, None] * mesh_cols + col_offset).reshape(-1)
    a, b, c, d = top_left, top_left + 1, top_left + mesh_cols, top_left + mesh_cols + 1
    indices = np.stack((a, c, b, b, c, d), axis=1).reshape(-1).astype(np.uint32)
    return MeshData(positions, normals, uvs, indices, row_grid.reshape(-1), col_grid.reshape(-1),
                    (mesh_rows, mesh_cols), surface_kind)


def _pad4(data: bytes) -> bytes:
    return data + b"\x00" * ((4 - len(data) % 4) % 4)


def _accessor(gltf: dict, view: int, component_type: int, count: int, kind: str,
              minimum=None, maximum=None) -> int:
    item = {"bufferView": view, "componentType": component_type, "count": count, "type": kind}
    if minimum is not None:
        item["min"], item["max"] = minimum, maximum
    gltf["accessors"].append(item)
    return len(gltf["accessors"]) - 1


def _append_view(gltf: dict, binary: bytearray, payload: bytes, target: int | None = None) -> int:
    offset = len(binary)
    binary.extend(_pad4(payload))
    view = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
    if target is not None:
        view["target"] = target
    gltf["bufferViews"].append(view)
    return len(gltf["bufferViews"]) - 1


def _write_glb(path: Path, mesh: MeshData, texture_png: bytes) -> None:
    binary = bytearray()
    gltf = {
        "asset": {"version": "2.0", "generator": "DepthWizard Increment 3"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}], "meshes": [], "buffers": [],
        "bufferViews": [], "accessors": [], "images": [{"bufferView": 0, "mimeType": "image/png"}],
        "textures": [{"source": 0}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                                    "metallicFactor": 0.0, "roughnessFactor": 1.0},
                        "doubleSided": True}],
    }
    position_view = _append_view(gltf, binary, mesh.positions.astype("<f4").tobytes(), 34962)
    normal_view = _append_view(gltf, binary, mesh.normals.astype("<f4").tobytes(), 34962)
    uv_view = _append_view(gltf, binary, mesh.uvs.astype("<f4").tobytes(), 34962)
    index_view = _append_view(gltf, binary, mesh.indices.astype("<u4").tobytes(), 34963)
    position = _accessor(gltf, position_view, 5126, len(mesh.positions), "VEC3",
                         mesh.positions.min(axis=0).tolist(), mesh.positions.max(axis=0).tolist())
    normal = _accessor(gltf, normal_view, 5126, len(mesh.normals), "VEC3")
    uv = _accessor(gltf, uv_view, 5126, len(mesh.uvs), "VEC2", [0.0, 0.0], [1.0, 1.0])
    indices = _accessor(gltf, index_view, 5125, len(mesh.indices), "SCALAR",
                        [int(mesh.indices.min())], [int(mesh.indices.max())])
    image_view = _append_view(gltf, binary, texture_png)
    gltf["images"][0]["bufferView"] = image_view
    gltf["meshes"].append({"primitives": [{"attributes": {"POSITION": position, "NORMAL": normal, "TEXCOORD_0": uv},
                                              "indices": indices, "material": 0}]})
    gltf_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_chunk = _pad4(gltf_json)
    bin_chunk = _pad4(bytes(binary))
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    gltf["buffers"] = [{"byteLength": len(bin_chunk)}]
    gltf_json = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"))
    total = 12 + 8 + len(gltf_json) + 8 + len(bin_chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(gltf_json), 0x4E4F534A)); fh.write(gltf_json)
        fh.write(struct.pack("<II", len(bin_chunk), 0x004E4942)); fh.write(bin_chunk)


def reconstruct(surface: np.ndarray, rgb: np.ndarray, out_dir: str | Path, *,
                stem: str = "surface", max_side: int = 128, surface_kind: str = "relative",
                crs=None, transform=None) -> ReconstructionResult:
    """Write ``stem.glb``, ``stem_texture.png``, and ``stem_metadata.json``."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mesh = build_mesh(surface, rgb, max_side=max_side, surface_kind=surface_kind)
    texture_path = out_dir / f"{stem}_texture.png"
    Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8)), mode="RGB").save(texture_path, format="PNG")
    texture_png = texture_path.read_bytes()
    glb_path = out_dir / f"{stem}.glb"
    _write_glb(glb_path, mesh, texture_png)
    metadata = {
        "schema": "depthwizard.reconstruction.v1", "surface_kind": surface_kind,
        "source_shape": list(surface.shape), "mesh_shape": list(mesh.grid_shape),
        "vertex_count": int(len(mesh.positions)), "triangle_count": int(len(mesh.indices) // 3),
        "max_side": max_side, "texture": texture_path.name,
        "coordinates": {"x": "source_col/(width-1)", "y": "-source_row/(height-1)",
                         "z": "surface value; unitless for relative, metres for metric"},
        "source_rows": mesh.source_rows.tolist(), "source_cols": mesh.source_cols.tolist(),
        "crs": str(crs) if crs is not None else None,
        "transform": list(transform) if transform is not None else None,
        "georeferencing": "mesh vertex source rows/cols map through source transform" if crs is not None else None,
    }
    metadata_path = out_dir / f"{stem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ReconstructionResult(glb_path, texture_path, metadata_path, len(mesh.positions),
                                len(mesh.indices) // 3, surface.shape, mesh.grid_shape)