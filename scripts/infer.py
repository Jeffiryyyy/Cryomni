from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import mrcfile
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cryomni import load_model
from data_processing.Gen_Box import generate_data_pair
from data_processing.Resize_Map import Resize_Map
from data_processing.Unify_Map import Unify_Map
from data_processing.map_utils import segment_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cryomni inference on MRC maps.")
    parser.add_argument("input_maps", nargs="+", help="Input .mrc/.map files.")
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoint" / "Cryomni.pt"),
        help="Path to Cryomni checkpoint file or save_pretrained directory.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "config.json"),
        help="Model config JSON used when --checkpoint is a single .pt file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs"),
        help="Directory for output MRC files.",
    )
    parser.add_argument(
        "--contour",
        type=float,
        default=None,
        help="Contour level for all maps. If omitted, --contour-json is used when provided.",
    )
    parser.add_argument(
        "--contour-json",
        default=None,
        help="Optional JSON mapping map id/stem to contour level.",
    )
    parser.add_argument("--box-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--masking-prob", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep unified/resized/segmented intermediate maps.",
    )
    return parser.parse_args()


def write_map(tensor: torch.Tensor, save_path: Path, reference_mrc_path: Path) -> None:
    if tensor.ndim != 5 or tensor.shape[:2] != (1, 1):
        raise ValueError("Expected one 3D map with shape (1, 1, D, H, W)")
    array = tensor[0, 0].detach().cpu().numpy().astype(np.float32)

    with mrcfile.open(reference_mrc_path, permissive=True) as ref_mrc:
        origin = ref_mrc.header.origin
        mapc = ref_mrc.header.mapc
        mapr = ref_mrc.header.mapr
        maps = ref_mrc.header.maps
        voxel_size = ref_mrc.voxel_size
        nxstart = ref_mrc.header.nxstart
        nystart = ref_mrc.header.nystart
        nzstart = ref_mrc.header.nzstart

    with mrcfile.new(save_path, overwrite=True) as mrc:
        mrc.set_data(array)
        mrc.header.origin = origin
        mrc.voxel_size = voxel_size
        mrc.header.mapc = mapc
        mrc.header.mapr = mapr
        mrc.header.maps = maps
        mrc.header.nxstart = nxstart
        mrc.header.nystart = nystart
        mrc.header.nzstart = nzstart
        mrc.update_header_from_data()


def load_contour_map(contour_json: str | None) -> dict[str, float]:
    if contour_json is None:
        return {}

    with Path(contour_json).open("r") as f:
        return {str(k): float(v) for k, v in json.load(f).items()}


def contour_for_map(
    map_path: Path,
    contour: float | None,
    contour_by_id: dict[str, float],
) -> float:
    if contour is not None:
        return contour
    return contour_by_id.get(map_path.stem, -1.0)


def preprocess_map(
    map_path: Path,
    work_dir: Path,
    contour: float,
    box_size: int,
    stride: int,
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int, int, int]], list[int], Path, list[Path]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    map_id = map_path.stem

    unified_map = work_dir / f"{map_id}_unified.mrc"
    resized_map = work_dir / f"{map_id}_resized.mrc"
    segmented_map = work_dir / f"{map_id}_segment.mrc"
    normalized_map = work_dir / f"{map_id}_normal.mrc"
    log_path = work_dir / "preprocess_log.txt"

    Unify_Map(str(map_path), str(unified_map))
    Resize_Map(str(unified_map), str(resized_map))
    segment_map(str(resized_map), str(segmented_map), contour)

    input_list, box_list, meaningful_indices = generate_data_pair(
        str(segmented_map),
        None,
        None,
        str(work_dir),
        contour,
        box_size,
        stride,
        map_id,
        visual_check=0,
        log_path=str(log_path),
    )

    return input_list, box_list, meaningful_indices, normalized_map, [
        unified_map,
        resized_map,
        segmented_map,
    ]


def new_output_volume(box_list: list[tuple[int, int, int, int, int, int]]) -> torch.Tensor:
    max_x = max(coord[1] for coord in box_list)
    max_y = max(coord[3] for coord in box_list)
    max_z = max(coord[5] for coord in box_list)
    return torch.zeros((1, 1, max_x, max_y, max_z), dtype=torch.float32)


def insert_box(
    volume: torch.Tensor,
    patch: torch.Tensor,
    box: tuple[int, int, int, int, int, int],
) -> None:
    x_start, x_end, y_start, y_end, z_start, z_end = box
    dx = x_end - x_start
    dy = y_end - y_start
    dz = z_end - z_start
    volume[0, 0, x_start:x_end, y_start:y_end, z_start:z_end] = patch[
        0, 0, :dx, :dy, :dz
    ].cpu()


def run_single_map(
    model: torch.nn.Module,
    map_path: Path,
    output_dir: Path,
    contour: float,
    box_size: int,
    stride: int,
    device: torch.device,
    keep_intermediate: bool,
) -> None:
    map_id = map_path.stem
    work_dir = output_dir / "preprocessed" / map_id
    input_list, box_list, meaningful_indices, normal_map, intermediate_paths = preprocess_map(
        map_path,
        work_dir,
        contour,
        box_size,
        stride,
    )

    if not meaningful_indices:
        raise RuntimeError(f"No meaningful boxes found for {map_path}")

    mask_volume = new_output_volume(box_list)
    recon_volume = new_output_volume(box_list)

    for idx in tqdm(meaningful_indices, desc=f"Inference {map_id}", leave=False):
        box_array = input_list[idx]
        box_tensor = torch.from_numpy(box_array).to(torch.float32)
        box_tensor = box_tensor.unsqueeze(0).unsqueeze(0).to(device)

        with torch.inference_mode():
            pre_patch, post_patch, mask_patch = model.forward_pred(box_tensor)
            recon_patch = post_patch + mask_patch

        insert_box(mask_volume, mask_patch, box_list[idx])
        insert_box(recon_volume, recon_patch, box_list[idx])

    output_dir.mkdir(parents=True, exist_ok=True)
    write_map(mask_volume, output_dir / f"mask_{map_id}.mrc", normal_map)
    write_map(recon_volume, output_dir / f"recon_{map_id}.mrc", normal_map)

    if not keep_intermediate:
        for path in intermediate_paths:
            if path.exists() or path.is_symlink():
                path.unlink()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    contour_by_id = load_contour_map(args.contour_json)

    model = load_model(args.checkpoint, args.config, map_location="cpu")
    model.masking_prob = args.masking_prob
    model.to(device)
    model.eval()

    for input_map in args.input_maps:
        map_path = Path(input_map)
        contour = contour_for_map(map_path, args.contour, contour_by_id)
        print(f"Processing {map_path} with contour {contour}")
        run_single_map(
            model=model,
            map_path=map_path,
            output_dir=output_dir,
            contour=contour,
            box_size=args.box_size,
            stride=args.stride,
            device=device,
            keep_intermediate=args.keep_intermediate,
        )


if __name__ == "__main__":
    main()
