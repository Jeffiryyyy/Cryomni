import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=0, help="0: Map processing and label assignment")
    parser.add_argument("--input_map_path", type=str, default=None, help="Path to the input map")
    parser.add_argument("--struct_file", type=str, default=None, help="Path to the structure file")
    parser.add_argument("--dssp_file", type=str, default=None, help="Path to  DSSP MMCIF format file")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to the output directory")
    parser.add_argument("--contour", type=float, default=-1, help="Contour level: -1, automatically determine")
    parser.add_argument("--box_size", type=int, default=64, help="Size of the box")
    parser.add_argument("--stride", type=int, default=32, help="Stride")
    parser.add_argument("--visual_check", type=int, default=0, help="Input box need visualization")
    args = parser.parse_args()
    return args