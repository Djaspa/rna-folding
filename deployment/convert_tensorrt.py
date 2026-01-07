import shutil
import subprocess
import sys

import fire


def convert_onnx_to_tensorrt(
    onnx_path: str,
    output_path: str = "model.engine",
    fp16: bool = False,
    verbose: bool = False,
):
    """
    Convert an ONNX model to a TensorRT engine using trtexec.

    Args:
        onnx_path: Path to the input ONNX file.
        output_path: Path to save the TensorRT engine.
        fp16: Whether to enable FP16 precision.
        verbose: Whether to print verbose output.
    """
    trtexec_path = shutil.which("trtexec")
    if not trtexec_path:
        print(
            "Error: trtexec not found in PATH.\n"
            "Please install TensorRT or add it to your PATH."
        )
        sys.exit(1)

    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
    ]

    if fp16:
        cmd.append("--fp16")

    if verbose:
        cmd.append("--verbose")

    # Input name must match the one in export_onnx.py ("input")
    # min: batch 1, len 10
    # opt: batch 4, len 100
    # max: batch 32, len 500
    # cmd.append("--minShapes=input:1x10")
    # cmd.append("--optShapes=input:4x100")
    # cmd.append("--maxShapes=input:32x500")

    print(f"Running TensorRT conversion for {onnx_path}...")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully converted {onnx_path} to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during TensorRT conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    fire.Fire(convert_onnx_to_tensorrt)
