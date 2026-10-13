#!/usr/bin/env python3
"""Extract hash-pinned Linux x86_64 native runtime libraries; no Python install needed."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

PACKAGES = (
    (
        "onnxruntime-1.31.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        "https://files.pythonhosted.org/packages/6c/44/1e9e762b95b7da0a8424913a1ed7c38cdaf88624a3c41ddba24ebac88bc9/onnxruntime-1.31.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        "5e129d6c56abd53e659cb70f00a108d6824086470ff99c2e47a82e5786563db3",
    ),
    (
        "onnxruntime_ep_webgpu-0.4.0-py3-none-manylinux_2_28_x86_64.whl",
        "https://files.pythonhosted.org/packages/c1/96/2a18a45079250afcd825687aef2895de266d5abc1cff9f52d2dede7598f2/onnxruntime_ep_webgpu-0.4.0-py3-none-manylinux_2_28_x86_64.whl",
        "5d8c961eda91a88961cad1fb7621f389f006b18497d6ea9ecf96f729f066cbfb",
    ),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for filename, url, checksum in PACKAGES:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise RuntimeError(f"Checksum mismatch for {filename}")
        files = []
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in archive.namelist():
                basename = Path(name).name
                if basename.startswith("libonnxruntime") and ".so" in basename:
                    destination = output / basename
                elif basename in {"LICENSE", "LICENSE.txt", "ThirdPartyNotices.txt", "NOTICE"}:
                    destination = output / "notices" / filename.removesuffix(".whl") / basename
                else:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))
                files.append(str(destination.relative_to(output)))
        manifest.append({"package": filename, "sha256": checksum, "files": files})
    for name in (
        "libonnxruntime.so.1.31.0",
        "libonnxruntime_providers_shared.so",
        "libonnxruntime_providers_webgpu.so",
    ):
        if not (output / name).is_file():
            raise RuntimeError(f"Required runtime library absent: {name}")
    alias = output / "libonnxruntime.so"
    if alias.is_symlink():
        alias.unlink()
    if alias.exists():
        raise RuntimeError(f"Refusing to overwrite existing {alias}")
    alias.symlink_to("libonnxruntime.so.1.31.0")
    (output / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Verified ONNX Runtime 1.31.0 and native WebGPU 0.4.0 in {output}")


if __name__ == "__main__":
    main()
