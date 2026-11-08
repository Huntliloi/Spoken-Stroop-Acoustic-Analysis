from pathlib import Path
import shutil

def output_root(package):
    root=Path(package)/'results/recomputed'
    root.mkdir(parents=True,exist_ok=True)
    # Frozen QC metadata is an input to downstream analyses.
    target=root/'02_acoustic_measurement_qc'
    if not target.exists():
        shutil.copytree(Path(package)/'results/02_acoustic_measurement_qc',target)
    return root
