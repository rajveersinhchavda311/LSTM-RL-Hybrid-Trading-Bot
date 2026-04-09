#!/usr/bin/env python3
"""
Generate a report/manifest for the stock-suggester-ai project.

Outputs:
 - report/manifest.json  (summary of files, sizes, model metadata)
 - report/files.csv      (file path, size bytes, size MB)

Run:
  python scripts/generate_report.py

This script attempts to inspect TFLite and SavedModel artifacts if TensorFlow
is available; otherwise it will still produce a file inventory and copy of
top-level docs.
"""
import os
import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report"
OUT_DIR.mkdir(exist_ok=True)

def sizeof(path: Path):
    try:
        s = path.stat().st_size
    except Exception:
        s = 0
    return s

def scan_files(root: Path):
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip virtual env
        if 'venv' in dirpath.split(os.sep) or '.git' in dirpath.split(os.sep):
            continue
        for fn in filenames:
            fp = Path(dirpath) / fn
            rel = fp.relative_to(root)
            size = sizeof(fp)
            rows.append({'path': str(rel), 'size_bytes': size, 'size_mb': round(size/1024/1024,4)})
    return rows

def write_csv(rows, out_path: Path):
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path','size_bytes','size_mb'])
        writer.writeheader()
        for r in sorted(rows, key=lambda x: x['size_bytes'], reverse=True):
            writer.writerow(r)

def read_text_file(p: Path):
    if not p.exists():
        return None
    try:
        return p.read_text(encoding='utf-8')
    except Exception:
        try:
            return p.read_text(encoding='latin-1')
        except Exception:
            return None

def inspect_tflite(path: Path):
    info = {'path': str(path), 'exists': path.exists(), 'size_bytes': sizeof(path)}
    if not path.exists():
        return info
    try:
        # Try to import TensorFlow Lite Interpreter
        try:
            from tensorflow.lite import Interpreter
        except Exception:
            # fallback to tflite_runtime if available
            from tflite_runtime.interpreter import Interpreter

        interp = Interpreter(model_path=str(path))
        interp.allocate_tensors()
        info['inputs'] = interp.get_input_details()
        info['outputs'] = interp.get_output_details()
    except Exception as e:
        info['error'] = str(e)
    return info

def inspect_saved_model(dirpath: Path):
    info = {'path': str(dirpath), 'exists': dirpath.exists(), 'files': []}
    if not dirpath.exists():
        return info
    for p in dirpath.rglob('*'):
        if p.is_file():
            info['files'].append({'path': str(p.relative_to(dirpath)), 'size_bytes': sizeof(p)})
    # Try to load signatures if TF available
    try:
        import tensorflow as tf
        loaded = tf.saved_model.load(str(dirpath))
        try:
            info['signatures'] = list(loaded.signatures.keys())
        except Exception:
            info['signatures'] = None
    except Exception as e:
        info['tf_error'] = str(e)
    return info

def main():
    print(f"Scanning project at: {ROOT}")
    files = scan_files(ROOT)
    write_csv(files, OUT_DIR / 'files.csv')

    manifest = {
        'project_root': str(ROOT),
        'num_files': len(files),
        'files_csv': str((OUT_DIR / 'files.csv').relative_to(ROOT)),
        'requirements': None,
        'run_instructions': None,
        'readme': None,
        'tflite': None,
        'saved_model': None
    }

    # Read textual artifacts
    manifest['requirements'] = read_text_file(ROOT / 'requirements.txt')
    manifest['run_instructions'] = read_text_file(ROOT / 'RUN_INSTRUCTIONS.md')
    manifest['readme'] = read_text_file(ROOT / 'README.md')

    # Inspect TFLite if present
    tflite_path = ROOT / 'optimized_lstm_model.tflite'
    if tflite_path.exists():
        manifest['tflite'] = inspect_tflite(tflite_path)

    # Inspect SavedModel dir if present
    saved_model_dir = ROOT / 'lstm_saved_model'
    if saved_model_dir.exists():
        manifest['saved_model'] = inspect_saved_model(saved_model_dir)

    # Save manifest
    with (OUT_DIR / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Report generated in: {OUT_DIR}")
    print(f" - manifest.json: {OUT_DIR / 'manifest.json'}")
    print(f" - files.csv: {OUT_DIR / 'files.csv'}")

if __name__ == '__main__':
    main()
