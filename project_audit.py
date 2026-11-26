from pathlib import Path
from datetime import datetime
import ast
import csv
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "PROJECT_AUDIT_REPORT.txt"

IGNORE_DIRS = {
    ".venv", ".git", "__pycache__", "node_modules",
    ".ipynb_checkpoints", "catboost_info"
}

IMPORTANT_EXTENSIONS = {
    ".py", ".ipynb", ".csv", ".xlsx", ".xls",
    ".json", ".yaml", ".yml", ".txt", ".md",
    ".joblib", ".pkl", ".sav", ".parquet",
    ".zip", ".pdf", ".png", ".jpg"
}

def human_size(size):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024

def file_hash(path):
    digest = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return "Unavailable"

def inspect_python(path):
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)

        imports = []
        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(x.name for x in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)

        return {
            "imports": sorted(set(imports)),
            "functions": functions,
            "classes": classes,
            "lines": len(source.splitlines()),
        }
    except Exception as e:
        return {"inspection_error": str(e)}

def inspect_notebook(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cells = data.get("cells", [])
        code_cells = [
            "".join(c.get("source", []))
            for c in cells if c.get("cell_type") == "code"
        ]
        markdown_cells = [
            "".join(c.get("source", []))
            for c in cells if c.get("cell_type") == "markdown"
        ]
        return {
            "total_cells": len(cells),
            "code_cells": len(code_cells),
            "markdown_cells": len(markdown_cells),
            "code_preview": "\n".join(code_cells)[:2500],
            "markdown_preview": "\n".join(markdown_cells)[:1200],
        }
    except Exception as e:
        return {"inspection_error": str(e)}

def inspect_csv(path):
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace",
                       newline="") as f:
            sample = f.read(8192)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample)
            reader = csv.reader(f, dialect)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
        return {
            "rows_estimated": rows,
            "columns": len(header),
            "column_names": header[:100],
        }
    except Exception as e:
        return {"inspection_note": str(e)}

def classify(path):
    name = path.name.lower()
    ext = path.suffix.lower()

    if ext == ".py":
        return "Python source"
    if ext == ".ipynb":
        return "Jupyter notebook"
    if ext in {".csv", ".xlsx", ".xls", ".parquet"}:
        return "Dataset or tabular output"
    if ext in {".joblib", ".pkl", ".sav"}:
        return "Serialized model/artifact"
    if ext == ".zip":
        return "Archive"
    if ext in {".md", ".txt", ".pdf"}:
        return "Documentation/report"
    if ext in {".png", ".jpg"}:
        return "Image/plot"
    if "requirements" in name:
        return "Dependency configuration"
    return "Other file"

files = []
folders = []
errors = []

for current, dirs, filenames in os.walk(ROOT):
    current_path = Path(current)
    dirs[:] = sorted(
        d for d in dirs
        if d not in IGNORE_DIRS
    )

    if current_path != ROOT:
        folders.append(current_path.relative_to(ROOT).as_posix())

    for filename in sorted(filenames):
        path = current_path / filename

        if path == Path(__file__) or path == OUTPUT:
            continue

        try:
            stat = path.stat()
            rel = path.relative_to(ROOT).as_posix()

            item = {
                "path": rel,
                "type": classify(path),
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "size": human_size(stat.st_size),
                "modified": datetime.fromtimestamp(
                    stat.st_mtime
                ).isoformat(timespec="seconds"),
            }

            if path.suffix.lower() == ".py":
                item["code_analysis"] = inspect_python(path)
            elif path.suffix.lower() == ".ipynb":
                item["notebook_analysis"] = inspect_notebook(path)
            elif path.suffix.lower() == ".csv" and stat.st_size < 100 * 1024 * 1024:
                item["csv_analysis"] = inspect_csv(path)

            files.append(item)

        except Exception as e:
            errors.append(f"{path}: {e}")

total_size = sum(x["size_bytes"] for x in files)

with OUTPUT.open("w", encoding="utf-8") as report:
    report.write("HYDROGEN DIGITAL TWIN - PROJECT AUDIT\n")
    report.write("=" * 75 + "\n")
    report.write(f"Project root: {ROOT}\n")
    report.write(f"Audit time: {datetime.now().isoformat()}\n")
    report.write(f"Files inspected: {len(files)}\n")
    report.write(f"Folders inspected: {len(folders)}\n")
    report.write(f"Total inventoried size: {human_size(total_size)}\n")
    report.write("Excluded generated/environment folders: ")
    report.write(", ".join(sorted(IGNORE_DIRS)) + "\n\n")

    report.write("1. COMPLETE FOLDER STRUCTURE\n")
    report.write("-" * 75 + "\n")
    report.write(".\n")
    for folder in folders:
        depth = folder.count("/")
        report.write("  " * (depth + 1) + folder.split("/")[-1] + "/\n")

    report.write("\n2. FILE INVENTORY\n")
    report.write("-" * 75 + "\n")

    for item in sorted(files, key=lambda x: x["path"].lower()):
        report.write(f"\n[{item['type']}] {item['path']}\n")
        report.write(
            f"Size: {item['size']} | Modified: {item['modified']}\n"
        )

        if "code_analysis" in item:
            info = item["code_analysis"]
            report.write(f"Lines: {info.get('lines', '?')}\n")
            report.write(f"Imports: {', '.join(info.get('imports', []))}\n")
            report.write(f"Functions: {', '.join(info.get('functions', []))}\n")
            report.write(f"Classes: {', '.join(info.get('classes', []))}\n")
            if "inspection_error" in info:
                report.write(f"Inspection error: {info['inspection_error']}\n")

        if "notebook_analysis" in item:
            info = item["notebook_analysis"]
            report.write(
                f"Cells: {info.get('total_cells', '?')} "
                f"(code: {info.get('code_cells', '?')}, "
                f"markdown: {info.get('markdown_cells', '?')})\n"
            )
            report.write("Markdown preview:\n")
            report.write(info.get("markdown_preview", "") + "\n")
            report.write("Code preview:\n")
            report.write(info.get("code_preview", "") + "\n")

        if "csv_analysis" in item:
            info = item["csv_analysis"]
            report.write(
                f"Rows (estimated): {info.get('rows_estimated', '?')} | "
                f"Columns: {info.get('columns', '?')}\n"
            )
            report.write(
                f"Column names: {', '.join(info.get('column_names', []))}\n"
            )

    report.write("\n3. LARGE FILES (OVER 50 MB)\n")
    report.write("-" * 75 + "\n")
    for item in sorted(
        [x for x in files if x["size_bytes"] > 50 * 1024 * 1024],
        key=lambda x: x["size_bytes"],
        reverse=True
    ):
        report.write(f"{item['size']:>12}  {item['path']}\n")

    report.write("\n4. EMPTY FILES\n")
    report.write("-" * 75 + "\n")
    for item in files:
        if item["size_bytes"] == 0:
            report.write(item["path"] + "\n")

    report.write("\n5. ACCESS ERRORS\n")
    report.write("-" * 75 + "\n")
    report.write("\n".join(errors) if errors else "None")

print("Audit completed.")
print(f"Files inspected: {len(files)}")
print(f"Folders inspected: {len(folders)}")
print(f"Total inventoried size: {human_size(total_size)}")
print(f"Report saved to: {OUTPUT}")
