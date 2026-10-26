"""One-time, backed-up migration of local research artifacts to MinuNet names.

Dry-run by default. Archives/backups are never traversed. No data is changed until
every selected source file has a verified backup. Original checkpoints, logs,
predictions, and plans remain recoverable from the backup directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import time


SCHEMA = 1
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".py", ".ps1", ".sh", ".yaml", ".yml"}
PROVENANCE_KEYS = {"sources", "source_hashes", "source_files", "git_diff", "git_status", "git_commit", "git_head"}
REPLACEMENTS = (
    ("mobile_unet_planner", "minunet_planner"),
    ("mobile_unet", "minunet"),
    ("MobileUNet", "MinuNet"),
    ("Mobile U-Net", "MinuNet"),
    ("MobilePlans", "MinuNetPlans"),
    ("mobile_plans", "minunet_plans"),
    ("mobile-plans", "minunet-plans"),
)


def renamed(value: str) -> str:
    for old, new in REPLACEMENTS:
        value = value.replace(old, new)
    value = re.sub(r"\bMOBILE\b", "MINUNET", value)
    return re.sub(r"\b[Mm]obile\b", "MinuNet", value)


def excluded(name: str) -> bool:
    return "archive" in name.lower() or "backup" in name.lower() or name == "__pycache__"


def frozen(relative: str) -> bool:
    path = Path(relative)
    return (
        any(p in {"sources", "workspace"} for p in path.parts)
        or "environment" in path.name.lower()
        or path.name == "source-hashes.json"
    )


def resolve_inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Path escapes the selected root: {relative}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def event(path: Path, value) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    events = []
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except ValueError:
            if index != len(lines) - 1:
                raise RuntimeError(f"Corrupt journal: {path}")
            # A power loss can truncate the final append. Preserve the valid prefix.
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for entry in events:
                    stream.write(json.dumps(entry) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    return events


def inventory(root: Path) -> list[dict]:
    selected = []
    for base_name in ("nnUNet_results", "benchmarks", "scripts", "documentation/research"):
        base = root / base_name
        if not base.exists():
            continue
        for current, directories, filenames in os.walk(base, followlinks=False):
            directories[:] = sorted(d for d in directories if not excluded(d))
            if Path(current).is_symlink():
                raise RuntimeError(f"Refusing linked source directory: {current}")
            for name in sorted(filenames):
                path = Path(current) / name
                if path.is_symlink():
                    raise RuntimeError(f"Refusing linked source file: {path}")
                selected.append(path)
    # The data caches are absent in this checkout; only top-level metadata is needed.
    preprocessed = root / "nnUNet_preprocessed"
    if preprocessed.exists():
        for dataset in sorted(preprocessed.glob("Dataset*")):
            selected.extend(p for p in sorted(dataset.glob("*.json")) if p.is_file())
            old_caches = [p for p in dataset.iterdir() if p.is_dir() and renamed(p.name) != p.name]
            if old_caches:
                raise RuntimeError(f"Preprocessed image caches need explicit migration scope: {old_caches}")
    selected.extend(root / name for name in (
        "readme.md", "nnunetv2/network_architecture/mobile_unet.py",
        "nnunetv2/experiment_planning/experiment_planners/mobile_unet_planner.py",
    ) if (root / name).exists())
    records = []
    for path in sorted(set(selected)):
        relative = path.relative_to(root).as_posix()
        resolve_inside(root, relative)
        stat = path.stat()
        records.append({"path": relative, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return records


def final_name(relative: str) -> str:
    if relative.startswith("nnunetv2/") or relative == "readme.md":
        return relative
    # Frozen source snapshots retain their filenames and contents. Their ancestors
    # can still move when a surrounding benchmark family directory is renamed.
    parts = Path(relative).parts
    result = []
    inside_snapshot = False
    for part in parts:
        if part in {"sources", "workspace"}:
            inside_snapshot = True
        result.append(part if inside_snapshot else renamed(part))
    return Path(*result).as_posix()


def preflight(root: Path, backup: Path, records: list[dict]) -> None:
    if backup.is_relative_to(root) or root.is_relative_to(backup):
        raise RuntimeError("Backup and source roots must be separate directory trees")
    destinations = {}
    for record in records:
        source = record["path"]
        target = final_name(source)
        folded = target.casefold()
        if folded in destinations and destinations[folded] != source:
            raise RuntimeError(f"Destination collision: {source}, {destinations[folded]}")
        destinations[folded] = source
        if target != source and resolve_inside(root, target).exists():
            raise RuntimeError(f"Destination already exists: {target}")
    # Detect directory collisions even for empty result directories.
    for base_name in ("nnUNet_results", "benchmarks"):
        for current, directories, _ in os.walk(root / base_name):
            directories[:] = [d for d in directories if not excluded(d)]
            for name in directories:
                if renamed(name) != name and (Path(current) / renamed(name)).exists():
                    raise RuntimeError(f"Directory collision at {Path(current) / name}")
    total = sum(record["size"] for record in records)
    if shutil.disk_usage(backup.parent).free < total + 2 * 1024**3:
        raise RuntimeError("Insufficient backup disk space")
    largest = max((record["size"] for record in records), default=0)
    if shutil.disk_usage(root).free < largest + 1024**3:
        raise RuntimeError("Insufficient source disk space for atomic replacement")


def backup_files(root: Path, backup: Path, records: list[dict]) -> list[dict]:
    directories = []
    for name in ("nnUNet_results", "benchmarks", "scripts", "documentation/research"):
        for current, names, _ in os.walk(root / name):
            names[:] = [n for n in names if not excluded(n)]
            relative = Path(current).relative_to(root).as_posix()
            resolve_inside(backup / "files", relative).mkdir(parents=True, exist_ok=True)
            directories.append(relative)
    atomic_json(backup / "directories.json", directories)
    journal = backup / "backup-journal.jsonl"
    completed = {entry["path"]: entry for entry in read_events(journal)}
    last_report = 0
    byte_count = 0
    for index, record in enumerate(records, 1):
        source = resolve_inside(root, record["path"])
        target = resolve_inside(backup / "files", record["path"])
        stat = source.stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise RuntimeError(f"Source changed since inventory: {source}")
        previous = completed.get(record["path"])
        if previous:
            digest = sha256(target)
            if digest != previous["sha256"] or sha256(source) != digest:
                raise RuntimeError(f"Resume backup hash mismatch: {source}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".copying")
            digest_state = hashlib.sha256()
            with source.open("rb") as src, temporary.open("wb") as dst:
                for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                    digest_state.update(chunk)
                    dst.write(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            digest = digest_state.hexdigest()
            after_stat = source.stat()
            if (after_stat.st_size, after_stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
                raise RuntimeError(f"Source changed during backup: {source}")
            if sha256(temporary) != digest:
                raise RuntimeError(f"Backup verification failed: {source}")
            shutil.copystat(source, temporary)
            os.replace(temporary, target)
            event(journal, {**record, "sha256": digest})
        record["sha256"] = digest
        byte_count += record["size"]
        if time.monotonic() - last_report > 10 or index == len(records):
            print(f"Backup verified: {index}/{len(records)} files, {byte_count / 1024**3:.2f} GiB", flush=True)
            last_report = time.monotonic()
    return records


def transform(obj):
    """Preserve tensor/OrderedDict objects, including state_dict metadata."""
    if isinstance(obj, dict):
        for key in list(obj):
            if key in PROVENANCE_KEYS:
                continue
            value = transform(obj[key])
            new_key = renamed(key) if isinstance(key, str) else key
            if new_key != key:
                if new_key in obj:
                    raise RuntimeError(f"Dictionary key collision: {key}")
                del obj[key]
            obj[new_key] = value
        return obj
    if isinstance(obj, list):
        obj[:] = [transform(value) for value in obj]
        return obj
    if isinstance(obj, tuple):
        return tuple(transform(value) for value in obj)
    return renamed(obj) if isinstance(obj, str) else obj


def text_replacement(path: Path) -> bytes | None:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise RuntimeError(f"Non-UTF-8 operational text needs inspection: {path}")
    if renamed(text) == text:
        return None
    if path.suffix == ".json":
        original = json.loads(text.lstrip("\ufeff"))
        changed = transform(original)
        # Preserve content and numeric values; skip source provenance fields.
        output = (json.dumps(changed, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    elif path.suffix == ".jsonl":
        output = ("\n".join(json.dumps(transform(json.loads(line)), ensure_ascii=False) if line.strip() else ""
                            for line in text.splitlines()) + "\n").encode("utf-8")
    else:
        output = renamed(text).encode("utf-8")
    return output if output != data else None


def make_plan(root: Path, records: list[dict]) -> dict:
    changes = []
    directories = set()
    for record in records:
        relative = record["path"]
        path = resolve_inside(root, relative)
        # Source compatibility code and README are backed up for subsequent code
        # cleanup, not rewritten as data by this migration.
        if relative.startswith(("nnunetv2/",)) or relative == "readme.md":
            continue
        if not frozen(relative):
            if path.suffix == ".pth":
                changes.append({"path": relative, "kind": "checkpoint", "before": record["sha256"]})
            elif path.suffix in TEXT_SUFFIXES and text_replacement(path) is not None:
                changes.append({"path": relative, "kind": "text", "before": record["sha256"]})
        for parent in path.parents:
            if parent == root:
                break
            rel = parent.relative_to(root).as_posix()
            if not frozen(rel) and renamed(parent.name) != parent.name:
                directories.add(rel)
        if not frozen(relative) and renamed(path.name) != path.name:
            changes.append({"path": relative, "kind": "rename", "target": path.with_name(renamed(path.name)).relative_to(root).as_posix()})
    # Empty result/benchmark directories also belong to the migration.
    for base_name in ("nnUNet_results", "benchmarks"):
        for current, names, _ in os.walk(root / base_name):
            names[:] = [name for name in names if not excluded(name)]
            for name in names:
                relative = (Path(current) / name).relative_to(root).as_posix()
                if not frozen(relative) and renamed(name) != name:
                    directories.add(relative)
    for relative in sorted(directories, key=lambda p: (-len(Path(p).parts), p)):
        path = Path(relative)
        changes.append({"path": relative, "kind": "rename", "target": path.with_name(renamed(path.name)).as_posix()})
    return {"schema": SCHEMA, "operations": [{"id": index, **op} for index, op in enumerate(changes)]}


def assert_equal(expected, actual, location="checkpoint") -> None:
    import numpy as np
    import torch

    if type(expected) is not type(actual):
        raise RuntimeError(f"Type changed at {location}")
    if isinstance(expected, torch.Tensor):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise RuntimeError(f"Tensor metadata changed at {location}")
        if not torch.equal(expected.detach().contiguous().reshape(-1).view(torch.uint8),
                           actual.detach().contiguous().reshape(-1).view(torch.uint8)):
            raise RuntimeError(f"Tensor bytes changed at {location}")
    elif isinstance(expected, dict):
        if list(expected) != list(actual):
            raise RuntimeError(f"Dictionary keys changed at {location}")
        for key in expected:
            assert_equal(expected[key], actual[key], f"{location}.{key}")
        if hasattr(expected, "_metadata"):
            assert_equal(expected._metadata, actual._metadata, f"{location}._metadata")
    elif isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise RuntimeError(f"Sequence length changed at {location}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_equal(left, right, f"{location}[{index}]")
    elif isinstance(expected, (np.ndarray, np.generic)):
        if expected.dtype != actual.dtype or expected.tobytes() != actual.tobytes():
            raise RuntimeError(f"NumPy data changed at {location}")
    elif isinstance(expected, float):
        if struct.pack("d", expected) != struct.pack("d", actual):
            raise RuntimeError(f"Float bits changed at {location}")
    elif expected != actual:
        raise RuntimeError(f"Value changed at {location}")


def load_checkpoint(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as error:
        if "mmap" not in str(error):
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def write_checkpoint(source: Path, temporary: Path) -> bool:
    import torch

    checkpoint = load_checkpoint(source)
    # Five baseline checkpoints are included in the backup but need no edits.
    plans = checkpoint.get("init_args", {}).get("plans", {})
    before = json.dumps(plans, sort_keys=True)
    transform(plans)
    if json.dumps(plans, sort_keys=True) == before:
        return False
    torch.save(checkpoint, temporary)
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    restored = load_checkpoint(temporary)
    assert_equal(checkpoint, restored)
    return True


def migrate(root: Path, backup: Path, plan: dict) -> dict:
    import torch

    torch.set_num_threads(2)
    journal = backup / "migration-journal.jsonl"
    history = {}
    for entry in read_events(journal):
        history[entry["id"]] = entry
    changed = skipped = 0
    last_report = 0
    for index, operation in enumerate(plan["operations"], 1):
        op_id = operation["id"]
        previous = history.get(op_id, {})
        if previous.get("status") == "done":
            continue
        source = resolve_inside(root, operation["path"])
        if operation["kind"] == "rename":
            target = resolve_inside(root, operation["target"])
            if source.exists() and not target.exists():
                source.rename(target)
            elif source.exists() or not target.exists():
                raise RuntimeError(f"Ambiguous rename state: {source} -> {target}")
            entry = {"id": op_id, "status": "done", "kind": "rename"}
        else:
            current_hash = sha256(source)
            if previous.get("status") == "prepared" and current_hash == previous["after"]:
                entry = {**previous, "status": "done"}
            else:
                if current_hash != operation["before"]:
                    raise RuntimeError(f"Unexpected source changes: {source}")
                temporary = source.with_name(source.name + ".minunet-migrating")
                if operation["kind"] == "checkpoint":
                    edited = write_checkpoint(source, temporary)
                    gc.collect()  # Release Windows memory maps before replacing files.
                else:
                    payload = text_replacement(source)
                    edited = payload is not None
                    if edited:
                        with temporary.open("wb") as stream:
                            stream.write(payload)
                            stream.flush()
                            os.fsync(stream.fileno())
                if edited:
                    new_hash = sha256(temporary)
                    event(journal, {"id": op_id, "status": "prepared", "after": new_hash,
                                    "kind": operation["kind"], "changed": True})
                    os.replace(temporary, source)
                    changed += 1
                else:
                    new_hash = current_hash
                    skipped += 1
                entry = {"id": op_id, "status": "done", "after": new_hash, "kind": operation["kind"], "changed": edited}
        event(journal, entry)
        history[op_id] = entry
        if time.monotonic() - last_report > 10 or index == len(plan["operations"]):
            print(f"Migration {index}/{len(plan['operations'])}: {operation['kind']} {operation['path']}", flush=True)
            last_report = time.monotonic()
    return {"changed_this_run": changed, "unchanged_checkpoints_this_run": skipped}


def verify(root: Path, backup: Path, manifest: dict, plan: dict) -> dict:
    done = {entry["id"]: entry for entry in read_events(backup / "migration-journal.jsonl") if entry["status"] == "done"}
    hashes = {op["path"]: done[op["id"]]["after"] for op in plan["operations"] if op["kind"] != "rename"}
    last_report = 0
    verified = 0
    for index, record in enumerate(manifest["files"], 1):
        if record["path"].startswith("nnunetv2/") or record["path"] == "readme.md":
            # Code is backed up for the subsequent compatibility-code cleanup,
            # and can intentionally change after the data migration completes.
            continue
        target = resolve_inside(root, final_name(record["path"]))
        expected = hashes.get(record["path"], record["sha256"])
        if not target.is_file() or sha256(target) != expected:
            raise RuntimeError(f"Final verification failed: {target}")
        verified += 1
        if time.monotonic() - last_report > 10 or index == len(manifest["files"]):
            print(f"Final hashes verified: {index}/{len(manifest['files'])} files", flush=True)
            last_report = time.monotonic()
    report = {"status": "complete", "files_verified": verified,
              "checkpoint_files_changed": sum(e.get("kind") == "checkpoint" and e.get("changed", False) for e in done.values()),
              "text_files_changed": sum(e.get("kind") == "text" for e in done.values()),
              "paths_renamed": sum(e.get("kind") == "rename" for e in done.values()),
              "tensor_verification": "Every rewritten checkpoint was reloaded and compared byte-for-byte, including optimizer and logging state"}
    atomic_json(backup / "verification.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Back up, verify, then migrate. Default is read-only inventory.")
    parser.add_argument("--resume", action="store_true", help="Resume only this script's existing backup/migration journal.")
    args = parser.parse_args(argv)
    root, backup = args.root.resolve(), args.backup_dir.resolve()
    if not args.apply:
        records = inventory(root)
        preflight(root, backup, records)
        print(json.dumps({"files": len(records), "GiB": sum(r['size'] for r in records) / 1024**3,
                          "backup": str(backup), "excluded": "All archive/backup directory trees; historical logs and source snapshots are not rewritten"}, indent=2))
        return
    if backup.exists() and not args.resume:
        raise RuntimeError(f"Backup destination already exists; refusing to merge: {backup}")
    backup.mkdir(parents=True, exist_ok=True)
    lock = backup / "migration.lock"
    # Exclusive lock also rejects a concurrent --resume. An interrupted process
    # leaves the lock for explicit inspection/removal before resuming.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        inventory_file = backup / "inventory.json"
        if inventory_file.exists():
            inv = json.loads(inventory_file.read_text(encoding="utf-8"))
            if inv["schema"] != SCHEMA or inv["root"] != str(root):
                raise RuntimeError("Backup belongs to a different migration")
            records = inv["files"]
        else:
            if args.resume:
                raise RuntimeError("No matching inventory exists to resume")
            records = inventory(root)
            preflight(root, backup, records)
            atomic_json(inventory_file, {"schema": SCHEMA, "root": str(root), "files": records})
        manifest_file = backup / "manifest.json"
        if not manifest_file.exists():
            records = backup_files(root, backup, records)
            atomic_json(manifest_file, {"schema": SCHEMA, "root": str(root), "files": records, "backup_complete": True})
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if args.resume:
            for record in manifest["files"]:
                saved = resolve_inside(backup / "files", record["path"])
                if not saved.is_file() or sha256(saved) != record["sha256"]:
                    raise RuntimeError(f"Existing backup verification failed: {saved}")
        plan_file = backup / "migration-plan.json"
        if not plan_file.exists():
            # A complete backup is the mandatory gate before any source mutation.
            current = {r["path"]: (r["size"], r["mtime_ns"]) for r in inventory(root)}
            expected = {r["path"]: (r["size"], r["mtime_ns"]) for r in manifest["files"]}
            if current != expected:
                raise RuntimeError("Source inventory changed during backup; refusing to migrate")
            preflight(root, backup, manifest["files"])
            plan = make_plan(root, manifest["files"])
            atomic_json(plan_file, plan)
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        print(f"Verified backup complete. Applying {len(plan['operations'])} operations.", flush=True)
        migrate(root, backup, plan)
        print(json.dumps(verify(root, backup, manifest, plan), indent=2), flush=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
