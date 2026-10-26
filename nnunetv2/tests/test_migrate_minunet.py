import json

import pytest
import torch

from nnunetv2.run.migrate_minunet import (
    assert_equal, inventory, main, preflight, renamed, sha256,
)


def fixture_tree(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "readme.md").write_text("Original project documentation")
    plans = {"plans_name": "MobileUNetPlans", "configurations": {
        "MN": {"architecture": {"network_class_name": "nnunetv2.network_architecture.mobile_unet.MobileUNet"}},
        "4x": {"data_identifier": "MobileUNetPlans_4x"},
    }}
    model = root / "nnUNet_results/Dataset001_Test/Trainer__MobileUNetPlans__MN-4x-M"
    fold = model / "fold_0"
    fold.mkdir(parents=True)
    (model / "plans.json").write_text(json.dumps(plans))
    weights = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    checkpoint = {"network_weights": {"weight": weights},
                  "optimizer_state": {"state": {0: {"momentum": torch.ones(3)}}},
                  "logging": {"loss": [float("nan"), 0.5]}, "current_epoch": 1000,
                  "init_args": {"plans": plans, "configuration": "MN-4x-M"}}
    torch.save(checkpoint, fold / "checkpoint_last.pth")
    (fold / "training_log.txt").write_text("Historical MobileUNetPlans\n")
    empty = root / "nnUNet_results/Dataset001_Test/Trainer__MobileUNetPlans__empty"
    empty.mkdir()
    archive = root / "nnUNet_results/Archive_Test/untouched.pth"
    archive.parent.mkdir()
    archive.write_bytes(b"not even a checkpoint; must never be read")
    source = root / "benchmarks/run/sources/mobile_unet.py"
    source.parent.mkdir(parents=True)
    source.write_text("class MobileUNet: pass\n")
    return root, tmp_path / "backup", model, checkpoint, archive


def test_backup_migration_preserves_state_and_excludes_archives(tmp_path):
    root, backup, model, checkpoint, archive = fixture_tree(tmp_path)
    archive_hash = sha256(archive)
    old_checkpoint = model / "fold_0/checkpoint_last.pth"
    original_hash = sha256(old_checkpoint)
    main(["--root", str(root), "--backup-dir", str(backup), "--apply"])
    saved = backup / "files" / old_checkpoint.relative_to(root)
    assert sha256(saved) == original_hash
    new_model = model.with_name(renamed(model.name))
    restored = torch.load(new_model / "fold_0/checkpoint_last.pth", weights_only=False)
    assert_equal(checkpoint["network_weights"], restored["network_weights"])
    assert_equal(checkpoint["optimizer_state"], restored["optimizer_state"])
    assert restored["init_args"]["plans"]["plans_name"] == "MinuNetPlans"
    assert restored["current_epoch"] == 1000
    assert sha256(archive) == archive_hash
    assert not (backup / "files/nnUNet_results/Archive_Test").exists()
    assert (new_model / "fold_0/training_log.txt").read_text() == "Historical MobileUNetPlans\n"
    assert (root / "benchmarks/run/sources/mobile_unet.py").read_text() == "class MobileUNet: pass\n"
    assert (new_model.parent / "Trainer__MinuNetPlans__empty").is_dir()
    assert (backup / "files" / (model.parent / "Trainer__MobileUNetPlans__empty").relative_to(root)).is_dir()
    assert json.loads((backup / "verification.json").read_text())["checkpoint_files_changed"] == 1
    # A fully completed migration can be resumed without any further mutations.
    result_hash = sha256(new_model / "fold_0/checkpoint_last.pth")
    (root / "readme.md").write_text("Updated documentation after compatibility cleanup")
    main(["--root", str(root), "--backup-dir", str(backup), "--apply", "--resume"])
    assert sha256(new_model / "fold_0/checkpoint_last.pth") == result_hash


def test_collision_fails_before_backup_or_changes(tmp_path):
    root, backup, model, _, _ = fixture_tree(tmp_path)
    model.with_name(renamed(model.name)).mkdir()
    with pytest.raises(RuntimeError, match="collision"):
        preflight(root, backup, inventory(root))
    assert not backup.exists()
    assert model.exists()


def test_existing_backup_destination_is_never_merged(tmp_path):
    root, backup, *_ = fixture_tree(tmp_path)
    backup.mkdir()
    sentinel = backup / "user-file"
    sentinel.write_text("keep")
    with pytest.raises(RuntimeError, match="already exists"):
        main(["--root", str(root), "--backup-dir", str(backup), "--apply"])
    assert sentinel.read_text() == "keep"


def test_corrupted_checkpoint_stops_after_verified_backup(tmp_path):
    root, backup, model, _, _ = fixture_tree(tmp_path)
    checkpoint = model / "fold_0/checkpoint_last.pth"
    checkpoint.write_bytes(b"corrupt checkpoint")
    with pytest.raises(Exception):
        main(["--root", str(root), "--backup-dir", str(backup), "--apply"])
    manifest = json.loads((backup / "manifest.json").read_text())
    assert manifest["backup_complete"] is True
    assert (backup / "files" / checkpoint.relative_to(root)).read_bytes() == b"corrupt checkpoint"
    assert checkpoint.read_bytes() == b"corrupt checkpoint"


def test_resume_recovers_replacement_before_done_journal_append(tmp_path, monkeypatch):
    from nnunetv2.run import migrate_minunet as module

    root, backup, model, _, _ = fixture_tree(tmp_path)
    original_event = module.event

    def interrupt_after_replacement(path, entry):
        if path.name == "migration-journal.jsonl" and entry.get("kind") == "checkpoint" and entry["status"] == "done":
            raise RuntimeError("simulated interruption after atomic replacement")
        original_event(path, entry)

    monkeypatch.setattr(module, "event", interrupt_after_replacement)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        main(["--root", str(root), "--backup-dir", str(backup), "--apply"])
    monkeypatch.setattr(module, "event", original_event)
    main(["--root", str(root), "--backup-dir", str(backup), "--apply", "--resume"])
    report = json.loads((backup / "verification.json").read_text())
    assert report["checkpoint_files_changed"] == 1
    assert model.with_name(renamed(model.name)).is_dir()


def test_resume_rejects_tampered_backup(tmp_path):
    root, backup, *_ = fixture_tree(tmp_path)
    main(["--root", str(root), "--backup-dir", str(backup), "--apply"])
    (backup / "files/readme.md").write_text("tampered")
    with pytest.raises(RuntimeError, match="Existing backup verification failed"):
        main(["--root", str(root), "--backup-dir", str(backup), "--apply", "--resume"])


def test_new_source_files_during_backup_stop_migration(tmp_path, monkeypatch):
    from nnunetv2.run import migrate_minunet as module

    root, backup, model, *_ = fixture_tree(tmp_path)
    original_backup = module.backup_files

    def backup_then_add_file(root, backup, records):
        result = original_backup(root, backup, records)
        (model / "new-result.json").write_text("{}")
        return result

    monkeypatch.setattr(module, "backup_files", backup_then_add_file)
    with pytest.raises(RuntimeError, match="inventory changed"):
        main(["--root", str(root), "--backup-dir", str(backup), "--apply"])
    assert (backup / "manifest.json").is_file()
    assert not (backup / "migration-journal.jsonl").exists()
    assert model.exists()
