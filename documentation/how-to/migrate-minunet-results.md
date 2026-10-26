# Migrate existing research results to MinuNet

The one-time migration utility updates saved plans, result paths, checkpoint
metadata, benchmark artifacts, and local research scripts. It preserves `MN-*`
configuration names, weights, optimizer state, metrics, predictions, and historical
training logs.

Preview the scope without creating files:

```powershell
uv run python -m nnunetv2.run.migrate_minunet --root . --backup-dir "$Env:USERPROFILE/Desktop/nnunet-results-backup"
```

Add `--apply` to back up and migrate. The backup directory must be new. Stop
training and other processes that write the selected artifacts before running it.

The backup includes active dataset results, non-archived benchmark artifacts,
top-level preprocessed metadata, research reports, and local scripts. Original
files are under `files/`, retaining their workspace-relative paths. The manifest
records sizes, modification times, and SHA-256 hashes. `directories.json` records
the original directory structure, including empty directories.

Directories containing `archive` or `backup` in their names are excluded without
being traversed. Raw datasets and ground-truth directories are outside the scope.
Frozen benchmark sources and environment/source-hash records are backed up and
retained unchanged. Existing preprocessed image caches require an explicit scope
extension; the script stops if it finds caches that would need renaming.

Every selected file is copied and hash-verified before migration begins. The
utility rejects destination collisions, checks for changes to source files, writes
replacements atomically, and records each operation in a journal. Every rewritten
checkpoint is reloaded and compared with the expected state, including all tensor
bytes, optimizer state, logger state, and epoch information. The final verification
checks the hashes of all migrated data and all untouched backed-up data files.
The README and compatibility modules are backup-only code files; later code cleanup
can change them independently of the data verification.

After an interrupted run, use the same command with `--apply --resume`. An
unexpected process termination may leave `migration.lock`; check that the process
has exited before removing that lock and resuming. Existing backup files are
verified before reuse. `verification.json` records successful completion.

For recovery, `manifest.json` supplies the original filenames and hashes,
`directories.json` supplies original folders, and `migration-plan.json` records the
renames. The backup remains in the original naming scheme.
