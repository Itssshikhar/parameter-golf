#!/usr/bin/env bash
# safe_launch.sh — verify train_pr1493.py integrity, then exec torchrun.
# Reason this exists: prior session had train_pr1493.py rolled back to a
# pre-stacking version mid-session ("FS turbulence at 06:54"), which silently
# turned PAIRED_HEAD_MUON_ENABLED into a no-op. This wrapper aborts before
# torchrun if HEAD, md5, blob sha, or stacking-symbol count disagree with the
# expected snapshot.
set -euo pipefail

REPO_ROOT="/workspace/parameter-golf"
FILE="$REPO_ROOT/train_pr1493.py"
BACKUP="$REPO_ROOT/train_pr1493.py.bak.74dc702"

EXPECTED_HEAD="74dc7028a06a0f52e2ce23a925ef24404e93ca1b"
EXPECTED_MD5="968e5ab744772b096a8f9b521656019d"
EXPECTED_BLOB="1e4f7b4391f9a82b0ca7f735bbbb0db6eea8e8ad"
EXPECTED_SYMBOLS=6
SYMBOL_REGEX="(paired_head_muon_enabled|fold_iha_mixes|tagged=22|paired_head_zeropower|ns5_3d|stacking)"

cd "$REPO_ROOT"

fail() { echo "[safe_launch] ABORT: $*" >&2; exit 2; }

actual_head="$(git rev-parse HEAD)"
[[ "$actual_head" == "$EXPECTED_HEAD" ]] || fail "HEAD=$actual_head != expected $EXPECTED_HEAD"

[[ -f "$FILE" ]] || fail "missing $FILE"

actual_md5="$(md5sum "$FILE" | awk '{print $1}')"
[[ "$actual_md5" == "$EXPECTED_MD5" ]] || fail "md5=$actual_md5 != expected $EXPECTED_MD5"

actual_blob="$(git hash-object "$FILE")"
[[ "$actual_blob" == "$EXPECTED_BLOB" ]] || fail "blob=$actual_blob != expected $EXPECTED_BLOB"

actual_symbols="$(grep -cE "$SYMBOL_REGEX" "$FILE")"
[[ "$actual_symbols" -ge "$EXPECTED_SYMBOLS" ]] || fail "stacking symbols=$actual_symbols < $EXPECTED_SYMBOLS"

if ! git diff --quiet HEAD -- train_pr1493.py; then
  fail "train_pr1493.py has uncommitted diff vs HEAD"
fi

if [[ ! -f "$BACKUP" ]]; then
  fail "missing backup $BACKUP"
fi
backup_md5="$(md5sum "$BACKUP" | awk '{print $1}')"
[[ "$backup_md5" == "$EXPECTED_MD5" ]] || fail "backup md5=$backup_md5 != expected $EXPECTED_MD5"

ts="$(date -u +%Y%m%dT%H%M%SZ)"
echo "[safe_launch] OK ts=$ts head=$actual_head md5=$actual_md5 blob=$actual_blob symbols=$actual_symbols"
echo "[safe_launch] exec: $*"
exec "$@"
