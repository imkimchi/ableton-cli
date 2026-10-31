"""Collect a set's samples into a new project folder, like Live's Collect All and Save.

Live's Remote Script API cannot save, open or collect a set, so this works on the
file. Sample references (``SampleRef/FileRef``) are resolved the way Live stores
them:

- ``RelativePathType`` 0: the absolute ``Path``.
- ``RelativePathType`` 1: ``RelativePath`` against the set's folder.
- ``RelativePathType`` 3: ``RelativePath`` against the project folder (the nearest
  ancestor holding ``Ableton Project Info``).

Every other type points into a library (Core Library, User Library, Packs) that
Live resolves itself; those references are left unchanged and counted as retained,
as Live's own collect does by default.

Each resolved file is copied once into the new project: project samples keep
their relative path, everything else goes to ``Samples/Imported``. Name clashes
between different files get a content-hash suffix. Collected references are
rewritten to project-relative (type 3) with ``Path`` pointing at the copy.
``OriginalFileRef/FileRef`` elements that point at a collected file (the media a
warped or edited sample came from) follow the same rewrite.

A missing sample aborts the collect before anything is written: the result would
otherwise look complete while still depending on the missing file.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ...errors import AppError, ErrorCode, ExitCode
from ..document import LiveSetDocument
from ..findings import listed_details
from ..media import PROJECT_INFO_DIR, PROJECT_RELATIVE, find_project_root
from ..save import save_live_set

ABSOLUTE = "0"
SET_RELATIVE = "1"
IMPORTED_DIR = PurePosixPath("Samples/Imported")
ANALYSIS_SUFFIX = ".asd"
_FIELDS = ("RelativePathType", "RelativePath", "Path")
_CHUNK = 1024 * 1024


@dataclass(frozen=True, slots=True)
class CollectPlan:
    #: Source file -> destination relative to the new project folder.
    destinations: dict[Path, PurePosixPath]
    #: FileRef elements to rewrite, with the source file each one points at.
    rewrites: list[tuple[ET.Element, Path]]
    retained_count: int


def _invalid(message: str, hint: str, **details: Any) -> AppError:
    return AppError(
        error_code=ErrorCode.INVALID_ARGUMENT,
        message=message,
        hint=hint,
        exit_code=ExitCode.INVALID_ARGUMENT,
        details=details,
    )


def _value(file_ref: ET.Element, tag: str) -> str:
    element = file_ref.find(tag)
    return "" if element is None else element.get("Value", "")


def _is_safe_relative(relative: str) -> bool:
    path = PurePosixPath(relative)
    return bool(relative) and not path.is_absolute() and ".." not in path.parts


def _resolve(file_ref: ET.Element, set_dir: Path, project_root: Path | None) -> Path | None:
    kind, relative, stored = (_value(file_ref, tag) for tag in _FIELDS)
    if kind == PROJECT_RELATIVE and project_root is not None and _is_safe_relative(relative):
        return project_root / relative
    if kind == SET_RELATIVE and relative:
        return set_dir / relative
    if kind == ABSOLUTE and stored and Path(stored).is_absolute():
        return Path(stored)
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _name_key(destination: PurePosixPath) -> str:
    """macOS and Windows file systems compare names case- and normalization-insensitively."""
    return unicodedata.normalize("NFC", destination.as_posix()).casefold()


def _destination(file_ref: ET.Element, source: Path) -> PurePosixPath:
    relative = _value(file_ref, "RelativePath")
    if _value(file_ref, "RelativePathType") == PROJECT_RELATIVE and _is_safe_relative(relative):
        return PurePosixPath(relative)
    return IMPORTED_DIR / source.name


def _assign(destinations: dict[Path, PurePosixPath], source: Path, wanted: PurePosixPath) -> None:
    if source in destinations:
        return
    taken = {_name_key(destination) for destination in destinations.values()}
    if _name_key(wanted) in taken:
        stem, suffix = PurePosixPath(wanted.name).stem, wanted.suffix
        wanted = wanted.with_name(f"{stem}-{sha256(source)[:12]}{suffix}")
    destinations[source] = wanted


def plan_collect(document: LiveSetDocument) -> CollectPlan:
    set_dir = document.path.parent.resolve()
    project_root = find_project_root(document.path)
    destinations: dict[Path, PurePosixPath] = {}
    rewrites: list[tuple[ET.Element, Path]] = []
    missing: list[dict[str, str]] = []
    retained = 0
    for file_ref in document.root.iterfind(".//SampleRef/FileRef"):
        if not (_value(file_ref, "RelativePath") or _value(file_ref, "Path")):
            continue
        source = _resolve(file_ref, set_dir, project_root)
        if source is None and _value(file_ref, "RelativePathType") not in {
            ABSOLUTE,
            SET_RELATIVE,
            PROJECT_RELATIVE,
        }:
            retained += 1
            continue
        if source is None or not source.is_file():
            relative = _value(file_ref, "RelativePath") or _value(file_ref, "Path")
            missing.append(
                {
                    "relative_path_type": _value(file_ref, "RelativePathType"),
                    "file_name": PurePosixPath(relative.replace("\\", "/")).name,
                }
            )
            continue
        source = source.resolve()
        _assign(destinations, source, _destination(file_ref, source))
        rewrites.append((file_ref, source))
    if missing:
        raise _invalid(
            f"{document.path.name}: {len(missing)} sample reference(s) cannot be found",
            "Relink the missing samples in Live (File > Manage Files) and save, then collect.",
            path=str(document.path),
            **listed_details("files", missing),
        )
    for file_ref in document.root.iterfind(".//OriginalFileRef/FileRef"):
        source = _resolve(file_ref, set_dir, project_root)
        if source is not None and source.is_file() and source.resolve() in destinations:
            rewrites.append((file_ref, source.resolve()))
    return CollectPlan(destinations=destinations, rewrites=rewrites, retained_count=retained)


def _copy_verified(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256(source)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, _CHUNK)
    if sha256(target) != expected:
        raise AppError(
            error_code=ErrorCode.INTERNAL_ERROR,
            message=f"Copied sample does not match its source: {source.name}",
            hint="Check the destination disk, then collect again into a new folder.",
            exit_code=ExitCode.EXECUTION_FAILED,
            details={"file_name": source.name},
        )
    return target.stat().st_size


def _copy_samples(plan: CollectPlan, staging: Path) -> tuple[int, int]:
    """Copy every sample plus its ``.asd`` analysis file; return (files, bytes)."""
    files = size = 0
    for source, destination in plan.destinations.items():
        pairs = [(source, staging / destination)]
        analysis = source.with_name(source.name + ANALYSIS_SUFFIX)
        if analysis.is_file():
            pairs.append((analysis, staging / f"{destination.as_posix()}{ANALYSIS_SUFFIX}"))
        for origin, target in pairs:
            size += _copy_verified(origin, target)
            files += 1
    return files, size


def _rewrite(plan: CollectPlan, out: Path) -> None:
    for file_ref, source in plan.rewrites:
        destination = plan.destinations[source]
        values = {
            "RelativePathType": PROJECT_RELATIVE,
            "RelativePath": destination.as_posix(),
            "Path": str(out / destination),
        }
        for tag, value in values.items():
            element = file_ref.find(tag)
            if element is None:
                element = ET.SubElement(file_ref, tag)
            element.set("Value", value)


def _require_new_project(out: Path) -> None:
    if out.exists():
        raise _invalid(
            f"Output already exists: {out}",
            "Choose an --out folder that does not exist yet.",
            out=str(out),
        )
    if not out.parent.is_dir():
        raise _invalid(
            f"Output directory does not exist: {out.parent}",
            "Create the parent directory first.",
            out=str(out),
        )


def collect_live_set(document: LiveSetDocument, out: Path) -> dict[str, Any]:
    """Write ``out/<set name>`` plus its samples; ``out`` appears only when complete."""
    out = out.absolute()
    _require_new_project(out)
    plan = plan_collect(document)
    _rewrite(plan, out)
    staging = Path(tempfile.mkdtemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp"))
    try:
        (staging / PROJECT_INFO_DIR).mkdir()
        file_count, byte_count = _copy_samples(plan, staging)
        save_live_set(document, staging / document.path.name)
        _require_new_project(out)
        os.rename(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "path": str(document.path),
        "out": str(out),
        "set": str(out / document.path.name),
        "rewritten_reference_count": len(plan.rewrites),
        "retained_reference_count": plan.retained_count,
        "copied_file_count": file_count,
        "copied_byte_count": byte_count,
        "samples": listed_details(
            "files", sorted(destination.as_posix() for destination in plan.destinations.values())
        ),
    }
