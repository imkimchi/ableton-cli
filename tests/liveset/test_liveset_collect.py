from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from live_set_builder import SampleSpec, SetSpec, TrackSpec, build_live_set_xml, write_live_set

from ableton_cli.errors import AppError
from ableton_cli.liveset.document import load_live_set
from ableton_cli.liveset.ops.collect import collect_live_set


def _set(tmp_path: Path, samples: list[SampleSpec], *, project: bool = False) -> Path:
    folder = tmp_path / ("Song Project" if project else "Loose")
    if project:
        (folder / "Ableton Project Info").mkdir(parents=True)
    spec = SetSpec(tracks=[TrackSpec(tag="AudioTrack", samples=samples)])
    return write_live_set(folder / "Song.als", build_live_set_xml(spec))


def _file(path: Path, content: bytes = b"RIFF") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _refs(set_path: Path) -> list[tuple[str, str, str]]:
    document = load_live_set(set_path)
    return [
        tuple(ref.find(tag).get("Value") for tag in ("RelativePathType", "RelativePath", "Path"))
        for ref in document.root.iterfind(".//SampleRef/FileRef")
    ]


def _tree(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def test_collects_absolute_and_set_relative_samples_into_imported(tmp_path: Path) -> None:
    kick = _file(tmp_path / "Library" / "kick.wav", b"kick")
    _file(tmp_path / "Library" / "kick.wav.asd", b"analysis")
    set_path = _set(
        tmp_path,
        [
            SampleSpec(relative_path="", path=str(kick), relative_path_type=0),
            SampleSpec(relative_path="snare.wav", path="/old/snare.wav", relative_path_type=1),
        ],
    )
    _file(set_path.parent / "snare.wav", b"snare")
    before = set_path.read_bytes()
    out = tmp_path / "Collected" / "Song Project"
    out.parent.mkdir()

    result = collect_live_set(load_live_set(set_path), out)

    assert _tree(out) == [
        "Ableton Project Info",
        "Samples",
        "Samples/Imported",
        "Samples/Imported/kick.wav",
        "Samples/Imported/kick.wav.asd",
        "Samples/Imported/snare.wav",
        "Song.als",
    ]
    assert (out / "Samples/Imported/kick.wav").read_bytes() == b"kick"
    assert _refs(out / "Song.als") == [
        ("3", "Samples/Imported/kick.wav", str(out / "Samples/Imported/kick.wav")),
        ("3", "Samples/Imported/snare.wav", str(out / "Samples/Imported/snare.wav")),
    ]
    assert result["copied_file_count"] == 3
    assert result["copied_byte_count"] == len(b"kick" + b"analysis" + b"snare")
    assert result["rewritten_reference_count"] == 2
    assert result["samples"] == {
        "count": 2,
        "files": ["Samples/Imported/kick.wav", "Samples/Imported/snare.wav"],
    }
    assert set_path.read_bytes() == before
    assert sorted(p.name for p in out.parent.iterdir()) == ["Song Project"]


def test_project_samples_keep_their_relative_path(tmp_path: Path) -> None:
    set_path = _set(tmp_path, [SampleSpec(relative_path="Samples/Recorded/vox.wav")], project=True)
    _file(set_path.parent / "Samples/Recorded/vox.wav")
    out = tmp_path / "Copy Project"

    collect_live_set(load_live_set(set_path), out)

    assert (out / "Samples/Recorded/vox.wav").is_file()
    assert _refs(out / "Song.als")[0][:2] == ("3", "Samples/Recorded/vox.wav")


def test_shared_sample_is_copied_once(tmp_path: Path) -> None:
    kick = _file(tmp_path / "kick.wav")
    sample = SampleSpec(relative_path="", path=str(kick), relative_path_type=0)
    set_path = _set(tmp_path, [sample, sample])

    result = collect_live_set(load_live_set(set_path), tmp_path / "Out")

    assert result["copied_file_count"] == 1
    assert result["rewritten_reference_count"] == 2


def test_different_files_with_the_same_name_get_a_hash_suffix(tmp_path: Path) -> None:
    first = _file(tmp_path / "a" / "Kick.wav", b"one")
    second = _file(tmp_path / "b" / "kick.wav", b"two")
    set_path = _set(
        tmp_path,
        [
            SampleSpec(relative_path="", path=str(first), relative_path_type=0),
            SampleSpec(relative_path="", path=str(second), relative_path_type=0),
        ],
    )

    result = collect_live_set(load_live_set(set_path), tmp_path / "Out")

    names = [Path(name).name for name in result["samples"]["files"]]
    assert "Kick.wav" in names
    assert len({name.casefold() for name in names}) == 2


def test_library_samples_are_retained_unchanged(tmp_path: Path) -> None:
    library = SampleSpec(relative_path="Samples/IR.wav", path="/lib/IR.wav", relative_path_type=7)
    set_path = _set(tmp_path, [library])

    result = collect_live_set(load_live_set(set_path), tmp_path / "Out")

    assert result["retained_reference_count"] == 1
    assert result["copied_file_count"] == 0
    assert _refs(tmp_path / "Out" / "Song.als") == [("7", "Samples/IR.wav", "/lib/IR.wav")]


def test_original_file_ref_to_a_collected_sample_is_rewritten(tmp_path: Path) -> None:
    kick = _file(tmp_path / "kick.wav")
    spec = SetSpec(
        tracks=[
            TrackSpec(
                tag="AudioTrack",
                samples=[SampleSpec(relative_path="", path=str(kick), relative_path_type=0)],
            )
        ]
    )
    root = build_live_set_xml(spec)
    origin = ET.SubElement(ET.SubElement(root.find("LiveSet"), "OriginalFileRef"), "FileRef")
    for tag, value in (("RelativePathType", "0"), ("RelativePath", ""), ("Path", str(kick))):
        ET.SubElement(origin, tag, Value=value)
    set_path = write_live_set(tmp_path / "Song.als", root)
    out = tmp_path / "Out"

    result = collect_live_set(load_live_set(set_path), out)

    rewritten = load_live_set(out / "Song.als").root.find(".//OriginalFileRef/FileRef")
    assert rewritten.find("RelativePathType").get("Value") == "3"
    assert rewritten.find("Path").get("Value") == str(out / "Samples/Imported/kick.wav")
    assert result["rewritten_reference_count"] == 2


def test_missing_sample_aborts_without_writing(tmp_path: Path) -> None:
    set_path = _set(
        tmp_path, [SampleSpec(relative_path="", path="/gone/kick.wav", relative_path_type=0)]
    )
    out = tmp_path / "Out"

    with pytest.raises(AppError) as raised:
        collect_live_set(load_live_set(set_path), out)

    assert raised.value.error_code == "INVALID_ARGUMENT"
    assert raised.value.details["files"] == [{"relative_path_type": "0", "file_name": "kick.wav"}]
    assert not out.exists()
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_project_sample_without_project_folder_is_missing(tmp_path: Path) -> None:
    set_path = _set(tmp_path, [SampleSpec(relative_path="Samples/kick.wav")])

    with pytest.raises(AppError) as raised:
        collect_live_set(load_live_set(set_path), tmp_path / "Out")

    assert raised.value.details["count"] == 1


def test_failed_write_leaves_no_output(tmp_path: Path, monkeypatch) -> None:
    kick = _file(tmp_path / "kick.wav")
    set_path = _set(tmp_path, [SampleSpec(relative_path="", path=str(kick), relative_path_type=0)])
    out = tmp_path / "Out"

    def _fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("ableton_cli.liveset.ops.collect.save_live_set", _fail)
    with pytest.raises(OSError):
        collect_live_set(load_live_set(set_path), out)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["Loose", "kick.wav"]


def _collect(runner, cli_app, *args: str, global_args: tuple[str, ...] = ()):
    result = runner.invoke(cli_app, ["--output", "json", *global_args, "set", "collect", *args])
    return result, json.loads(result.stdout)


def test_cli_collects_and_output_passes_lint(runner, cli_app, tmp_path: Path) -> None:
    kick = _file(tmp_path / "kick.wav")
    set_path = _set(tmp_path, [SampleSpec(relative_path="", path=str(kick), relative_path_type=0)])
    out = tmp_path / "Song Project"

    result, payload = _collect(runner, cli_app, "--path", str(set_path), "--out", str(out))

    assert result.exit_code == 0, result.stdout
    assert payload["command"] == "set collect"
    assert payload["result"]["set"] == str(out / "Song.als")
    lint = runner.invoke(
        cli_app, ["--output", "json", "set", "lint", "--path", str(out / "Song.als")]
    )
    lint_result = json.loads(lint.stdout)["result"]
    assert lint_result["warning_count"] == 0
    assert lint_result["media"]["checked_count"] == 1


@pytest.mark.parametrize("case", ["exists", "missing_parent"])
def test_cli_rejects_unusable_output(runner, cli_app, tmp_path: Path, case: str) -> None:
    set_path = _set(tmp_path, [])
    out = {"exists": tmp_path / "Loose", "missing_parent": tmp_path / "nowhere" / "P"}[case]

    result, payload = _collect(runner, cli_app, "--path", str(set_path), "--out", str(out))

    assert result.exit_code == 2
    assert payload["error"]["code"] == "INVALID_ARGUMENT"
    assert payload["error"]["details"]["out"] == str(out)


@pytest.mark.parametrize(
    ("flag", "exit_code"), [("--read-only", 20), ("--dry-run", 0), ("--plan", 0)]
)
def test_cli_global_flags_do_not_write(
    runner, cli_app, tmp_path: Path, flag: str, exit_code: int
) -> None:
    set_path = _set(tmp_path, [])
    out = tmp_path / "Out"

    result, _ = _collect(
        runner, cli_app, "--path", str(set_path), "--out", str(out), global_args=(flag,)
    )

    assert result.exit_code == exit_code, result.stdout
    assert not out.exists()
