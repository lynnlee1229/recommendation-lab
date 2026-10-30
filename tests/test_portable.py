import hashlib
import json
import stat
import zipfile
import pytest
from scripts.export_portable import build_release
from scripts.restore_portable import restore, verify_packets, digest, require_managed_target


def release(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'app.py').write_text('print("hello")\n')
    (source / 'sample.bin').write_bytes(b'0123456789' * 10)
    (source / 'lab.sh').write_text('#!/bin/sh\ntrue\n')
    out = tmp_path / 'release'
    payload = {'code/app.py': source / 'app.py', 'data/raw.bin': source / 'sample.bin', 'lab.sh': source / 'lab.sh'}
    spec = build_release(payload, {}, out, 'test-revision', budget=50)
    return source, out, spec


def test_roundtrip_multiple_packets_and_large_raw_payload(tmp_path):
    source, out, spec = release(tmp_path)
    target = tmp_path / 'restored'
    assert any(p['kind'] == 'file' for p in spec['packets'])
    result = restore(out, target)
    assert result['state'] == 'PASS'
    assert (target / 'code/app.py').read_bytes() == (source / 'app.py').read_bytes()
    assert (target / 'data/raw.bin').read_bytes() == (source / 'sample.bin').read_bytes()
    assert (target / 'lab.sh').stat().st_mode & stat.S_IXUSR
    assert result['environment'] == 'NOT_INSTALLED'


def test_corruption_rejected_before_destination_is_created(tmp_path):
    _, out, spec = release(tmp_path)
    p = out / spec['packets'][0]['name']
    p.write_bytes(p.read_bytes() + b'corrupt')
    target = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match='checksum'):
        restore(out, target)
    assert not target.exists()


def test_existing_destination_is_preserved(tmp_path):
    _, out, _ = release(tmp_path)
    target = tmp_path / 'existing'
    target.mkdir()
    (target / 'notes.txt').write_text('learner edits')
    with pytest.raises(FileExistsError):
        restore(out, target)
    assert (target / 'notes.txt').read_text() == 'learner edits'


def malicious_release(tmp_path, names, symlink=False):
    out = tmp_path / 'evil'
    out.mkdir()
    archive = out / 'content-01.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for name in names:
            info = zipfile.ZipInfo(name)
            if symlink:
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, b'x')
    spec = {'format': 'local-study-release-v1', 'source_revision': 'test', 'target_profile': 'test',
            'files': [{'path':'safe.txt','bytes':1,'sha256':hashlib.sha256(b'x').hexdigest()}],
            'packets':[{'name':archive.name,'kind':'zip','bytes':archive.stat().st_size,'sha256':digest(archive)}]}
    (out / 'manifest.json').write_text(json.dumps(spec))
    return out


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a/../../escape', 'C:\\escape', 'a\\b'])
def test_archive_traversal_is_rejected_even_with_valid_packet_checksum(tmp_path, name):
    out = malicious_release(tmp_path, [name])
    with pytest.raises(ValueError):
        verify_packets(out)
    assert not (tmp_path / 'escape').exists()


def test_symlink_payload_is_rejected(tmp_path):
    out = malicious_release(tmp_path, ['safe.txt'], symlink=True)
    with pytest.raises(ValueError, match='symlink'):
        verify_packets(out)


def test_duplicate_archive_member_is_rejected(tmp_path):
    with pytest.warns(UserWarning, match='Duplicate name'):
        out = malicious_release(tmp_path, ['safe.txt', 'safe.txt'])
    with pytest.raises(ValueError, match='duplicate'):
        verify_packets(out)


def test_setup_requires_a_restored_target_and_rejects_redirected_venv(tmp_path):
    with pytest.raises(ValueError, match='newly restored'):
        require_managed_target(tmp_path)
    (tmp_path / '.portable-receipt.json').write_text('{}')
    (tmp_path / '.venv').symlink_to(tmp_path / 'outside')
    with pytest.raises(ValueError, match='redirected'):
        require_managed_target(tmp_path)
