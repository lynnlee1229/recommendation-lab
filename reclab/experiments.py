"""Shared append-only experiment receipts for recommendation, search and ads."""
from datetime import datetime, timezone
import hashlib
import platform
import re
import uuid
from .paths import ARTIFACTS, ROOT, atomic_json, read_json

DOMAINS = ('recommendation', 'search', 'ads')
BASE = ARTIFACTS / 'experiments'


def begin(domain, config, provenance):
    if domain not in DOMAINS:
        raise ValueError('Unknown experiment domain')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    folder = BASE / domain / run_id
    folder.mkdir(parents=True, exist_ok=False)
    receipt = {'schema_version': 1, 'domain': domain, 'run_id': run_id,
               'state': 'running', 'config': config, 'provenance': provenance,
               'python': platform.python_version(),
               'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((ROOT / 'reclab').rglob('*.py'))}}
    atomic_json(folder / 'receipt.json', receipt)
    return folder


def finish(folder, result):
    receipt = read_json(folder / 'receipt.json')
    if receipt['state'] != 'running' or (folder / 'result.json').exists():
        raise ValueError('Completed results are immutable; create a new experiment')
    atomic_json(folder / 'result.json', result)
    receipt['state'] = 'completed'
    atomic_json(folder / 'receipt.json', receipt)
    return receipt


def catalog():
    rows = []
    for p in sorted(BASE.glob('*/*/receipt.json')):
        row = read_json(p)
        annotation = p.with_name('annotation.json')
        if annotation.exists():
            row['annotation'] = read_json(annotation)
        rows.append(row)
    return rows


def result(domain, run_id):
    if domain not in DOMAINS or not re.fullmatch(r'[0-9]{8}T[0-9]{6}-[a-f0-9]{8}', run_id):
        raise ValueError('Invalid experiment identity')
    return read_json(BASE / domain / run_id / 'result.json')
