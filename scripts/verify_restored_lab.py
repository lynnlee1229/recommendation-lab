"""Verify restored inputs and teaching APIs without starting a server or training."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    from fastapi.testclient import TestClient
    from reclab.service import app

    assets = json.loads((ROOT / 'DATA_DOWNLOADS.json').read_text())['assets']
    for row in assets:
        path = ROOT / row['destination']
        assert path.stat().st_size == row['bytes'], path
        assert hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() == row['sha256'], path
    with TestClient(app, base_url='http://localhost') as client:
        health = client.get('/api/health').json()
        assert health['project'] == str(ROOT) and health['data_ready']
        books = client.get('/api/notebooks').json()['notebooks']
        assert len(books) >= 16 and all(x['available'] for x in books)
        lessons = client.get('/api/curriculum').json()['lessons']
        assert len(lessons) == 45
        for lesson in lessons:
            response = client.get('/api/lectures/' + lesson['file'])
            assert response.status_code == 200 and response.content.startswith(b'%PDF-'), lesson['file']
        summary = client.get('/api/summary').json()
        assert len(summary['runs']) >= 4
        user = summary['users'][0]['user_idx']
        features = client.get(f'/api/users/{user}/features?run_id=baseline-history32')
        assert features.status_code == 200 and len(features.json()['user_vector']) == 32
        recommendation = client.post('/api/recommend', json={
            'user_idx': user, 'run_id': 'baseline-history32', 'channel': 'two_tower', 'limit': 10})
        assert recommendation.status_code == 200 and len(recommendation.json()['items']) == 10
    receipt = {'project': str(ROOT), 'public_assets_verified': len(assets),
               'lecture_api_verified': len(lessons), 'notebook_entries': len(books),
               'retailrocket_runs': len(summary['runs']), 'feature_dimensions': 32,
               'recommendations': 10, 'status': 'PASS'}
    (ROOT / 'logs/restoration-acceptance.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    main()
