"""Download public, checksummed inputs only; no training or cloud account credentials."""
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,shutil,time,urllib.request,subprocess
ROOT=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(1<<20),b''):h.update(data)
 return h.hexdigest()

def fetch(row):
 relative=PurePosixPath(row['destination'])
 if relative.is_absolute() or '..' in relative.parts or '\\' in row['destination']:
  raise ValueError('Unsafe destination')
 destination=ROOT.joinpath(*relative.parts)
 if destination.is_file():
  if sha(destination)!=row['sha256']:raise ValueError('Existing file mismatch: '+str(relative))
  return str(relative)
 if not row['url'].startswith(('https://www.kaggle.com/','https://raw.githubusercontent.com/','https://huggingface.co/')):
  raise ValueError('Unexpected public source')
 destination.parent.mkdir(parents=True,exist_ok=True)
 partial=destination.with_suffix(destination.suffix+'.part')
 for attempt in range(3):
  try:
   if partial.is_file() and partial.stat().st_size==row['bytes']:
    if sha(partial)==row['sha256']:break
    partial.unlink()
   offset=partial.stat().st_size if partial.is_file() else 0
   if offset>row['bytes']:
    partial.unlink();offset=0
   if row.get('downloader')=='curl':
    subprocess.run(['curl','-fLsS','--retry','2','--max-time','180','-o',str(partial),row['url']],check=True)
    if partial.stat().st_size!=row['bytes'] or sha(partial)!=row['sha256']:
     raise ValueError('Downloaded file differs from frozen data: '+str(relative))
    break
   request=urllib.request.Request(row['url'],headers={'Range':f'bytes={offset}-'} if offset else {})
   with urllib.request.urlopen(request,timeout=90) as response:
    resume=response.status==206 and response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
    if response.status==206 and not resume:raise ValueError('Unexpected partial response')
    with partial.open('ab' if resume else 'wb') as f:shutil.copyfileobj(response,f)
   if partial.stat().st_size!=row['bytes'] or sha(partial)!=row['sha256']:
    raise ValueError('Downloaded file differs from frozen data: '+str(relative))
   break
  except Exception as exc:
   print('RETRY',relative,attempt+1,type(exc).__name__,flush=True)
   if attempt==2:raise
   time.sleep(attempt+1)
 partial.replace(destination)
 print('VERIFIED',relative,flush=True)
 return str(relative)

if __name__=='__main__':
 spec=json.loads((ROOT/'DATA_DOWNLOADS.json').read_text())
 with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,spec['assets']))
 print('PASS',len(spec['assets']),'public input files; no account credentials used')
