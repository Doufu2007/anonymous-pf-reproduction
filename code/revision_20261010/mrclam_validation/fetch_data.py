"""Download small MR.CLAM text archives from their official anonymous FTP.

Raw third-party files are stored outside the public reproduction repository.
The official page invites scientific use with a dataset citation, but does not
specify a standardized redistribution license. No redistribution is assumed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import ftplib
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import zipfile

SOURCE = 'https://asrl.utias.utoronto.ca/datasets/mrclam/'
FTP_HOST = 'asrl3.utias.utoronto.ca'

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--datasets', type=int, nargs='+', default=[1,8,9])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(SOURCE, timeout=30) as r:
        page = r.read()
        page_headers = dict(r.headers)
    (args.out/'official-source-page.html').write_bytes(page)
    manifest = {'dataset':'UTIAS MR.CLAM', 'source':SOURCE,
                'retrieved_at':datetime.now(timezone.utc).isoformat(),
                'page_headers':page_headers,
                'page_sha256':sha256(args.out/'official-source-page.html'),
                'license':'No standardized license on official page; scientific use with citation requested. Raw files kept local; redistribution not assumed.',
                'citation':'Leung, Halpern, Barfoot, Liu. The UTIAS Multi-Robot Cooperative Localization and Mapping Dataset. IJRR 30(8):969-974, 2011.',
                'archives':[]}
    with ftplib.FTP(FTP_HOST, timeout=45) as ftp:
        ftp.login()
        ftp.cwd('/MRCLAM')
        listing = ftp.nlst()
        manifest['official_ftp_listing'] = listing
        for number in args.datasets:
            name = f'MRCLAM{number}.zip'
            if name not in listing:
                raise FileNotFoundError(name)
            ftp.voidcmd('TYPE I')
            expected = ftp.size(name)
            archive = args.out/name
            if not archive.exists() or archive.stat().st_size != expected:
                partial = archive.with_suffix('.zip.part')
                # Resume interrupted downloads rather than silently discarding
                # a partial archive. curl opens a separate anonymous FTP session.
                subprocess.run(['curl.exe','--noproxy','*','--connect-timeout','10',
                    '--max-time','180','--retry','2','--retry-delay','1',
                    '--speed-time','30','--speed-limit','1','--continue-at','-',
                    '--output',str(partial),f'ftp://{FTP_HOST}/MRCLAM/{name}'],check=True)
                if partial.stat().st_size != expected:
                    raise RuntimeError('FTP size mismatch')
                partial.replace(archive)
            with zipfile.ZipFile(archive) as z:
                if z.testzip() is not None:
                    raise RuntimeError('ZIP CRC failure')
                root = (args.out/'extracted').resolve()
                root.mkdir(exist_ok=True)
                for item in z.infolist():
                    target = (root/item.filename).resolve()
                    if not target.is_relative_to(root):
                        raise RuntimeError('Unsafe ZIP member')
                z.extractall(root)
            manifest['archives'].append({'dataset_id':number,
                'source':f'ftp://{FTP_HOST}/MRCLAM/{name}',
                'filename':name,'bytes':archive.stat().st_size,
                'sha256':sha256(archive), 'members':z.namelist()})
            (args.out/'download-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            print(f'Downloaded and CRC-verified {name}: {archive.stat().st_size} bytes', flush=True)
    print('Downloaded all selected datasets; no scientific results computed.', flush=True)

if __name__ == '__main__':
    main()
