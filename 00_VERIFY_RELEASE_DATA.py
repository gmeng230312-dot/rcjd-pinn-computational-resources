# -*- coding: utf-8 -*-
"""Check every published binary dataset and checkpoint against their SHA-256 digests."""
import csv, hashlib, pathlib, sys
root=pathlib.Path(__file__).resolve().parent
with (root/'DATA_MANIFEST_SHA256.csv').open(newline='',encoding='utf-8') as f:
    records=list(csv.DictReader(f))
issues=[]
for record in records:
    p=root/record['relative_path']
    if not p.is_file():
        issues.append('MISSING: '+record['relative_path']);continue
    if p.stat().st_size!=int(record['bytes']):
        issues.append('SIZE: '+record['relative_path']);continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    if h.hexdigest()!=record['sha256']:
        issues.append('SHA256: '+record['relative_path'])
if issues:
    print('DATA_INTEGRITY_HOLD\n'+'\n'.join(issues));sys.exit(2)
print('DATA_INTEGRITY_PASS | files:',len(records))
