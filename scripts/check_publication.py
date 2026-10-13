"""Check the exact publication set without printing any detected secret."""
import argparse
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
KEY=re.compile(rb'sk-(?:cp-)?[A-Za-z0-9_-]{24,}')
ASSIGN=re.compile(rb'(?m)^\s*(?:[A-Z0-9_]*API_KEY|[A-Z0-9_]*TOKEN)\s*=\s*[\"\']?([^\s\"\'#]+)')
PLACEHOLDERS=(b'replace_',b'your_',b'test-',b'example',b'not-needed',b'ollama')

def check(tracked=True):
    if tracked:
        raw=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT)
        files=[ROOT/p.decode() for p in raw.split(b'\0') if p]
    else:
        files=[p for p in ROOT.rglob('*') if p.is_file() and not set(p.relative_to(ROOT).parts)&{'.git','.venv','artifacts','.pytest_cache','__pycache__'}]
    problems=[]
    for p in files:
        rel=p.relative_to(ROOT)
        if p.name=='.env' or set(rel.parts)&{'.venv','artifacts'} or p.name in ('server.pid','server.log'):
            problems.append(str(rel)+': private runtime file in publication set');continue
        if p.stat().st_size>90*1024*1024:problems.append(str(rel)+': oversized file');continue
        data=p.read_bytes()
        if KEY.search(data):problems.append(str(rel)+': credential-like token detected');continue
        for match in ASSIGN.finditer(data):
            value=match.group(1)
            if not value.startswith(PLACEHOLDERS) and value not in (b'""',b"''"):
                problems.append(str(rel)+': non-placeholder key assignment');break
    if problems:raise SystemExit('\n'.join(problems))
    print(f'Publication check passed: {len(files)} files; no runtime credentials or private artifacts detected.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tracked',action='store_true');a=p.parse_args();check(a.tracked)
