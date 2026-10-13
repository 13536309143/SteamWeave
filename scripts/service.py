"""Start/stop a detached localhost service without touching other applications."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
PID=ROOT/'artifacts'/'server.pid'
LOG=ROOT/'artifacts'/'server.log'
PYTHON=ROOT/'.venv'/'bin'/'python'
PORT=8787

def alive(pid):
    try:
        out=subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True)
        return 'sg_agents.app:app' in out
    except subprocess.CalledProcessError:return False

def start():
    if PID.exists() and alive(int(PID.read_text())):
        print(f'Already running: http://127.0.0.1:{PORT}');return
    ROOT.joinpath('artifacts').mkdir(exist_ok=True)
    with LOG.open('a') as log:
        p=subprocess.Popen([str(PYTHON),'-m','uvicorn','sg_agents.app:app','--host','127.0.0.1','--port',str(PORT)],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    PID.write_text(str(p.pid))
    for _ in range(40):
        if p.poll() is not None:raise RuntimeError('Service failed to start; see artifacts/server.log')
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/health',timeout=2)
            print(f'Running: http://127.0.0.1:{PORT} (PID {p.pid})');return
        except Exception:time.sleep(.5)
    raise RuntimeError('Startup timed out; see artifacts/server.log')

def stop():
    if PID.exists():
        pid=int(PID.read_text())
        if alive(pid):os.kill(pid,signal.SIGTERM);print('Stopped SteamWeave service.')
        PID.unlink()
    else:print('Service is not running.')

def install_launch_agent():
    import plistlib
    target=Path.home()/'Library'/'LaunchAgents'/'local.sglab.prognostics.plist'
    target.parent.mkdir(parents=True,exist_ok=True)
    body={'Label':'local.sglab.prognostics','ProgramArguments':[str(PYTHON),'-m','uvicorn','sg_agents.app:app','--host','127.0.0.1','--port',str(PORT)],
          'WorkingDirectory':str(ROOT),'RunAtLoad':True,'KeepAlive':True,
          'StandardOutPath':str(LOG),'StandardErrorPath':str(LOG)}
    target.write_bytes(plistlib.dumps(body))
    print(f'Created login startup configuration: {target}')
    print('Run: launchctl bootstrap gui/$(id -u) '+repr(str(target)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','stop','restart','install-login-startup']);a=p.parse_args()
    if a.action=='start':start()
    elif a.action=='stop':stop()
    elif a.action=='restart':stop();time.sleep(1);start()
    else:install_launch_agent()
