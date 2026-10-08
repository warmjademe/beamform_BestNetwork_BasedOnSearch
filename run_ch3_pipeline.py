"""Sequential GPU search, matched controls, and development audit; no test access."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback

from beamnas.common import save_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--data', default='data/ch3_grid_good_v11')
    args = p.parse_args()
    out = Path(args.out)
    assert not out.exists()
    manifest = json.loads((Path(args.data)/'manifest.json').read_text())
    assert manifest['complete']
    out.mkdir(parents=True)
    tasks = [('candidate','search'),('fixed','fixed_dnnabf_modules'),('mse','matched_mse'),('random','random')]
    try:
        for role, task in tasks:
            command = [sys.executable,'-u','run_ch3_reconstruction.py','--task',task,
                '--out',str(out/role),'--data',args.data]
            if role != 'candidate':
                command += ['--reference-search',str(out/'candidate')]
            save_json(out/'status.json',{'state':'running','stage':role,'time':datetime.now(timezone.utc).isoformat(),
                'command':command,'test_accessed':False})
            with (out/(role+'.log')).open('x') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        command = [sys.executable,'-u','evaluate_ch3_reconstruction.py','--out',str(out/'development_audit'),
            '--data',args.data,'--runs',str(out),'--split','selection_validation']
        save_json(out/'status.json',{'state':'running','stage':'development_audit','test_accessed':False})
        with (out/'development_audit.log').open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        report = json.loads((out/'development_audit/report.json').read_text())
        save_json(out/'status.json',{'state':'development_complete','test_accessed':False,
            'candidate_meets_proximity':report['candidate_meets_proximity'],
            'development_report_sha256':sha256(out/'development_audit/report.json')})
    except Exception:
        save_json(out/'status.json',{'state':'failed','traceback':traceback.format_exc(),'test_accessed':False})
        raise


if __name__ == '__main__':
    main()
