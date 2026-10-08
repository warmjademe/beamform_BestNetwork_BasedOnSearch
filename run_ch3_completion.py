"""Finish the original-form comparator and a single frozen held-out test."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from beamnas.common import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs/ch3_grid_v11');args=p.parse_args()
    root=Path(args.runs);status=root/'completion_status.json'
    assert not status.exists()
    save_json(status,{'state':'waiting_for_matched_controls','test_accessed':False})
    try:
        while True:
            current=json.loads((root/'status.json').read_text())
            assert current['state']!='failed',current
            if current['state']=='development_complete':break
            time.sleep(5)
        jobs=[('strict_DNNABF',[sys.executable,'-u','ch3_strict_baseline.py','--out',str(root/'strict_DNNABF')]),
            ('full_development',[sys.executable,'-u','evaluate_ch3_reconstruction.py','--out',str(root/'full_development'),
                '--runs',str(root),'--split','selection_validation','--strict-run',str(root/'strict_DNNABF')]),
            ('test',[sys.executable,'-u','evaluate_ch3_reconstruction.py','--out',str(root/'test'),
                '--runs',str(root),'--split','test','--strict-run',str(root/'strict_DNNABF'),
                '--development-audit',str(root/'full_development')])]
        for name,command in jobs:
            extra={}
            if name=='test':
                report=json.loads((root/'full_development/report.json').read_text())
                extra['pretest_candidate_development_gates_passed']=report['candidate_meets_proximity']
                extra['validation_shortfalls_disclosed']=not report['candidate_meets_proximity']
            save_json(status,{'state':'running','stage':name,'command':command,
                'test_accessed':name=='test',**extra})
            with (root/(name+'_completion.log')).open('x') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((root/'test/report.json').read_text())
        save_json(status,{'state':'complete','candidate_meets_proximity':report['candidate_meets_proximity'],
            'test_accessed':True,'test_used_for_selection':False})
    except Exception:
        save_json(status,{'state':'failed','traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
