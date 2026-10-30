"""Explicit project-local teaching training; creates an immutable run, never latest."""
import json
from pathlib import Path
from reclab.ads_curriculum import ctr_interaction_workshop
from reclab.paths import ROOT

if __name__=='__main__':
    run_id,result=ctr_interaction_workshop()
    output=ROOT/'results/2026-10-30-ads-curriculum'
    output.mkdir(parents=True,exist_ok=True)
    # Keep the course's frozen receipt immutable; reruns receive a unique summary.
    target=output/('fm.json' if not (output/'fm.json').exists() else f'fm-{run_id}.json')
    target.write_text(json.dumps(dict(run_id=run_id,result=result),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'run_id':run_id,'summary':str(target),'rows':result['rows']},ensure_ascii=False))
