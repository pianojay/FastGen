"""One preregistered candidate; audit decisions are evidence records, never FID gates."""
import argparse,json,signal,time,traceback
from pathlib import Path
import torch
from afm_fastgen.common import *
from afm_fastgen.training import Trainer
from afm_fastgen.audit import audit
from afm_fastgen.reporting import fixed_panel


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--resume',action='store_true'); args=p.parse_args()
    path=Path(args.config); c=json.loads(path.read_text()); configure(c)
    root=ROOT/c['run_dir']; root.mkdir(parents=True,exist_ok=True)
    with GpuLease():
        campaign_path=ROOT/'runs/stability_campaign/record.json'
        campaign=json.loads(campaign_path.read_text())
        entries=campaign['runs']; candidates=[r for r in entries if r['kind']=='candidate']
        confirmations=[r for r in entries if r['kind']=='confirmation']
        assert len(candidates)<=6 and len(confirmations)<=1
        entry=next(r for r in entries if r['run_dir']==c['run_dir'])
        assert entry['config_sha256']==digest(path)
        assert campaign['protocol_sha256']==digest(ROOT/'resources/stability_audit_protocol.json')
        if entry['kind']=='confirmation':
            assert campaign.get('first_qualifying_candidate') and entry['reference']==campaign['first_qualifying_candidate']
        rp=root/'record.json'; prior=json.loads(rp.read_text()) if rp.exists() else None
        ident={'source':source_identity(),'config_sha256':digest(path),'protected':verify_protected()}
        if prior:
            assert args.resume and prior['state'] not in ['completed','stopped','stopped_by_user']
            assert ident==prior['identity'] and prior['recoveries']<c['max_recoveries']
        else: assert not args.resume
        record=prior or {'identity':ident,'created_unix':time.time(),'attempts':[],'recoveries':0}
        if prior: record['recoveries']+=1
        attempt={'process':process_identity(),'start_unix':time.time(),'resume':args.resume}; record['attempts'].append(attempt)
        def persist(state,**kw):
            record.update(state=state,**kw); attempt['elapsed_seconds']=time.time()-attempt['start_unix']; atomic_json(rp,record)
            current=json.loads(campaign_path.read_text())
            current.update(current_run=c['run_dir'],current_process=process_identity(),pending_decision=record.get('pending_audit') if state=='awaiting_evidence_decision' else None)
            for r in current['runs']:
                if r['run_dir']==c['run_dir']:
                    r.update(state=state,attempt_seconds=sum(a.get('elapsed_seconds',0) for a in record['attempts']),recoveries=record['recoveries'])
                    if 'g_updates' in record: r.update(g_updates=record['g_updates'],d_updates=record['d_updates'])
            current['cumulative_run_seconds']=sum(r.get('attempt_seconds',0) for r in current['runs'])
            atomic_json(campaign_path,current)
        remaining=c['watchdog_seconds']-(time.time()-record['created_unix'])
        assert remaining>0
        def timeout(*_): raise TimeoutError('Preregistered candidate watchdog exhausted')
        signal.signal(signal.SIGALRM,timeout); signal.alarm(int(remaining))
        try:
            persist('starting'); t=Trainer(c,root/'latest.pt' if prior else None)
            atomic_json(root/'parameters.json',t.parameters)
            if not prior: t.save(root/'latest.pt')
            persist('training')
            while t.step<c['horizon_g_updates']:
                row=t.cycle(); row.update(attempt=len(record['attempts']),unix_time=time.time()); append_json(root/'metrics.jsonl',row)
                if t.step%50==0 or t.step==1:
                    atomic_json(root/'status.json',{'state':'training',**row,'memory':memory(),'process':process_identity()})
                    persist('training',g_updates=t.g_updates,d_updates=t.d_updates)
                if t.step%500==0: t.save(root/'latest.pt')
                if t.step%2000==0:
                    saved=rng_state(); fixed_panel(t.ema,c,root/f'panels/{t.step:06d}.png',t.step); restore_rng(saved)
                if t.step in c['audit_updates']:
                    folder=root/f'audits/{t.step:06d}'
                    # A crash during an audit retains its incomplete directory and reruns deterministically.
                    if folder.exists(): folder.rename(folder.with_name(folder.name+f'_interrupted_{time.time_ns()}'))
                    t.save(root/'latest.pt'); persist('auditing',g_updates=t.g_updates,d_updates=t.d_updates)
                    start=time.monotonic(); audit(t.g,t.ema,c,folder,t.step,seed=c['audit_seed']+t.step)
                    append_json(root/'audit_cost.jsonl',{'step':t.step,'seconds':time.monotonic()-start,'memory':memory()})
                    persist('awaiting_evidence_decision',pending_audit=str(folder))
                    # Agent reviews full-class evidence and writes the frozen-rule decision.
                    decision=folder/'decision.json'
                    while not decision.exists(): time.sleep(2)
                    d=json.loads(decision.read_text()); assert d['action'] in ['continue','stop','qualify'] and d['reason']
                    if d['action']!='continue':
                        t.export(root/'one_nfe.pt'); persist('completed' if d['action']=='qualify' else 'stopped',decision=d); break
                    persist('training')
            else:
                t.save(root/'latest.pt'); t.export(root/'one_nfe.pt'); persist('completed',decision={'action':'horizon','reason':'Frozen horizon reached'})
            atomic_json(root/'final.json',{'g_updates':t.g_updates,'d_updates':t.d_updates,'g_real_presentations':t.gs.presentations,'d_real_presentations':t.ds.presentations,'generated_presentations':t.gs.presentations+t.ds.presentations,'optimizer_seconds':t.seconds,'memory':memory(),'parameters':t.parameters,'checkpoint_sha256':digest(root/'latest.pt'),'elapsed_seconds':time.time()-record['created_unix']})
        except BaseException as e:
            persist('stopped_by_user' if isinstance(e,KeyboardInterrupt) else 'failed',error=repr(e),traceback=traceback.format_exc()); raise
        finally: signal.alarm(0)
if __name__=='__main__': main()
