"""Bind a tested runtime revision to unchanged model/data assets without copying them."""
from __future__ import annotations
import argparse,copy,json,subprocess
from pathlib import Path
from backend.online.bundle import bundle_identity,file_sha256,resolve_asset,required_asset_names,validate_manifest,verify_asset

ROOT=Path(__file__).resolve().parents[1]


def rebind(source,output,note_metadata_index=None,*,rank_dir=None,validation_snapshot=None,numerical_evidence=None,recall_policy_file=None,release_evidence=None):
    source=resolve_asset(ROOT,str(source));output=resolve_asset(ROOT,str(output))
    if output.exists():raise FileExistsError('Runtime manifests are immutable')
    original=json.loads(source.read_text())
    if original['bundle_id']!=bundle_identity(original):raise ValueError('Original manifest identity changed')
    updated=copy.deepcopy(original)
    for name,expected in original['assets'].items():
        path=resolve_asset(ROOT,name)
        if path.is_relative_to(ROOT/'src') and path.suffix=='.py':updated['assets'][name]=file_sha256(path)
        else:verify_asset(path,expected)
    sources=list((ROOT/'src/backend/online').rglob('*.py'))
    sources.extend(ROOT/name for name in ['src/recall/dssm_trainer.py','src/recall/cf_retrieval.py','src/recall/fusion.py','src/recall/lexical.py','src/recall/context_fusion.py','src/recall/residual_dssm.py','src/recall/weighted_context_fusion.py',
        'src/training/rank_stack.py','src/training/history.py','src/training/dien_ranker.py','src/training/scoring.py','src/training/recall_prior.py'])
    for name in ('learned_context_fusion.py', 'head_guard.py', 'multi_interest.py', 'exact_ip_filter.py'):
        runtime=ROOT/'src/recall'/name
        if runtime.is_file():sources.append(runtime)
    for path in sources:updated['assets'][path.relative_to(ROOT).as_posix()]=file_sha256(path)
    if note_metadata_index:
        catalog=resolve_asset(ROOT,str(note_metadata_index));updated['note_metadata_index']=catalog.relative_to(ROOT).as_posix()
        for path in [catalog,catalog.with_suffix(catalog.suffix+'.manifest.json')]:updated['assets'][path.relative_to(ROOT).as_posix()]=file_sha256(path)
    if rank_dir:
        old_rank=resolve_asset(ROOT,original['rank_dir']);new_rank=resolve_asset(ROOT,str(rank_dir))
        old_meta=json.loads((old_rank/'manifest.json').read_text());new_meta=json.loads((new_rank/'manifest.json').read_text())
        expected=copy.deepcopy(old_meta);expected['dien']['inference_contract']='request_fp32_v1'
        if new_meta!=expected:raise ValueError('Runtime rebind may only add the canonical request inference contract')
        for filename in ('gbdt.txt','dien.pt'):
            if file_sha256(old_rank/filename)!=file_sha256(new_rank/filename):raise ValueError('Runtime rebind cannot change learned weights')
        updated['rank_dir']=new_rank.relative_to(ROOT).as_posix()
        for filename in ('manifest.json','gbdt.txt','dien.pt'):
            path=new_rank/filename;updated['assets'][path.relative_to(ROOT).as_posix()]=file_sha256(path)
    if recall_policy_file:
        source_policy=resolve_asset(ROOT,str(recall_policy_file));policy=json.loads(source_policy.read_text())
        weights=policy.get('weights',{});expected=copy.deepcopy(original['recall_policy'])
        expected.update(context_policy='weighted_filter_history',weights=weights)
        if (original['scene']!='rec' or policy!=expected or set(weights)!={'ann','swing','usercf'}
                or weights['ann'] not in (1,2,4,8) or weights['swing'] not in (1,2) or weights['usercf']!=1
                or any(isinstance(value,bool) for value in weights.values())):
            raise ValueError('Policy rebind only permits the frozen eight-trial Recommendation fusion family')
        updated['recall_policy']=policy
        updated['policy_rebind']={'previous_bundle_id':original['bundle_id'],'policy_source_sha256':file_sha256(source_policy),
                                  'learned_weights_changed':False,'requires_full_test_evidence':True}
    for field,path in [('validation_snapshot',validation_snapshot),('numerical_replay',numerical_evidence),('release_evidence',release_evidence)]:
        if path:
            path=resolve_asset(ROOT,str(path));name=path.relative_to(ROOT).as_posix();digest=file_sha256(path)
            updated.setdefault('evaluation',{})[field]={'path':name,'sha256':digest};updated['assets'][name]=digest
    updated['runtime_rebind']={'previous_bundle_id':original['bundle_id'],'previous_manifest_sha256':file_sha256(source),
        'changed_model_or_original_data_assets':False,'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'script_sha256':file_sha256(Path(__file__)),'note_metadata_index_added':bool(note_metadata_index),
        'status':'offline only; requires fresh serving replay before promotion'}
    updated['bundle_id']=bundle_identity(updated)
    validate_manifest(updated,updated['scene'],ROOT,tag=updated['tag'])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(updated,indent=2)+'\n')
    print(json.dumps({'output':str(output),'bundle_id':updated['bundle_id'],'scene':updated['scene'],'activated':False}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--note-metadata-index',type=Path)
    parser.add_argument('--rank-dir',type=Path);parser.add_argument('--validation-snapshot',type=Path);parser.add_argument('--numerical-evidence',type=Path)
    parser.add_argument('--recall-policy-file',type=Path);parser.add_argument('--release-evidence',type=Path)
    args=parser.parse_args();rebind(args.bundle,args.output,args.note_metadata_index,rank_dir=args.rank_dir,validation_snapshot=args.validation_snapshot,numerical_evidence=args.numerical_evidence,recall_policy_file=args.recall_policy_file,release_evidence=args.release_evidence)
