"""Bind a tested runtime revision to unchanged model/data assets without copying them."""
from __future__ import annotations
import argparse,copy,json,subprocess
from pathlib import Path
from backend.online.bundle import bundle_identity,file_sha256,resolve_asset,required_asset_names,validate_manifest,verify_asset

ROOT=Path(__file__).resolve().parents[1]


def rebind(source,output,note_metadata_index=None):
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
    sources.extend(ROOT/name for name in ['src/recall/dssm_trainer.py','src/recall/cf_retrieval.py','src/recall/fusion.py','src/recall/lexical.py',
        'src/training/rank_stack.py','src/training/history.py','src/training/dien_ranker.py','src/training/scoring.py','src/training/recall_prior.py'])
    for path in sources:updated['assets'][path.relative_to(ROOT).as_posix()]=file_sha256(path)
    if note_metadata_index:
        catalog=resolve_asset(ROOT,str(note_metadata_index));updated['note_metadata_index']=catalog.relative_to(ROOT).as_posix()
        for path in [catalog,catalog.with_suffix(catalog.suffix+'.manifest.json')]:updated['assets'][path.relative_to(ROOT).as_posix()]=file_sha256(path)
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
    args=parser.parse_args();rebind(args.bundle,args.output,args.note_metadata_index)