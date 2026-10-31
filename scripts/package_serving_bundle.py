"""Package a content-bound serving bundle offline. Never activates or overwrites a release."""
from __future__ import annotations
import argparse,copy,json,shutil,uuid
from pathlib import Path
import faiss
import numpy as np
import torch
from backend.online.bundle import bundle_identity,file_sha256,required_asset_names,resolve_asset,validate_manifest
from training.history import HISTORY_ORDER

ROOT=Path(__file__).resolve().parents[1]

def contained(root:Path,value)->Path:
    return resolve_asset(root,str(value))

def load_json(path:Path):
    return json.loads(path.read_text())

def copy_asset(source:Path,destination:Path):
    """Copy bytes into a newly created bundle; fail on concurrent source mutation."""
    before=source.stat();destination.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as reader,destination.open('xb') as writer:
        shutil.copyfileobj(reader,writer,length=8*1024*1024)
    after=source.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError(f'Packaging source changed during copy: {source}')
    if file_sha256(source)!=file_sha256(destination):raise ValueError('Packaged bytes differ from source')

def selected_entry(path:Path):
    source=load_json(path);entry=source.get('selected',source)
    if not isinstance(entry,dict) or not {'epoch','directory','checkpoint_sha256','item_vectors_sha256'}.issubset(entry):
        raise ValueError('Selected DSSM JSON must bind epoch, directory, checkpoint and item vectors')
    return entry

def validate_towers(scene,checkpoint,user_path,item_path,selected):
    """A new release cannot accidentally pair old, unmasked or unrelated towers."""
    if file_sha256(checkpoint)!=selected['checkpoint_sha256']:raise ValueError('Selected full DSSM checksum mismatch')
    full=torch.load(checkpoint,map_location='cpu',weights_only=True)
    user=torch.load(user_path,map_location='cpu',weights_only=True)
    item=torch.load(item_path,map_location='cpu',weights_only=True)
    meta=user.get('meta',{});state=user.get('state_dict',{})
    if meta.get('scene')!=scene or meta.get('history_contract')!=HISTORY_ORDER:
        raise ValueError('DSSM scene/history contract mismatch')
    if not meta.get('masked_user_dense') or not bool(state.get('user_mask_profile',torch.tensor(False)).item()):
        raise ValueError('New serving DSSM must persist its masked-profile contract')
    if scene=='rec' and (meta.get('sequence_padding_contract')!='packed_masked_v1' or
            not bool(state.get('seq_mask_padding',torch.tensor(False)).item())):
        raise ValueError('New Recommendation serving DSSM requires packed history masking')
    known=meta.get('known_user_ids');unknown=meta.get('unknown_user_id');size=meta.get('cat_vocabs',{}).get('user_idx',0)
    if not isinstance(known,list) or len(known)!=len(set(known)) or isinstance(unknown,bool) or not isinstance(unknown,int) or unknown in known or not 0<=unknown<size or any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<size for i in known):
        raise ValueError('DSSM known-user/neutral-unknown policy is missing or invalid')
    if item.get('meta',{}).get('history_contract')!=meta['history_contract']:
        raise ValueError('Item/user tower metadata differs')
    for tower in (user,item):
        for name,value in tower.get('state_dict',{}).items():
            if name not in full or not torch.equal(value,full[name]):
                raise ValueError(f'Partial tower does not match selected full checkpoint: {name}')
    if selected.get('user_tower_sha256') and file_sha256(user_path)!=selected['user_tower_sha256']:
        raise ValueError('Selected user tower checksum mismatch')
    return meta

def package_bundle(args,root:Path=ROOT):
    root=Path(root).resolve();scene=args.scene;tag=args.tag
    if scene not in {'search','rec'} or tag not in {'easy','hard'}:raise ValueError('Invalid scene/tag')
    rank=contained(root,args.rank_dir);experiment=contained(root,args.recall_experiment_dir)
    policy_source=contained(root,args.policy_file);policy=copy.deepcopy(load_json(policy_source))
    selected_path=contained(root,args.selected_dssm or experiment/'dssm/selected.json')
    index_selection_path=contained(root,args.selected_index or experiment/'index/selected.json')
    index_selection=load_json(index_selection_path);selected=selected_entry(selected_path)
    epoch=contained(root,selected['directory']);prefix=f'dssm_{scene}_{tag}';trained=getattr(args,'model_prefix',None) or f'dssm_{scene}_optimized'
    full=contained(root,epoch/f'{trained}.pt');user=contained(root,epoch/f'{trained}_user_tower.pt');item=contained(root,epoch/f'{trained}_item_tower.pt')
    metadata=validate_towers(scene,full,user,item,selected)
    vectors=contained(root,getattr(args,'item_vectors',None) or epoch/'item_vectors.bin');ids_path=contained(root,getattr(args,'item_ids',None) or experiment/'dssm/item_ids.npy')
    ids=np.load(ids_path,allow_pickle=False)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(ids)==0 or np.any(ids<0) or len(np.unique(ids))!=len(ids) or np.any(ids[1:]<=ids[:-1]) or int(ids[-1])>=int(metadata.get('cat_vocabs',{}).get('note_idx',0)):raise ValueError('Invalid catalog ID mapping')
    dim=int(metadata.get('retrieval_dim',metadata.get('hidden_dim',0)))
    if dim<=0 or vectors.stat().st_size!=len(ids)*dim*4 or file_sha256(vectors)!=selected['item_vectors_sha256']:
        raise ValueError('Selected catalog vectors do not match mapping/dimension/checksum')
    index_path=contained(root,args.ann_index or experiment/'index/ivfpq.faiss')
    index=faiss.read_index(str(index_path))
    if index.d!=dim or index.ntotal!=len(ids) or index.metric_type!=faiss.METRIC_INNER_PRODUCT:
        raise ValueError('ANN index geometry/catalog differs from selected towers')
    expected_index=index_selection.get('index_sha256')
    if expected_index and file_sha256(index_path)!=expected_index:raise ValueError('Selected ANN checksum mismatch')
    bound_pair=False
    actual_index_hash=file_sha256(index_path)
    provenance_paths=[index_selection_path,experiment/'index/build.json',experiment/'index/manifest.json']
    for provenance in dict.fromkeys(provenance_paths):
        if not provenance.is_file():continue
        document=load_json(provenance);recipe=document.get('recipe',{})
        vector_hash=document.get('selected_item_vectors_sha256',document.get('item_vectors_sha256',recipe.get('item_vectors_sha256')))
        index_hash=document.get('index_sha256')
        model_hash=document.get('selected_checkpoint_sha256')
        if vector_hash and vector_hash!=selected['item_vectors_sha256']:
            raise ValueError('ANN provenance binds a different selected model vector set')
        if model_hash and model_hash!=selected['checkpoint_sha256']:
            raise ValueError('ANN provenance binds a different selected checkpoint')
        if index_hash and index_hash!=actual_index_hash:
            raise ValueError('ANN provenance checksum differs from the selected index')
        bound_pair|=bool(vector_hash and index_hash)
    if not bound_pair:raise ValueError('ANN build metadata must bind both index and selected vector checksums')
    nprobe=args.nprobe if args.nprobe is not None else index_selection.get('nprobe')
    if isinstance(nprobe,bool) or not isinstance(nprobe,int) or not 1<=int(nprobe)<=int(getattr(index,'nlist',max(1,int(nprobe)))):
        raise ValueError('Bind an explicit frozen nprobe via selected-index JSON or --nprobe')
    if args.nprobe is not None and index_selection.get('nprobe') is not None and args.nprobe!=index_selection['nprobe']:
        raise ValueError('Explicit nprobe differs from frozen selection')
    cutoff=int(args.training_cutoff_exclusive)
    if cutoff<=0:raise ValueError('Training cutoff must be positive')
    rank_meta=contained(root,args.rank_manifest or rank/'manifest.json')
    if not all(path.is_file() for path in (rank_meta,rank/'gbdt.txt',rank/'dien.pt')):raise FileNotFoundError('Frozen rank stack is incomplete')
    encoder=None;lexical=None
    if scene=='search':
        if not args.bm25_index or not args.query_encoder_dir or not args.query_encoder_revision:
            raise ValueError('Search needs BM25 and a pinned local query encoder')
        lexical=contained(root,args.bm25_index);encoder=contained(root,args.query_encoder_dir)
        if not lexical.with_suffix(lexical.suffix+'.manifest.json').is_file():raise ValueError('BM25 source manifest is missing')
    else:
        graph=contained(root,policy.get('cf_graph_dir',''))
        if not all((graph/name).is_file() for name in ('graph.npz','users.npy','items.npy','swing.npz')):
            raise ValueError('Frozen CF graph assets are missing')
    output=contained(root,args.output_dir or root/'outputs/deploy'/scene/tag/'bundles'/uuid.uuid4().hex)
    if output.exists():raise FileExistsError(f'Bundle output already exists: {output}')
    output.mkdir(parents=True,exist_ok=False)
    def copy_bound(source,destination):
        copy_asset(contained(root,source),contained(root,destination))
    # Copy only into this brand-new output. On failure leave it for inspection; no recursive cleanup.
    rank_out=output/'rank';recall_out=output/'recall';index_out=recall_out/'index'
    for source,name in ((rank_meta,'manifest.json'),(rank/'gbdt.txt','gbdt.txt'),(rank/'dien.pt','dien.pt')):
        copy_bound(source,rank_out/name)
    copy_bound(user,recall_out/'models'/f'{prefix}_user_tower.pt');copy_bound(item,recall_out/'models'/f'{prefix}_item_tower.pt')
    copy_bound(vectors,index_out/f'{prefix}_item_emb.bin');copy_bound(index_path,index_out/f'{prefix}_ivfpq.faiss')
    copy_bound(ids_path,index_out/f'{prefix}_row2note.npy')
    (index_out/f'{prefix}_item_map.json').write_text(json.dumps({str(int(item)):int(row) for row,item in enumerate(ids)},separators=(',',':')))
    (index_out/f'{prefix}_item_meta.json').write_text(json.dumps({'num_items':len(ids),'dim':dim,'dtype':'float32',
        'selected_checkpoint_sha256':selected['checkpoint_sha256'],'history_contract':metadata['history_contract']},indent=2))
    ann_meta={'scene':scene,'tag':tag,'num_items':len(ids),'dim':dim,'metric':'ip','nprobe':int(nprobe),
        'index_sha256':file_sha256(index_path),'selected_checkpoint_sha256':selected['checkpoint_sha256'],
        'selected_item_vectors_sha256':selected['item_vectors_sha256'],'selected_index_source_sha256':file_sha256(index_selection_path)}
    (index_out/f'{prefix}_ivfpq_meta.json').write_text(json.dumps(ann_meta,indent=2))
    manifest={'version':1,'scene':scene,'tag':tag,'history_order':HISTORY_ORDER,
        'selection_split':'train_internal_validation','allow_dpr':False,'training_cutoff_exclusive':cutoff,
        'rank_dir':rank_out.relative_to(root).as_posix(),'recall_dir':recall_out.relative_to(root).as_posix(),
        'input_cap':args.input_cap,'coarse_cap':args.coarse_cap,'recall_policy':policy,
        'ranking_strategy':getattr(args,'ranking_strategy','learned'),'schema_enabled':getattr(args,'ranking_strategy','learned')!='recall_only',
        'packaging_provenance':{'selected_epoch':selected['epoch'],'selected_dssm_source':str(selected_path.relative_to(root)),
            'selected_dssm_sha256':file_sha256(selected_path),'selected_index_source':str(index_selection_path.relative_to(root)),
            'selected_index_sha256':file_sha256(index_selection_path),'policy_source_sha256':file_sha256(policy_source),
            'rank_manifest_source_sha256':file_sha256(rank_meta),'full_checkpoint_source_sha256':selected['checkpoint_sha256'],
            'producer_script_sha256':file_sha256(Path(__file__)),'publication':'offline package only; no active promotion'}}
    if scene=='search':
        lexical_out=recall_out/'lexical/bm25.sqlite'
        copy_bound(lexical,lexical_out);copy_bound(lexical.with_suffix(lexical.suffix+'.manifest.json'),lexical_out.with_suffix(lexical_out.suffix+'.manifest.json'))
        encoder_out=recall_out/'query_encoder'
        for path in sorted(encoder.rglob('*')):
            if path.is_file():copy_bound(contained(root,path),encoder_out/path.relative_to(encoder))
        manifest.update(bm25_index=lexical_out.relative_to(root).as_posix(),query_encoder_dir=encoder_out.relative_to(root).as_posix(),
            query_encoder_revision=args.query_encoder_revision)
    else:
        graph_out=recall_out/'cf'
        for name in ('graph.npz','users.npy','items.npy','swing.npz'):copy_bound(graph/name,graph_out/name)
        if (graph/'meta.json').is_file():copy_bound(graph/'meta.json',graph_out/'meta.json')
        else:
            source_report=graph/'northstar_metrics.json'
            report=load_json(source_report) if source_report.is_file() else {}
            if not report or float(report.get('last_train_event',cutoff))>=cutoff:
                raise ValueError('Missing CF metadata requires a pretest source report with a legal final timestamp')
            graph_meta={'version':1,'source_graph_dir':graph.relative_to(root).as_posix(),'source_report_sha256':file_sha256(source_report),
                'training_cutoff_exclusive':cutoff,'last_train_event':report['last_train_event'],
                'configuration':policy['cf_config'],'assets':{name:file_sha256(graph/name) for name in ('graph.npz','users.npy','items.npy','swing.npz')}}
            (graph_out/'meta.json').write_text(json.dumps(graph_meta,indent=2))
        policy['cf_graph_dir']=graph_out.relative_to(root).as_posix()
    if getattr(args,'validation_snapshot',None):
        snapshot=contained(root,args.validation_snapshot)
        snapshot_out=output/'evaluation/validation.json';copy_bound(snapshot,snapshot_out)
        manifest['evaluation']={'validation_snapshot':{'path':snapshot_out.relative_to(root).as_posix(),'sha256':file_sha256(snapshot_out)}}
        manifest['packaging_provenance']['validation_source_sha256']=file_sha256(snapshot)
    if getattr(args,'release_evidence',None):
        evidence=contained(root,args.release_evidence);evidence_out=output/'evaluation/release.json'
        copy_bound(evidence,evidence_out)
        manifest.setdefault('evaluation',{})['release_evidence']={'path':evidence_out.relative_to(root).as_posix(),'sha256':file_sha256(evidence_out)}
    if getattr(args,'note_metadata_index',None):
        catalog=contained(root,args.note_metadata_index)
        manifest['note_metadata_index']=catalog.relative_to(root).as_posix()
    names=required_asset_names(manifest,root)
    if manifest.get('evaluation',{}).get('release_evidence'):
        names.add(manifest['evaluation']['release_evidence']['path'])
    runtime_files=[
        'backend/online/'+name for name in ['bundle.py','pipeline.py','live_context.py','shared_query_encoder.py','item_features.py',
            'recall/service.py','recall/dssm.py','preranking/service.py','preranking/gbdt.py','ranking/service.py','ranking/stack.py']]
    runtime_files += ['recall/'+name for name in ['dssm_trainer.py','cf_retrieval.py','fusion.py','lexical.py','context_fusion.py','residual_dssm.py','weighted_context_fusion.py']]
    runtime_files += ['training/'+name for name in ['rank_stack.py','history.py','dien_ranker.py','scoring.py','recall_prior.py']]
    names.update('src/'+name for name in runtime_files if (root/'src'/name).is_file())
    names.update(path.relative_to(root).as_posix() for path in (root/'src/backend/online').rglob('*.py'))
    manifest['assets']={name:file_sha256(contained(root,name)) for name in sorted(names)}
    manifest['bundle_id']=bundle_identity(manifest)
    validate_manifest(manifest,scene,root,tag=tag)
    destination=output/'bundle.json';destination.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return destination,manifest

def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--scene',choices=['search','rec'],required=True);result.add_argument('--tag',choices=['easy','hard'],default='hard')
    for name in ['rank-dir','recall-experiment-dir','policy-file']:
        result.add_argument('--'+name,type=Path,required=True)
    for name in ['selected-dssm','selected-index','ann-index','rank-manifest','output-dir','bm25-index','query-encoder-dir','item-vectors','item-ids']:
        result.add_argument('--'+name,type=Path)
    result.add_argument('--model-prefix',help='Frozen checkpoint basename, optionally relative to selected directory')
    result.add_argument('--release-evidence',type=Path)
    result.add_argument('--validation-snapshot',type=Path)
    result.add_argument('--note-metadata-index',type=Path)
    result.add_argument('--ranking-strategy',choices=['learned','recall_only'],default='learned')
    result.add_argument('--query-encoder-revision');result.add_argument('--nprobe',type=int)
    result.add_argument('--training-cutoff-exclusive',type=int,required=True)
    result.add_argument('--input-cap',type=int,required=True);result.add_argument('--coarse-cap',type=int,required=True)
    return result

def main():
    args=parser().parse_args();path,manifest=package_bundle(args)
    print(json.dumps({'manifest':str(path),'bundle_id':manifest['bundle_id'],'assets':len(manifest['assets']),
        'activated':False,'scene':manifest['scene']},ensure_ascii=False))

if __name__=='__main__':main()