"""Fixed, retrospective application experiment on frozen PFAD point sets.

CT enters only the downstream registration/evaluation stage. No visibility
filter derived from CT is used to construct any source input. The endpoint
measures recovery of the released reference alignment, not clinical TRE.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, math, time
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[3]
PRED=ROOT/'results/first_arrival_refinement/rebuild_v2/selector_attentive_loocv/nested_policy'
DATA=Path(__import__('os').environ.get('PFAD_DATA_ROOT', str(ROOT / 'data/UltraBones100k')))
OUT=ROOT/'artifacts/pfad_registration_application_20260906'
METHODS=('raw','random_matched_refined','geometry_matched_refined','pfad_refined')
ANATOMIES=('foot','tibia','fibula')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply(T,p):
    return p@T[:3,:3].T+T[:3,3]


def sample_mesh(mesh,n,seed):
    rng=np.random.default_rng(seed)
    idx=rng.choice(len(mesh.faces),n,p=mesh.area_faces/mesh.area_faces.sum())
    uv=rng.random((n,2));u=np.sqrt(uv[:,0]);v=uv[:,1]
    weights=np.column_stack([1-u,u*(1-v),u*v])
    points=(mesh.triangles[idx]*weights[:,:,None]).sum(axis=1)
    return points,np.asarray(mesh.face_normals)[idx]


def voxel(points,size,normals=None,cap=None,seed=0):
    keys=np.floor(points/size).astype(np.int64)
    _,inverse=np.unique(keys,axis=0,return_inverse=True)
    n=int(inverse.max())+1;counts=np.bincount(inverse,minlength=n)
    result=np.column_stack([np.bincount(inverse,weights=points[:,j],minlength=n)/counts for j in range(3)])
    normal_result=None
    if normals is not None:
        normal_result=np.column_stack([np.bincount(inverse,weights=normals[:,j],minlength=n)/counts for j in range(3)])
        length=np.linalg.norm(normal_result,axis=1);valid=length>1e-12
        result=result[valid];normal_result=normal_result[valid]/length[valid,None]
    if cap is not None and len(result)>cap:
        idx=np.sort(np.random.default_rng(seed).choice(len(result),cap,replace=False))
        result=result[idx]
        if normal_result is not None:normal_result=normal_result[idx]
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(result))
    if normal_result is not None:cloud.normals=o3d.utility.Vector3dVector(normal_result)
    return cloud


def pose_metrics(estimate,imposed,targets):
    reference=np.linalg.inv(imposed)
    rotation=estimate[:3,:3]@reference[:3,:3].T
    rre=float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))))
    rte=float(np.linalg.norm(estimate[:3,3]-reference[:3,3]))
    residual=estimate@imposed
    v_rms=float(np.sqrt(np.mean(np.sum((apply(residual,targets)-targets)**2,axis=1))))
    return {'rotation_error_deg':rre,'translation_error_mm':rte,
            'virtual_target_rms_displacement_mm':v_rms,'registration_recall':float(rte<4 and rre<5)}


def sanity():
    p=np.array([[0.,0.,0.],[10.,2.,3.],[-4.,8.,6.]])
    D=np.eye(4);D[:3,:3]=Rotation.from_rotvec([.1,-.2,.03]).as_matrix();D[:3,3]=[3.,4.,5.]
    m=pose_metrics(np.linalg.inv(D),D,p)
    assert m['virtual_target_rms_displacement_mm']<1e-10 and m['translation_error_mm']<1e-10
    assert m['rotation_error_deg']<1e-5 and m['registration_recall']==1
    z=pose_metrics(np.eye(4),D,p);assert z['virtual_target_rms_displacement_mm']>0
    return {'known_transform_inverse_recovery':True,'metric_convention_verified':True}


def build_sources(archive,case_seed):
    with np.load(archive) as a:
        q=a['ray_origins_mm'].astype(float)+a['ray_depths_mm'].astype(float)[:,None]*a['ray_directions'].astype(float)
        refined=a['ray_origins_mm'].astype(float)+(a['ray_depths_mm'].astype(float)+a['correction'].astype(float))[:,None]*a['ray_directions'].astype(float)
        keep=a['selection_keep'].astype(bool)
        counts=a['counts'].astype(int)
        geometry_score=-a['cross_sweep_nearest_mm'].astype(float)
        assert counts.sum()==len(q) and np.isfinite(q).all() and np.isfinite(refined).all()
        random_keep=np.zeros(len(q),bool);geometry_keep=random_keep.copy();cursor=0
        rng=np.random.default_rng(case_seed)
        for count in counts:
            idx=np.arange(cursor,cursor+count);k=int(keep[idx].sum())
            random_keep[rng.choice(idx,k,replace=False)]=True
            rank=np.lexsort((idx,-geometry_score[idx]))
            geometry_keep[idx[rank[:k]]]=True
            cursor+=count
        assert keep.sum()==random_keep.sum()==geometry_keep.sum()
        sources={'raw':q,'random_matched_refined':refined[random_keep],
                 'geometry_matched_refined':refined[geometry_keep],'pfad_refined':refined[keep]}
        center=np.median(q,axis=0)
        return {m:p-center for m,p in sources.items()},center,{
            'source_counts':{m:len(p) for m,p in sources.items()},'query_count':len(q),
            'reference_center_from_raw_ultrasound_mm':center.tolist(),
            'counts_matched_per_record':True,'CT_used_to_select_inputs':False}


def case_run(case):
    specimen,anatomy=case;sid=int(specimen[-2:]);aid=ANATOMIES.index(anatomy)
    seed=20260906+sid*1000+aid*100
    destination=OUT/'cases'/f'{specimen}_{anatomy}.json'
    archive=PRED/f'{specimen}_{anatomy}.npz'
    if destination.exists():
        existing=json.loads(destination.read_text())
        if (existing['status']=='complete' and existing['input_prediction_sha256']==sha(archive) and existing['protocol_sha256']==sha(OUT/'protocol.json')):return {'case':specimen+'_'+anatomy,'reused':True}
        raise RuntimeError('Existing case output has mismatched provenance')
    start=time.perf_counter()
    sources,center,metadata=build_sources(archive,seed)
    # Input choices are fixed before target CT is opened.
    mesh_path=DATA/specimen/'CT_bone_segmentations'/f'{anatomy}.stl'
    mesh=trimesh.load_mesh(mesh_path,process=False)
    target_points,target_normals=sample_mesh(mesh,80000,seed+1)
    virtual_targets,_=sample_mesh(mesh,4096,seed+2)
    target_points-=center;virtual_targets-=center
    scales=(2.,1.,.5)
    target_clouds=[voxel(target_points,v,target_normals) for v in scales]
    source_clouds={m:[voxel(points,v,cap=6000,seed=seed+3+k) for k,v in enumerate(scales)] for m,points in sources.items()}
    metadata['source_points_per_scale']={m:[len(c.points) for c in clouds] for m,clouds in source_clouds.items()}
    perturbations=[('aligned',0,0,np.eye(4))]
    rng=np.random.default_rng(seed+4)
    for level,(deg,mm) in enumerate(((5,5),(10,10),(20,20)),1):
        for draw in range(5):
            axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
            direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
            D=np.eye(4);D[:3,:3]=Rotation.from_rotvec(axis*np.deg2rad(deg)).as_matrix();D[:3,3]=mm*direction
            perturbations.append((f'{deg}deg_{mm}mm',level,draw,D))
    estimator=o3d.pipelines.registration.TransformationEstimationPointToPlane(o3d.pipelines.registration.HuberLoss(k=2.))
    records=[]
    for index,(condition,level,draw,D) in enumerate(perturbations):
        # Balanced method order limits systematic timing/warm-cache bias.
        order=METHODS[index%4:]+METHODS[:index%4]
        for method in order:
            T=np.eye(4);fitness=0.;error=None;tick=time.perf_counter()
            try:
                for k,(gate,iterations) in enumerate(zip((30.,15.,5.),(50,30,20))):
                    cloud=o3d.geometry.PointCloud(source_clouds[method][k]);cloud.transform(D)
                    answer=o3d.pipelines.registration.registration_icp(cloud,target_clouds[k],gate,T,estimator,
                        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,relative_rmse=1e-6,max_iteration=iterations))
                    T=np.asarray(answer.transformation).copy();fitness=float(answer.fitness)
                if not np.isfinite(T).all():raise ValueError('Nonfinite registration transform')
            except Exception as exc:
                error=type(exc).__name__+': '+str(exc);T=np.eye(4)
            elapsed=time.perf_counter()-tick
            row=pose_metrics(T,D,virtual_targets)
            if error or fitness==0:row['registration_recall']=0.
            row.update({'specimen':specimen,'anatomy':anatomy,'condition':condition,'level':level,'draw':draw,
                        'method':method,'registration_only_seconds':elapsed,'fitness':fitness,'error':error,
                        'imposed_transform':D.tolist(),'estimated_transform':T.tolist()})
            records.append(row)
    payload={'status':'complete','specimen':specimen,'anatomy':anatomy,'case_seed':seed,
             'input_prediction_sha256':sha(archive),'target_mesh_sha256':sha(mesh_path),
             'protocol_sha256':sha(OUT/'protocol.json'),'records':records,'metadata':metadata,
             'case_seconds':time.perf_counter()-start}
    tmp=destination.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2));tmp.replace(destination)
    return {'case':specimen+'_'+anatomy,'seconds':round(payload['case_seconds'],2),'records':len(records),
            'errors':sum(r['error'] is not None for r in records)}


def summarize():
    rows=[]
    for sid in range(1,15):
        for anatomy in ANATOMIES:
            p=OUT/'cases'/f'specimen{sid:02}_{anatomy}.json'
            if not p.exists():raise RuntimeError('Incomplete cohort; no final summary')
            d=json.loads(p.read_text());assert d['status']=='complete' and len(d['records'])==64
            rows.extend(d['records'])
    keys=['virtual_target_rms_displacement_mm','registration_recall','rotation_error_deg','translation_error_mm','registration_only_seconds']
    summary={};specimen_values={}
    for method in METHODS:
        specimen_values[method]={}
        for key in keys:
            values=[]
            for sid in range(1,15):
                per_anatomy=[]
                for anatomy in ANATOMIES:
                    records=[r for r in rows if r['method']==method and r['specimen']==f'specimen{sid:02}' and r['anatomy']==anatomy and r['level']>0]
                    assert len(records)==15;per_anatomy.append(np.mean([r[key] for r in records]))
                values.append(float(np.mean(per_anatomy)))
            specimen_values[method][key]=values
        summary[method]={key:{'mean':float(np.mean(v)),'sd_across_specimens':float(np.std(v,ddof=1))} for key,v in specimen_values[method].items()}
    comparisons={};tests=[]
    for method in METHODS[:-1]:
        comparisons[method]={}
        for key in keys[:2]:
            ref=np.array(specimen_values[method][key]);pfad=np.array(specimen_values['pfad_refined'][key])
            gain=ref-pfad if key.endswith('_mm') else pfad-ref
            if np.allclose(gain,0):p=1.
            else:p=float(wilcoxon(gain,alternative='two-sided',method='auto').pvalue)
            rng=np.random.default_rng(20260906);boots=gain[rng.integers(0,14,(100000,14))].mean(axis=1)
            obj={'mean_gain_favoring_pfad':float(gain.mean()),'gain_95ci':np.quantile(boots,[.025,.975]).tolist(),
                 'wilcoxon_two_sided_p':p,'wins':int((gain>0).sum()),'losses':int((gain<0).sum())}
            comparisons[method][key]=obj;tests.append(obj)
    running=0.
    for rank,obj in enumerate(sorted(tests,key=lambda x:x['wilcoxon_two_sided_p'])):
        running=max(running,min(1.,(len(tests)-rank)*obj['wilcoxon_two_sided_p']));obj['holm_p']=running
    stratified={}
    for anatomy in ANATOMIES:
        stratified[anatomy]={m:{k:float(np.mean([r[k] for r in rows if r['anatomy']==anatomy and r['method']==m and r['level']>0])) for k in keys} for m in METHODS}
    payload={'status':'complete','cases':42,'specimens':14,'registration_runs':len(rows),'primary_runs':sum(r['level']>0 for r in rows),
             'summary':summary,'comparisons':comparisons,'specimen_values':specimen_values,'by_anatomy_descriptive':stratified,
             'failures_retained':sum(r['error'] is not None for r in rows),'protocol_sha256':sha(OUT/'protocol.json'),
             'interpretation':'Retrospective registration recovery relative to released alignment; synthetic initialization on real anatomy; not clinical TRE or independent acquisition validation'}
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2));print('FINAL',json.dumps(payload['summary']),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--one-case',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);(OUT/'cases').mkdir(exist_ok=True)
    (OUT/'numerical_sanity.json').write_text(json.dumps(sanity(),indent=2))
    print('Solver',o3d.__version__,'workers',args.workers,flush=True)
    cases=[(f'specimen{sid:02}',a) for sid in range(1,15) for a in ANATOMIES]
    if args.one_case:cases=cases[:1]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(case_run,case) for case in cases]
        for f in concurrent.futures.as_completed(tasks):print(json.dumps(f.result()),flush=True)
    if not args.one_case:summarize()


if __name__=='__main__':main()
