"""GT-free endpoint proposals and explicitly oracle-only gap diagnostics.

The proposal API accepts probability/RGB only. Ground truth is used solely by
the diagnostic scorer, never to generate, order or filter inference proposals.
All lengths refer to the unchanged 512-pixel primary comparison grid.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.graph import route_through_array
from skimage.morphology import skeletonize, disk, black_tophat
from flow_matching_refiner.calibration_metrics import truth_cache,score_mask

NEIGHBOURS=np.ones((3,3),dtype=np.uint8)
CONFIG=dict(threshold=.5,max_endpoint_distance=32.,minimum_endpoint_distance=2.,
            minimum_component_pixels=8,minimum_outward_cosine=0.,
            tangent_trace_steps=6,path_bbox_margin=6,max_path_length=64,
            proposal_tube_radius=2,max_missing_skeleton_component=32)

def _count(mask):return ndi.label(mask,NEIGHBOURS)

def _outward(sk,point,steps):
    current=tuple(point);visited={current};last=current
    for _ in range(steps):
        y,x=current;near=[]
        for yy in range(max(0,y-1),min(sk.shape[0],y+2)):
            for xx in range(max(0,x-1),min(sk.shape[1],x+2)):
                if sk[yy,xx] and (yy,xx) not in visited:near.append((yy,xx))
        if len(near)!=1:break
        current=near[0];visited.add(current);last=current
    v=np.asarray(point,dtype=float)-np.asarray(last,dtype=float);norm=np.linalg.norm(v)
    return v/norm if norm else None

def propose(prob,rgb,config=CONFIG):
    """Propose all eligible disconnected endpoint pairs, no GT argument."""
    p=np.asarray(prob,dtype=np.float32);rgb=np.asarray(rgb,dtype=np.float32)
    assert p.ndim==2 and rgb.shape==(*p.shape,3) and np.isfinite(p).all()
    binary=p>=config['threshold'];labels,nc=_count(binary);sizes=np.bincount(labels.ravel())
    sk=skeletonize(binary);degrees=ndi.convolve(sk.astype(np.uint8),NEIGHBOURS,mode='constant')-sk
    points=np.argwhere(sk&(degrees==1));points=np.asarray([q for q in points if sizes[labels[tuple(q)]]>=config['minimum_component_pixels']],dtype=int).reshape(-1,2)
    directions=[_outward(sk,q,config['tangent_trace_steps']) for q in points]
    union=np.zeros_like(binary);records=[]
    if len(points)<2:return records,union,dict(endpoints=len(points),prediction_components=nc)
    # Dark vessel-like image contrast is computed without labels. Its percentile
    # normalization is per image, without a fitted cross-subject transformation.
    contrast=black_tophat(rgb[:,:,1],footprint=disk(4)).astype(np.float32)
    scale=float(np.percentile(contrast,99));contrast=np.clip(contrast/max(scale,1e-6),0,1)
    cost=(1.+4.*(1.-p)+2.*(1.-contrast)).astype(np.float64)
    pairs=sorted(cKDTree(points).query_pairs(config['max_endpoint_distance']))
    for i,j in pairs:
        a,b=points[i],points[j];distance=float(np.linalg.norm(b-a))
        if distance<config['minimum_endpoint_distance'] or labels[tuple(a)]==labels[tuple(b)]:continue
        va,vb=directions[i],directions[j]
        if va is None or vb is None:continue
        direction=(b-a)/distance;ca=float(va@direction);cb=float(vb@(-direction))
        if min(ca,cb)<config['minimum_outward_cosine']:continue
        low=np.maximum(np.minimum(a,b)-config['path_bbox_margin'],0)
        high=np.minimum(np.maximum(a,b)+config['path_bbox_margin']+1,p.shape)
        sy,sx=low;ey,ex=high
        path,cost_value=route_through_array(cost[sy:ey,sx:ex],tuple(a-low),tuple(b-low),fully_connected=True,geometric=True)
        path=np.asarray(path,dtype=int)+low
        if len(path)>config['max_path_length']:continue
        local=np.zeros_like(binary);local[path[:,0],path[:,1]]=True
        tube=ndi.binary_dilation(local,structure=disk(config['proposal_tube_radius']))
        new=tube&~binary
        if not new.any():continue
        union|=tube;records.append(dict(candidate_id=len(records),start=a.tolist(),end=b.tolist(),endpoint_distance=distance,outward_cosine_start=ca,outward_cosine_end=cb,path_pixels=len(path),path_cost=float(cost_value),new_pixel_opportunities=int(new.sum()),path_yx=path.tolist()))
    return records,union,dict(endpoints=len(points),prediction_components=nc)

def bounded_gaps(binary,cache,config=CONFIG):
    """Annotation-assisted upper bound: short missed GT-skeleton components.

    Require at least two disconnected covered-GT-skeleton anchor groups in a
    one-pixel neighbourhood. A stricter subset joins distinct prediction
    components. This is a proxy for a gap, not a clinical topology annotation.
    """
    missing=cache['sk']&~binary;covered=cache['sk']&binary
    labels,nc=_count(missing);pred_labels,_=_count(binary);bounded=np.zeros_like(binary);strict=np.zeros_like(binary);records=[]
    slices=ndi.find_objects(labels)
    for idx,sl in enumerate(slices,1):
        if sl is None:continue
        sy=max(0,sl[0].start-1);ey=min(binary.shape[0],sl[0].stop+1)
        sx=max(0,sl[1].start-1);ex=min(binary.shape[1],sl[1].stop+1)
        c=labels[sy:ey,sx:ex]==idx;n=int(c.sum())
        if n>config['max_missing_skeleton_component']:continue
        anchors=ndi.binary_dilation(c,NEIGHBOURS)&covered[sy:ey,sx:ex]
        _,groups=_count(anchors)
        if groups<2:continue
        local=bounded[sy:ey,sx:ex];local[c]=True
        component_ids=np.unique(pred_labels[sy:ey,sx:ex][anchors]);component_ids=component_ids[component_ids>0]
        disconnected=len(component_ids)>=2
        if disconnected:strict[sy:ey,sx:ex][c]=True
        records.append(dict(missing_component_id=idx,skeleton_pixels=n,anchor_groups=groups,distinct_prediction_components=len(component_ids),joins_disconnected_predictions=bool(disconnected),thin_points=int((c&cache['thin'][sy:ey,sx:ex]).sum())))
    return bounded,strict,records

def diagnose(prob,rgb,gt,config=CONFIG):
    p=np.asarray(prob,dtype=np.float32);g=np.asarray(gt,dtype=bool);b=p>=config['threshold'];cache=truth_cache(g)
    proposals,tube,meta=propose(p,rgb,config);bounded,strict,components=bounded_gaps(b,cache,config)
    fn=g&~b;fp=b&~g;missing_thin=cache['thin']&~b;missing_sk=cache['sk']&~b
    distance_pred=ndi.distance_transform_edt(~b);distance_gt=ndi.distance_transform_edt(~g)
    oracle=b|(tube&g);raw=b|tube
    # Oracle restores only true pixels in proposal tubes. It is unreachable at
    # inference and is never counted as a learned or measured model improvement.
    scores={name:score_mask(mask,cache) for name,mask in [('O3',b),('oracle_GT_filtered_proposals',oracle),('unfiltered_proposal_union',raw)]}
    thin_n=int(missing_thin.sum());sk_n=int(missing_sk.sum());fn_n=int(fn.sum());fp_n=int(fp.sum())
    stats=dict(**meta,candidate_count=len(proposals),FN_pixels=fn_n,FP_pixels=fp_n,all_wrong_pixels=fn_n+fp_n,
       FN_within2pixels_of_prediction=int((fn&(distance_pred<=2)).sum()),FP_within2pixels_of_GT=int((fp&(distance_gt<=2)).sum()),
       missing_skeleton_points=sk_n,missing_thin_points=thin_n,
       bounded_gap_skeleton_points=int(bounded.sum()),strict_disconnected_gap_skeleton_points=int(strict.sum()),
       bounded_gap_thin_points=int((bounded&cache['thin']).sum()),strict_disconnected_gap_thin_points=int((strict&cache['thin']).sum()),
       proposal_covered_missing_skeleton_points=int((tube&missing_sk).sum()),proposal_covered_missing_thin_points=int((tube&missing_thin).sum()),
       proposed_new_GT_pixels=int((tube&fn).sum()),proposed_new_false_positive_pixels=int((tube&~g&~b).sum()),
       oracle_added_pixels=int((oracle&~b).sum()),bounded_gap_components=len(components),strict_gap_components=sum(c['joins_disconnected_predictions'] for c in components))
    return dict(stats=stats,metrics=scores,proposals=proposals,gap_components=components),dict(binary=b,gt=g,gt_skeleton=cache['sk'],missing_thin=missing_thin,bounded_gaps=bounded,strict_gaps=strict,proposal_tube=tube,oracle_mask=oracle,unfiltered_mask=raw)

def self_test():
    # A genuine internal break, an unrecoverable missing end branch, and a
    # false connection exercise the distinction between proposals and oracles.
    g=np.zeros((64,64),bool);g[31:34,5:58]=True
    p=np.where(g,.9,.05).astype(np.float32);p[31:34,28:34]=.1
    rgb=np.full((64,64,3),.7,dtype=np.float32);rgb[g]=.2
    r,m=diagnose(p,rgb,g);assert r['stats']['bounded_gap_skeleton_points']>0
    assert r['stats']['candidate_count']>0 and r['stats']['oracle_added_pixels']>0
    assert r['metrics']['oracle_GT_filtered_proposals']['dice']>r['metrics']['O3']['dice']
    assert r['stats']['strict_gap_components']>0
    p2=np.where(g,.9,.05).astype(np.float32);p2[:,35:]=.1
    c=truth_cache(g);a,_,_=bounded_gaps(p2>=.5,c);assert not a.any()
    # Inference proposals cannot change with an unused/altered GT.
    x,u,mm=propose(p,rgb);y,v,nn=propose(p.copy(),rgb.copy())
    assert x==y and np.array_equal(u,v) and mm==nn
    z=np.zeros((64,64),np.float32);q,uu,mmm=propose(z,rgb);assert not q and not uu.any()
    print('Gap diagnostic synthetic checks passed; no training.',flush=True)

if __name__=='__main__':self_test()
