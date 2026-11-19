"""Exact threshold metrics and ranking diagnostics; no fitting or inference."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from sklearn.metrics import average_precision_score
KEYS=('dice','precision','cldice','thin_skeleton_recall')
THRESHOLDS=np.asarray([i/100 for i in range(10,91)],dtype=np.float32)

def truth_cache(gt):
    g=np.asarray(gt)>=.5;assert g.ndim==2;sk=skeletonize(g);radius=ndi.distance_transform_edt(np.pad(g,1))[1:-1,1:-1];thin=sk&(radius<=1.);return dict(g=g,sk=sk,thin=thin,positive=int(g.sum()),negative=int((~g).sum()),thin_count=int(thin.sum()),sk_count=int(sk.sum()))
def cheap_curve(prob,cache,thresholds=THRESHOLDS):
    p=np.asarray(prob,dtype=np.float32);assert p.shape==cache['g'].shape and np.isfinite(p).all() and p.min()>=0 and p.max()<=1;t=np.asarray(thresholds,dtype=np.float32);ps=np.sort(p[cache['g']]);ns=np.sort(p[~cache['g']]);ts=np.sort(p[cache['thin']]);tp=len(ps)-np.searchsorted(ps,t,side='left');fp=len(ns)-np.searchsorted(ns,t,side='left');fn=len(ps)-tp;den=2*tp+fp+fn;dice=np.divide(2*tp,den,out=np.ones(len(t),dtype=float),where=den>0);precision=np.divide(tp,tp+fp,out=np.full(len(t),float(not len(ps))),where=(tp+fp)>0);hits=len(ts)-np.searchsorted(ts,t,side='left');thin=hits/len(ts) if len(ts) else np.ones(len(t));return dict(dice=dice,precision=precision,thin_skeleton_recall=thin,tp=tp,fp=fp,fn=fn,thresholds=t)
def score_mask(mask,cache):
    p=np.asarray(mask,dtype=bool);g=cache['g'];assert p.shape==g.shape;tp=int((p&g).sum());fp=int((p&~g).sum());fn=int((~p&g).sum());empty=not p.any() and not g.any();sp=skeletonize(p);pn=int(sp.sum());gp=cache['sk_count'];a=float((sp&g).sum()/pn) if pn else float(empty);b=float((cache['sk']&p).sum()/gp) if gp else float(empty);cl=2*a*b/(a+b) if a+b else float(empty);return dict(dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,precision=tp/(tp+fp) if tp+fp else float(empty),cldice=cl,thin_skeleton_recall=float((cache['thin']&p).sum()/cache['thin_count']) if cache['thin_count'] else 1.)
def score_prob(prob,cache,threshold):return score_mask(np.asarray(prob,dtype=np.float32)>=np.float32(threshold),cache)
def ap(prob,cache):return float(average_precision_score(cache['g'].reshape(-1),np.asarray(prob,dtype=np.float32).reshape(-1)))
def mean_scores(values):return {k:float(np.mean([r[k] for r in values])) for k in KEYS}
def qualifies(mm,ref):return mm['dice']>=ref['dice']-1e-12 and mm['precision']>=ref['precision']-1e-12 and mm['cldice']>ref['cldice']+1e-12 and mm['thin_skeleton_recall']>ref['thin_skeleton_recall']+1e-12
def gain(mm,ref):return .5*((mm['cldice']-ref['cldice'])+(mm['thin_skeleton_recall']-ref['thin_skeleton_recall']))
def selection_key(candidate,ref):
    m=candidate['metrics'];return (gain(m,ref),m['dice'],m['precision'],-abs(candidate['threshold']-.5),-candidate.get('beta',0),-candidate.get('epoch',0))

def self_test():
    from flow_matching_refiner.structure_metrics_all import metrics
    rng=np.random.default_rng(20261118)
    for g in (rng.random((29,31))>.7,np.zeros((29,31),bool),np.ones((29,31),bool)):
        c=truth_cache(g);p=rng.random(g.shape,dtype=np.float32);p.flat[:4]=[.1,.5,.9,np.nextafter(np.float32(.5),np.float32(0))];curve=cheap_curve(p,c)
        for i,t in enumerate(THRESHOLDS):
            mask=p>=t;a=score_mask(mask,c);b=metrics(mask,g,np.zeros_like(g),1.)
            assert all(a[k]==b[k] for k in KEYS)
            assert a['dice']==curve['dice'][i] and a['precision']==curve['precision'][i] and a['thin_skeleton_recall']==curve['thin_skeleton_recall'][i]
    b=dict(dice=.8,precision=.8,cldice=.8,thin_skeleton_recall=.6);assert not qualifies(b,b);assert qualifies(dict(b,cldice=.81,thin_skeleton_recall=.61),b);assert not qualifies(dict(b,precision=.7999,cldice=.81,thin_skeleton_recall=.61),b);g=np.array([[0,1],[1,0]],bool);c=truth_cache(g);p=np.array([[.1,.7],[.8,.2]],np.float32);assert ap(p,c)==1. and ap(p,c)==ap(p**2,c)
    print('Threshold ties, cheap exact counts, clDice/thin reference equality, AP ranking invariance and gate checks passed. No training.',flush=True)
if __name__=='__main__':self_test()
