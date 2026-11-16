"""Synthetic CPU scientific-factor and data-isolation tests; no dataset access."""
from pathlib import Path
import sys,random,unittest
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from torch.nn import functional as F
from flow_matching_refiner.controlled_ablations.specs import ARMS,BY_NAME,Arm,plan
from flow_matching_refiner.controlled_ablations.operations import condition_rgb,objective,aligned_augmentation,patch_fields,sample_patch,infer
from flow_matching_refiner.esfm_objective_all import losses as original_losses
from flow_matching_refiner.refinement_objectives_v3 import losses as v3_losses,correct_pixel_identity
from flow_matching_refiner.structure_metrics import soft_cldice_loss
from flow_matching_refiner.native_patch_model import training_center_pools

class Spy(torch.nn.Module):
    def __init__(self,gate=False):super().__init__();self.use_gate=gate
    def error_logits(self,coarse,rgb):return rgb.mean(1,keepdim=True)+coarse
    def forward(self,state,t,coarse,rgb,gate_logits=None):
        result=rgb.mean(1,keepdim=True)*.1
        return result if not self.use_gate else result*gate_logits.sigmoid()

class Factors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def tensors(self):
        torch.manual_seed(13)
        coarse=torch.rand(2,1,16,16)*.8+.1
        truth=(torch.rand_like(coarse)>.7).float();thin=(truth* (torch.rand_like(coarse)>.5)).float()
        errors=((coarse>=.5)!=(truth>=.5)).float()
        velocity=torch.randn_like(coarse,requires_grad=True);target=torch.randn_like(coarse)
        endpoint=torch.logit(coarse)+velocity
        gate=torch.randn_like(coarse,requires_grad=True);fov=torch.ones_like(coarse)
        return velocity,target,endpoint,coarse,truth,errors,thin,gate,fov
    def test_registered_inventory(self):
        self.assertEqual(len(ARMS),16);self.assertEqual(plan()['new_final_fits'],240)
        self.assertEqual(len({a.name for a in ARMS}),16)
    def test_base_recovers_original_objective(self):
        x=self.tensors();spec=Arm('control')
        expected,_=original_losses(*x[:-1],False,False,soft_cldice_loss,x[-1])
        actual,_=objective(spec,*x)
        self.assertTrue(torch.equal(actual,expected))
    def test_o1_factorial_joint_matches_retained_recipe(self):
        x=self.tensors();spec=Arm('joint',thin_weight=.5,anchor_weight=1.)
        actual,_=objective(spec,*x)
        expected,_=v3_losses(*x[:-1],'option1',1.,soft_cldice_loss,x[-1],thin_weight=.5)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-7,rtol=0))
    def test_o2_endpoint_only_matches_zero_anchor_v3(self):
        x=self.tensors();actual,_=objective(BY_NAME['o2_endpoint_only'],*x)
        expected,_=v3_losses(*x[:-1],'option2',0.,soft_cldice_loss,x[-1])
        self.assertTrue(torch.equal(actual,expected))
    def test_o2_anchor_only_preserves_selective_segmentation(self):
        x=self.tensors();actual,parts=objective(BY_NAME['o2_anchor_only'],*x)
        original,_=original_losses(*x[:-1],True,True,soft_cldice_loss,x[-1])
        expected=original+.1*correct_pixel_identity(x[2].sigmoid(),x[3],x[4],x[8])
        self.assertTrue(torch.equal(actual,expected));self.assertGreater(parts['identity'].item(),0)
    def test_dice_weight_only_changes_dice_component(self):
        x=self.tensors();control,parts=objective(Arm('base'),*x)
        for weight in [0.,.25,1.]:
            value,_=objective(Arm('weight',dice_weight=weight),*x)
            self.assertTrue(torch.equal(value,control+(weight-.5)*parts['dice']))
    def test_factor_zero_terms_do_not_create_spurious_gradients(self):
        x=self.tensors();loss,_=objective(BY_NAME['o1_anchor_only'],*x)
        loss.backward();self.assertTrue(torch.isfinite(x[0].grad).all());self.assertTrue(torch.isfinite(x[7].grad).all())
    def test_aug_off_preserves_draw_stream(self):
        a,b=random.Random(42),random.Random(42);field=torch.arange(64).reshape(1,8,8)
        disabled,_=aligned_augmentation([field],a,False)
        aligned_augmentation([field],b,True)
        self.assertTrue(torch.equal(disabled[0],field));self.assertEqual(a.getstate(),b.getstate())
    def test_aug_is_aligned(self):
        field=torch.arange(64).reshape(1,8,8)
        actual,_=aligned_augmentation([field,field+100],random.Random(2),True)
        self.assertTrue(torch.equal(actual[1]-actual[0],torch.full_like(field,100)))
    def sample(self):
        return dict(coarse=torch.full((1,8,8),.4),rgb=torch.rand(3,8,8),g=(torch.rand(1,8,8)>.7).float(),fov=torch.ones(1,8,8),
            n_coarse=torch.full((1,32,48),.4),n_rgb=torch.rand(3,32,48),n_g=(torch.rand(1,32,48)>.7).float(),n_fov=torch.ones(1,32,48))
    def test_information_rgb_only_keeps_gt_and_source_immutable(self):
        d=self.sample();old={k:v.clone() for k,v in d.items()};view=patch_fields(d,BY_NAME['native_degraded_rgb'],'CHASE')
        self.assertTrue(torch.equal(view['gt'],d['n_g']));self.assertTrue(torch.equal(view['fov'],d['n_fov']))
        self.assertFalse(torch.equal(view['rgb'],d['n_rgb']))
        self.assertTrue(all(torch.equal(v,d[k]) for k,v in old.items()))
    def test_information_gt_only_keeps_rgb_and_binary_targets(self):
        d=self.sample();view=patch_fields(d,BY_NAME['native_degraded_gt'],'DRIVE')
        self.assertTrue(torch.equal(view['rgb'],d['n_rgb']))
        expected=F.interpolate(d['g'][None],size=(32,48),mode='nearest-exact')[0]
        self.assertTrue(torch.equal(view['gt'],expected));self.assertTrue(((view['gt']==0)|(view['gt']==1)).all())
    def test_information_gt_hrf_uses_recorded_nearest_rule(self):
        d=self.sample();view=patch_fields(d,BY_NAME['native_degraded_gt'],'HRF')
        self.assertTrue(torch.equal(view['gt'],F.interpolate(d['g'][None],size=(32,48),mode='nearest')[0]))
    def test_information_both_is_factor_product(self):
        d=self.sample();both=patch_fields(d,BY_NAME['native_degraded_both'],'CHASE')
        rgb=patch_fields(d,BY_NAME['native_degraded_rgb'],'CHASE');gt=patch_fields(d,BY_NAME['native_degraded_gt'],'CHASE')
        self.assertTrue(torch.equal(both['rgb'],rgb['rgb']));self.assertTrue(torch.equal(both['gt'],gt['gt']))
    def test_degraded_target_does_not_define_sampling_centres(self):
        truth=torch.zeros(1,320,320);truth[0,159,160]=1;fov=torch.ones_like(truth)
        view=dict(coarse=torch.full_like(truth,.5),rgb=torch.zeros(3,320,320),gt=torch.zeros_like(truth),fov=fov)
        spec=replace(BY_NAME['native_degraded_gt'],vessel_probability=1.)
        crop,meta=sample_patch(view,truth,training_center_pools(truth,fov),random.Random(1),spec)
        self.assertTrue(meta['used_vessel_centre']);self.assertEqual((meta['centre_y'],meta['centre_x']),(159,160))
        self.assertEqual(crop['gt'].sum().item(),0.)
    def test_grid_inference_has_no_gt_argument(self):
        sample=dict(coarse=torch.full((1,16,16),.4),rgb=torch.rand(3,16,16))
        p,_,meta=infer(Spy(),sample,Arm('direct'),'CHASE','cpu')
        self.assertEqual(p.shape,(1,16,16));self.assertEqual(meta['velocity_calls'],1)
    def test_no_rgb_direct_is_invariant_to_rgb(self):
        spec=BY_NAME['direct_no_rgb'];sample=dict(coarse=torch.full((1,16,16),.4),rgb=torch.rand(3,16,16))
        a,_,_=infer(Spy(),sample,spec,'CHASE','cpu');sample['rgb']=torch.rand(3,16,16)
        b,_,_=infer(Spy(),sample,spec,'CHASE','cpu');self.assertTrue(torch.equal(a,b))
    def test_no_rgb_o2_gate_and_velocity_are_invariant(self):
        spec=BY_NAME['o2_no_rgb'];sample=dict(coarse=torch.full((1,16,16),.4),rgb=torch.rand(3,16,16))
        a,_,ma=infer(Spy(True),sample,spec,'CHASE','cpu');sample['rgb']=torch.rand(3,16,16)
        b,_,_=infer(Spy(True),sample,spec,'CHASE','cpu');self.assertTrue(torch.equal(a,b));self.assertEqual(ma['gate_calls'],1)
    def test_native_gt_degradation_inference_needs_no_gt(self):
        sample=dict(coarse=torch.full((1,512,512),.4),rgb=torch.rand(3,512,512),n_coarse=torch.full((1,32,48),.4),n_rgb=torch.rand(3,32,48))
        p,delta,meta=infer(Spy(),sample,BY_NAME['native_degraded_gt'],'CHASE','cpu')
        self.assertEqual(p.shape,(1,512,512));self.assertEqual(delta.shape,(1,32,48));self.assertEqual(meta['gate_calls'],0)
    def test_no_rgb_normalizes_exactly_to_zero(self):
        rgb=condition_rgb(torch.rand(2,3,16,16),BY_NAME['o2_no_rgb'])
        self.assertTrue(torch.equal(2*rgb-1,torch.zeros_like(rgb)))

if __name__=='__main__':unittest.main(verbosity=2)
