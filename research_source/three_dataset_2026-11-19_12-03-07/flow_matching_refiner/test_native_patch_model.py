"""Synthetic native-patch objective, coverage and training-only crop checks."""
import inspect
import random
import unittest

import torch

from flow_matching_refiner.esfm_model import MARGIN,source_logits
from flow_matching_refiner.esfm_objective_all import losses as old_losses
from flow_matching_refiner.native_patch_model import (
    ErrorSelectiveModel,direct_loss,positive_hann,predict_tiled,
    sample_training_patch,tile_starts,training_center_pools,apply_residual,project_residual_512,
)
from flow_matching_refiner.structure_metrics import soft_cldice_loss


class ConstantProbability(torch.nn.Module):
    use_gate=False
    def __init__(self,probability):super().__init__();self.probability=probability
    def error_logits(self,*args):raise AssertionError("Inference must not evaluate unused gate")
    def forward(self,state,time,coarse,rgb,gate_logits=None):
        return torch.logit(torch.full_like(state,self.probability))-state


class ConstantResidual(torch.nn.Module):
    use_gate=False
    def __init__(self,residual):super().__init__();self.residual=residual
    def forward(self,state,time,coarse,rgb,gate_logits=None):return torch.full_like(state,self.residual)


class NativePatchChecks(unittest.TestCase):
    def test_exact_zero_delta_identity_at_fp32_threshold_neighbors(self):
        half=torch.tensor(.5,dtype=torch.float32)
        below=torch.nextafter(half,torch.tensor(0.,dtype=torch.float32))
        above=torch.nextafter(half,torch.tensor(1.,dtype=torch.float32))
        coarse=torch.stack([torch.tensor(0.),below,half,above,torch.tensor(1.)]).reshape(1,1,5)
        output=apply_residual(coarse,torch.zeros_like(coarse))
        self.assertTrue(torch.equal(output,coarse))
        self.assertTrue(torch.equal(output>=.5,coarse>=.5))

    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_constant_stitching_covers_odd_shapes_edges_and_small_images(self):
        for height,width in [(27,35),(8,8),(3,5),(1,1),(11,37)]:
            coarse=torch.full((1,height,width),.6)
            rgb=torch.rand(3,height,width)
            model=ConstantProbability(.3);model.train()
            probability,meta=predict_tiled(model,coarse,rgb,"cpu",tile=8,stride=5,batch=4)
            self.assertEqual(tuple(probability.shape),(1,height,width))
            torch.testing.assert_close(probability,torch.full_like(probability,.3),rtol=0,atol=1e-7)
            self.assertGreater(meta['coverage_min'],0.)
            self.assertEqual(meta['velocity_calls'],meta['totaltiles'])
            self.assertEqual(meta['velocity_forward_batches'],(meta['totaltiles']+3)//4)
            self.assertEqual(meta['gate_calls'],0)
            self.assertTrue(model.training)
            self.assertEqual(probability.dtype,torch.float32)

    def test_positive_weight_and_last_edge_anchor(self):
        weight=positive_hann(16)
        self.assertTrue(bool((weight>0).all()))
        self.assertEqual(tile_starts(35,16,10),[0,10,19])
        covered=torch.zeros(35,dtype=torch.bool)
        for start in tile_starts(35,16,10):covered[start:start+16]=True
        self.assertTrue(bool(covered.all()))
        self.assertEqual(tile_starts(5,16,10),[0])
        with self.assertRaises(ValueError):tile_starts(35,16,17)

    def test_no_gt_argument_or_gate_in_inference(self):
        parameters=inspect.signature(predict_tiled).parameters
        self.assertNotIn('gt',parameters);self.assertNotIn('truth',parameters);self.assertNotIn('fov',parameters)
        coarse=torch.full((1,17,19),.4);rgb=torch.rand(3,17,19)
        p1,_=predict_tiled(ConstantProbability(.7),coarse,rgb,"cpu",tile=8,stride=4,batch=1)
        p4,_=predict_tiled(ConstantProbability(.7),coarse,rgb,"cpu",tile=8,stride=4,batch=4)
        self.assertTrue(torch.equal(p1,p4))

    def test_logit_residual_stitch_and_zero_identity_binary_mask(self):
        coarse=torch.tensor([0.,.2,.49,.5,.8,1.]).repeat(27*35//6+1)[:27*35].reshape(1,27,35)
        rgb=torch.rand(3,27,35)
        probability,delta,metadata=predict_tiled(ConstantResidual(0.),coarse,rgb,"cpu",8,5,4,True)
        self.assertTrue(torch.equal(probability>=.5,coarse>=.5))
        self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))
        self.assertIn('logit-residual',metadata['overlaprule'])
        probability,delta,_=predict_tiled(ConstantResidual(.75),coarse,rgb,"cpu",8,5,4,True)
        torch.testing.assert_close(delta,torch.full_like(delta,.75),rtol=0,atol=2e-7)
        torch.testing.assert_close(probability,apply_residual(coarse,delta),rtol=0,atol=0)

    def test_dataset_nearest_projection_rule_and_coarse512_identity(self):
        import numpy as np
        height,width=513,515
        native=torch.arange(height*width,dtype=torch.float32).reshape(1,height,width)
        centre_y=np.minimum(np.floor((np.arange(512)+.5)*height/512).astype(int),height-1)
        centre_x=np.minimum(np.floor((np.arange(512)+.5)*width/512).astype(int),width-1)
        corner_y=np.floor(np.arange(512)*height/512).astype(int)
        corner_x=np.floor(np.arange(512)*width/512).astype(int)
        expected_pil=native.numpy()[0][centre_y[:,None],centre_x[None,:]]
        expected_cv=native.numpy()[0][corner_y[:,None],corner_x[None,:]]
        for dataset in ('CHASE','DRIVE'):
            self.assertTrue(np.array_equal(project_residual_512(native,dataset).numpy()[0],expected_pil))
        self.assertTrue(np.array_equal(project_residual_512(native,'HRF').numpy()[0],expected_cv))
        self.assertFalse(np.array_equal(expected_cv,expected_pil))
        coarse=torch.rand(1,512,512)
        for dataset in ('CHASE','DRIVE','HRF'):
            projected=project_residual_512(torch.zeros_like(native),dataset)
            self.assertTrue(torch.equal(apply_residual(coarse,projected)>=.5,coarse>=.5))

    def test_training_crops_align_fields_and_use_training_vessel_support(self):
        height,width=21,25
        y,x=torch.meshgrid(torch.arange(height),torch.arange(width),indexing='ij')
        coarse=((y*width+x)/(height*width)).float()[None]
        rgb=coarse.repeat(3,1,1)
        truth=torch.zeros_like(coarse);truth[0,10,12]=1.
        fov=torch.ones_like(coarse)
        pools=training_center_pools(truth,fov)
        crop,meta=sample_training_patch(coarse,rgb,truth,fov,8,random.Random(71),1.,pools)
        self.assertEqual((meta['centre_y'],meta['centre_x']),(10,12))
        self.assertTrue(meta['used_vessel_centre'])
        top,left=meta['top'],meta['left']
        self.assertTrue(torch.equal(crop['coarse'],coarse[:,top:top+8,left:left+8]))
        self.assertTrue(torch.equal(crop['coarse'],crop['rgb'][:1]))
        self.assertEqual(crop['gt'].sum().item(),1.)
        repeated,repeated_meta=sample_training_patch(coarse,rgb,truth,fov,8,random.Random(71),1.,pools)
        self.assertEqual(meta,repeated_meta)
        self.assertTrue(torch.equal(crop['coarse'],repeated['coarse']))
        crop['coarse'].zero_()
        self.assertGreater(coarse.sum().item(),0.)

    def test_small_training_images_pad_targets_out_of_supervision(self):
        coarse=torch.full((1,3,5),.4);rgb=coarse.repeat(3,1,1)
        truth=torch.zeros_like(coarse);fov=torch.ones_like(coarse)
        crop,meta=sample_training_patch(coarse,rgb,truth,fov,8,random.Random(9),1.)
        self.assertTrue(meta['fallback_no_vessels'])
        self.assertEqual(crop['fov'].sum().item(),15.)
        self.assertTrue(bool((crop['coarse']==.4).all()))
        self.assertEqual(tuple(crop['rgb'].shape),(3,8,8))

    def test_direct_loss_exact_old_objective_and_parameter_gradients(self):
        torch.manual_seed(28)
        model=ErrorSelectiveModel(False)
        self.assertEqual(sum(p.numel() for p in model.parameters()),315974)
        coarse=.1+.8*torch.rand(2,1,16,16);rgb=torch.rand(2,3,16,16)
        truth=(torch.rand_like(coarse)>.7).float();fov=torch.ones_like(truth)
        source=source_logits(coarse);t=torch.zeros(2)
        gate=model.error_logits(coarse,rgb);velocity=model(source,t,coarse,rgb,gate)
        errors=((coarse>=.5)!=(truth>=.5)).float()
        expected,parts=old_losses(velocity,MARGIN*(2*truth-1)-source,source+velocity,
            coarse,truth,errors,torch.zeros_like(truth),gate,False,False,soft_cldice_loss,fov)
        actual,actual_parts=direct_loss(model,coarse,rgb,truth,fov)
        self.assertTrue(torch.equal(expected,actual))
        for key,value in parts.items():self.assertTrue(torch.equal(value,actual_parts[key]),key)
        actual.backward()
        self.assertGreater(model.velocity_net.head.weight.grad.abs().sum().item(),0.)
        self.assertGreater(model.error_net[-1].weight.grad.abs().sum().item(),0.)
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))


if __name__=='__main__':unittest.main()
