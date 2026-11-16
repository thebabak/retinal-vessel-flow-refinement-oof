import unittest
import torch

from flow_matching_refiner.esfm_model import (
    ErrorSelectiveModel, auxiliary_probability, balanced_gate_loss, make_target,
    predict, source_logits,
)


class SelectiveChecks(unittest.TestCase):
    def test_selective_path_leaves_correct_pixels_unchanged(self):
        coarse=torch.tensor([[[[.9,.1,.7,.2]]]])
        truth=torch.tensor([[[[1.,0.,0.,1.]]]])
        source=source_logits(coarse)
        target,errors=make_target(source,coarse,truth)
        self.assertTrue(torch.equal(errors,torch.tensor([[[[0.,0.,1.,1.]]]])))
        self.assertTrue(torch.equal(target[:,:,:,:2],source[:,:,:,:2]))
        self.assertTrue(torch.equal(target>=0,truth.bool()))

    def test_auxiliary_oracle_is_loss_only_and_blocks_correct_pixel_gradient(self):
        coarse=torch.tensor([[[[.8,.7]]]])
        endpoint=torch.zeros_like(coarse,requires_grad=True)
        errors=torch.tensor([[[[0.,1.]]]])
        p=auxiliary_probability(endpoint,coarse,errors,True)
        p.sum().backward()
        self.assertEqual(endpoint.grad[0,0,0,0].item(),0.)
        self.assertGreater(endpoint.grad[0,0,0,1].item(),0.)

    def test_identity_inference_and_counts_without_ground_truth(self):
        coarse=torch.rand(1,1,32,32)
        rgb=torch.rand(1,3,32,32)
        for gated in (False,True):
            model=ErrorSelectiveModel(gated)
            p,counts=predict(model,coarse,rgb,8)
            self.assertTrue(torch.equal(p>=.5,coarse>=.5))
            self.assertEqual(counts,dict(velocity_calls=8,error_gate_calls=int(gated)))
            p,counts=predict(model,coarse,rgb,8,True)
            self.assertEqual(counts["velocity_calls"],1)

    def test_gate_loss_handles_no_errors(self):
        logits=torch.zeros(2,1,4,4,requires_grad=True)
        loss=balanced_gate_loss(logits,torch.zeros_like(logits))
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_joint_selective_loss_reaches_velocity_and_error_heads(self):
        from flow_matching_refiner.esfm_model import losses
        from flow_matching_refiner.structure_metrics import soft_cldice_loss
        coarse=torch.rand(2,1,32,32)
        truth=(torch.rand_like(coarse)>.7).float()
        rgb=torch.rand(2,3,32,32)
        model=ErrorSelectiveModel(True)
        source=source_logits(coarse)
        target,errors=make_target(source,coarse,truth,True)
        t=torch.zeros(2)
        gate=model.error_logits(coarse,rgb)
        velocity=model(source,t,coarse,rgb,gate)
        total,_=losses(velocity,target-source,source+velocity,coarse,truth,errors,
                       torch.zeros_like(truth),gate,True,True,soft_cldice_loss)
        self.assertTrue(bool(torch.isfinite(total)))
        total.backward()
        self.assertGreater(model.velocity_net.head.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.error_net[-1].weight.grad.abs().sum().item(),0)


if __name__=="__main__":
    torch.set_num_threads(2)
    unittest.main()
