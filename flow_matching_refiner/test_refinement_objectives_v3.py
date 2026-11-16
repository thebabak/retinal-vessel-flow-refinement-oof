"""Synthetic objective checks; no datasets, predictions or models are loaded."""
import unittest

import torch

from flow_matching_refiner.esfm_objective_all import losses as old_losses
from flow_matching_refiner.refinement_objectives_v3 import (
    correct_pixel_identity,losses,missed_thin_recovery,
)
from flow_matching_refiner.structure_metrics import soft_cldice_loss


class RefinementObjectiveChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fields(self):
        torch.manual_seed(819)
        coarse=.1+.8*torch.rand(2,1,9,9)
        truth=(torch.rand_like(coarse)>.6).float()
        errors=((coarse>=.5)!=(truth>=.5)).float()
        return dict(velocity=torch.randn_like(coarse),target_velocity=torch.randn_like(coarse),
                    endpoint=torch.randn_like(coarse,requires_grad=True),coarse=coarse,truth=truth,
                    errors=errors,thin=(torch.rand_like(coarse)>.8).float()*truth,
                    gate_logits=torch.randn_like(coarse),fov=torch.ones_like(coarse))

    def test_zero_new_weights_exactly_recover_original_direct(self):
        fields=self.fields()
        original,original_parts=old_losses(**fields,use_fm=False,selective=False,soft_cldice=soft_cldice_loss)
        updated,updated_parts=losses(**fields,option=1,anchor_weight=0.,thin_weight=0.,soft_cldice=soft_cldice_loss)
        self.assertTrue(torch.equal(original,updated))
        for key,value in original_parts.items():
            self.assertTrue(torch.equal(value,updated_parts[key]),key)

    def test_option2_penalizes_new_damage_on_correct_pixels(self):
        coarse=torch.full((1,1,7,7),.8)
        truth=torch.ones_like(coarse)
        endpoint=torch.full_like(coarse,torch.logit(torch.tensor(.6))).requires_grad_()
        zeros=torch.zeros_like(coarse)
        fields=dict(velocity=endpoint*0.,target_velocity=zeros,endpoint=endpoint,
                    coarse=coarse,truth=truth,errors=zeros,thin=zeros,
                    gate_logits=zeros,fov=torch.ones_like(coarse))
        old,_=old_losses(**fields,use_fm=True,selective=True,soft_cldice=soft_cldice_loss)
        old_gradient=torch.autograd.grad(old,endpoint,retain_graph=True)[0]
        self.assertEqual(old_gradient.abs().sum().item(),0.)
        new,parts=losses(**fields,option=2,anchor_weight=.2,soft_cldice=soft_cldice_loss)
        gradient=torch.autograd.grad(new,endpoint)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(gradient.abs().sum().item(),0.)
        self.assertLess(gradient.mean().item(),0.)  # gradient descent restores higher vessel probability
        self.assertEqual(parts['thin_recovery'].item(),0.)

    def test_thin_recovery_uses_only_missed_gt_thin_support(self):
        coarse=torch.tensor([[[[.2,.8,.2,.2]]]])
        truth=torch.tensor([[[[1.,1.,1.,0.]]]])
        thin=torch.tensor([[[[1.,1.,0.,1.]]]])
        valid=torch.ones_like(coarse)
        endpoint=torch.zeros_like(coarse,requires_grad=True)
        first=missed_thin_recovery(endpoint,coarse,truth,thin,valid)
        gradient=torch.autograd.grad(first,endpoint)[0]
        self.assertLess(gradient[0,0,0,0].item(),0.)
        self.assertTrue(torch.equal(gradient[0,0,0,1:],torch.zeros(3)))
        elsewhere=endpoint.detach().clone();elsewhere[0,0,0,1:]=5.
        self.assertTrue(torch.equal(first,missed_thin_recovery(elsewhere,coarse,truth,thin,valid)))
        repaired=endpoint.detach().clone();repaired[0,0,0,0]=2.
        self.assertLess(missed_thin_recovery(repaired,coarse,truth,thin,valid).item(),first.item())
        self.assertEqual(missed_thin_recovery(endpoint,coarse,truth,torch.zeros_like(thin),valid).item(),0.)

    def test_changes_outside_fov_have_no_loss_or_gradient_effect(self):
        fields=self.fields();fields['fov'][:,:,:,0:3]=0.
        changed={k:value.detach().clone() for k,value in fields.items()}
        outside=fields['fov']==0
        for key in ['velocity','target_velocity','endpoint','gate_logits']:
            changed[key][outside]=9.
        changed['truth'][outside]=1-changed['truth'][outside]
        changed['coarse'][outside]=1-changed['coarse'][outside]
        changed['thin'][outside]=1.
        changed['errors']=((changed['coarse']>=.5)!=(changed['truth']>=.5)).float()
        for option in (1,2):
            first,parts=losses(**fields,option=option,anchor_weight=.2,soft_cldice=soft_cldice_loss)
            second,parts2=losses(**changed,option=option,anchor_weight=.2,soft_cldice=soft_cldice_loss)
            torch.testing.assert_close(first,second,rtol=0,atol=0)
            for key in parts:torch.testing.assert_close(parts[key],parts2[key],rtol=0,atol=0)
            gradient=torch.autograd.grad(first,fields['endpoint'],retain_graph=True)[0]
            self.assertTrue(torch.equal(gradient[outside],torch.zeros_like(gradient[outside])))

    def test_identity_balances_classes_and_handles_absent_classes(self):
        coarse=torch.tensor([[[[.8,.1],[.1,.1]]]])
        truth=torch.tensor([[[[1.,0.],[0.,0.]]]])
        prediction=torch.tensor([[[[.6,.2],[.2,.2]]]],requires_grad=True)
        valid=torch.ones_like(coarse)
        result=correct_pixel_identity(prediction,coarse,truth,valid)
        self.assertAlmostEqual(result.item(),.025,places=7)
        result.backward();self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        for label,base,p in [(0.,.1,.3),(1.,.8,.6)]:
            actual=correct_pixel_identity(torch.full_like(coarse,p),torch.full_like(coarse,base),
                                          torch.full_like(coarse,label),valid)
            self.assertAlmostEqual(actual.item(),.04,places=7)
        wrong=1-truth
        empty=correct_pixel_identity(prediction.detach().requires_grad_(),wrong,truth,valid)
        self.assertEqual(empty.item(),0.)

    def test_invalid_weights_and_options_rejected(self):
        fields=self.fields()
        for option,weight in [(0,.2),(True,.2),(1,-.1),(2,float('nan'))]:
            with self.assertRaises(ValueError):
                losses(**fields,option=option,anchor_weight=weight,soft_cldice=soft_cldice_loss)


if __name__=='__main__':unittest.main()
