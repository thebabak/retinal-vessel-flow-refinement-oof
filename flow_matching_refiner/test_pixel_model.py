import unittest

import torch

from flow_matching_refiner.model import integrate, linear_path
from flow_matching_refiner.pixel_model import (
    PixelVelocity, blend_probabilities, objective, probabilities, source_state,
)


class PixelChecks(unittest.TestCase):
    def test_initial_flow_preserves_decision_and_exact_fallback(self):
        torch.manual_seed(11)
        model = PixelVelocity()
        coarse = torch.rand(1,1,32,32)
        coarse[0,0,0,0], coarse[0,0,0,1] = 0., 1.
        rgb = torch.rand(1,3,32,32)
        endpoint, nfe = integrate(model, source_state(coarse), coarse, rgb, 8)
        self.assertTrue(torch.equal(probabilities(endpoint)>=.5, coarse>=.5))
        self.assertEqual(nfe, 8)
        self.assertTrue(torch.equal(blend_probabilities(coarse, 1-coarse, 0), coarse))

    def test_loss_gradients_and_masking(self):
        model = PixelVelocity()
        coarse, truth = torch.rand(2,1,32,32), torch.randint(0,2,(2,1,32,32)).float()
        rgb = torch.rand(2,3,32,32)
        t = torch.tensor([0., .6])
        source, target = source_state(coarse), 2*truth-1
        state, target_velocity = linear_path(source, target, t)
        velocity = model(state, t, coarse, rgb)
        endpoint = state + (1-t[:,None,None,None])*velocity
        loss, parts = objective(velocity, target_velocity, endpoint, truth, torch.ones_like(truth))
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)
        valid = torch.ones_like(truth)
        valid[:,:,:,16:] = 0
        alternative = endpoint.detach().clone()
        alternative[:,:,:,16:] = 50
        first = objective(velocity.detach(), target_velocity, endpoint.detach(), truth, valid)[0]
        second = objective(velocity.detach(), target_velocity, alternative, truth, valid)[0]
        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
