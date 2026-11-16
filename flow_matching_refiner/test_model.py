"""Synthetic software checks; these do not evaluate a retinal dataset."""
import unittest

import torch

from flow_matching_refiner.model import (
    ConditionalVelocity, integrate, linear_path, masked_dice_loss,
)


class FlowChecks(unittest.TestCase):
    def test_bridge_and_endpoint_estimate(self):
        source = torch.tensor([[[[1., 2.]]], [[[4., -3.]]]])
        target = torch.tensor([[[[5., -1.]]], [[[0., 7.]]]])
        for time in (0., .37, 1.):
            t = torch.full((2,), time)
            state, velocity = linear_path(source, target, t)
            torch.testing.assert_close(state + (1 - time) * velocity, target)
        self.assertTrue(torch.equal(linear_path(source, target, torch.zeros(2))[0], source))
        self.assertTrue(torch.equal(linear_path(source, target, torch.ones(2))[0], target))

    def test_integrator_direction_count_and_fixed_condition(self):
        source = torch.zeros(2, 1, 2, 2)
        coarse, condition = source + 3, torch.ones(2, 1)
        for solver, multiplier in (("euler", 1), ("midpoint", 2)):
            calls = []
            def constant(state, time, supplied_coarse, supplied_condition):
                self.assertIs(supplied_coarse, coarse)
                self.assertIs(supplied_condition, condition)
                self.assertTrue(bool(((time >= 0) & (time <= 1)).all()))
                calls.append(1)
                return torch.ones_like(state) * 2
            result, nfe = integrate(constant, source, coarse, condition, 5, solver)
            torch.testing.assert_close(result, source + 2)
            self.assertEqual(nfe, 5 * multiplier)
            self.assertEqual(len(calls), nfe)
            self.assertTrue(torch.equal(source, torch.zeros_like(source)))
        for steps in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                integrate(constant, source, coarse, condition, steps)

    def test_time_changes_velocity_and_gradient_reaches_time_network(self):
        torch.manual_seed(91)
        model = ConditionalVelocity()
        x = torch.randn(1, 64, 8, 8)
        condition = model.image_condition(torch.rand(1, 3, 32, 32))
        early = model(x, torch.zeros(1), x, condition)
        late = model(x, torch.ones(1), x, condition)
        self.assertGreater((early - late).abs().max().item(), 0)
        late.square().mean().backward()
        self.assertGreater(model.time_mlp[0].weight.grad.abs().sum().item(), 0)

    def test_invalid_pixels_do_not_change_dice_loss(self):
        truth = torch.tensor([[[[1., 0.]]]])
        valid = torch.tensor([[[[1., 0.]]]])
        p1, p2 = truth.clone(), torch.ones_like(truth)
        torch.testing.assert_close(
            masked_dice_loss(p1, truth, valid), masked_dice_loss(p2, truth, valid)
        )


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
