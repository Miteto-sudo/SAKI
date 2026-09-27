import unittest

import torch

from saki_opd.objectives import coupling_routed_k1_teacher_mode_loss


class CouplingRoutedObjectiveTest(unittest.TestCase):
    def _inputs(self):
        logits = torch.tensor(
            [[[1.2, 0.1, -0.4, 0.3], [0.2, 1.0, -0.7, 0.5], [0.3, -0.2, 1.1, 0.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        sampled = torch.tensor([[0, 1, 2]])
        current = torch.log_softmax(logits.detach(), dim=-1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        teacher_sample = current - torch.tensor([[0.4, -0.3, 0.2]])
        teacher_mode = torch.tensor([[3, 2, 1]])
        valid = torch.tensor([[True, True, True]])
        return logits, sampled, current, teacher_sample, teacher_mode, valid

    def test_correction_replaces_k1_and_uses_valid_denominator(self):
        logits, sampled, old, teacher_sample, teacher_mode, valid = self._inputs()
        correction = torch.tensor([[False, True, False]])
        output = coupling_routed_k1_teacher_mode_loss(
            logits,
            sampled,
            old,
            teacher_sample,
            teacher_mode,
            correction,
            valid,
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        current = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        k1 = current - teacher_sample
        expected_rkl = (k1[0, 0] + k1[0, 2]) / 3.0
        expected_tm = -log_probs[0, 1, teacher_mode[0, 1]] / 3.0
        torch.testing.assert_close(output.accepted_rkl, expected_rkl)
        torch.testing.assert_close(output.teacher_mode, expected_tm)
        torch.testing.assert_close(output.total, expected_rkl + expected_tm)
        self.assertEqual(int(output.correction_count), 1)
        self.assertEqual(int(output.valid_count), 3)

    def test_correction_gradient_matches_pure_teacher_mode_nll(self):
        logits, sampled, old, teacher_sample, teacher_mode, valid = self._inputs()
        correction = torch.tensor([[False, True, False]])
        output = coupling_routed_k1_teacher_mode_loss(
            logits,
            sampled,
            old,
            teacher_sample,
            teacher_mode,
            correction,
            valid,
        )
        grad = torch.autograd.grad(output.total, logits, retain_graph=True)[0]
        pure_tm = -torch.log_softmax(logits, dim=-1)[0, 1, teacher_mode[0, 1]] / 3.0
        pure_grad = torch.autograd.grad(pure_tm, logits)[0]
        torch.testing.assert_close(grad[:, 1], pure_grad[:, 1])

    def test_no_corrections_reduces_to_k1_branch(self):
        logits, sampled, old, teacher_sample, teacher_mode, valid = self._inputs()
        output = coupling_routed_k1_teacher_mode_loss(
            logits,
            sampled,
            old,
            teacher_sample,
            teacher_mode,
            torch.zeros_like(valid),
            valid,
        )
        self.assertEqual(float(output.teacher_mode.detach()), 0.0)
        torch.testing.assert_close(output.total, output.accepted_rkl)

    def test_empty_valid_batch_is_finite_zero(self):
        logits, sampled, old, teacher_sample, teacher_mode, valid = self._inputs()
        output = coupling_routed_k1_teacher_mode_loss(
            logits,
            sampled,
            old,
            teacher_sample,
            teacher_mode,
            torch.zeros_like(valid),
            torch.zeros_like(valid),
        )
        self.assertTrue(bool(torch.isfinite(output.total)))
        self.assertEqual(float(output.total.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
