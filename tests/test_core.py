import unittest

import torch

from saki_opd.engine import verify_q_blocks
from saki_opd.math import Bridge, maximal_coupling_samples, verify_speculative_block
from saki_opd.math.bridge import trust_region_bridge


class BridgeAndCouplingTest(unittest.TestCase):
    def test_bridge_respects_constraint_and_normalizes(self):
        student = torch.tensor([0.7, 0.2, 0.1]).log()
        teacher = torch.tensor([0.1, 0.3, 0.6]).log()
        bridge = trust_region_bridge(student, teacher, epsilon=0.02)
        torch.testing.assert_close(bridge.q.sum(), torch.tensor(1.0), atol=1e-6, rtol=0.0)
        self.assertLessEqual(float(bridge.kl_q_p), 0.020001)
        self.assertGreaterEqual(float(bridge.beta), 0.0)
        self.assertLessEqual(float(bridge.beta), 1.0)

    def test_zero_epsilon_is_exact_student_endpoint(self):
        student = torch.tensor([1.2, -0.3, 0.7])
        teacher = torch.tensor([-0.2, 1.1, 0.4])
        bridge = trust_region_bridge(student, teacher, epsilon=0.0)
        expected = torch.log_softmax(student.float(), dim=-1).exp()
        torch.testing.assert_close(bridge.q, expected, atol=0.0, rtol=0.0)
        self.assertEqual(float(bridge.beta), 0.0)
        self.assertEqual(float(bridge.kl_q_p), 0.0)

    def test_empirical_coupling_marginal_matches_q(self):
        student = torch.tensor([0.7, 0.2, 0.1]).log()
        teacher = torch.tensor([0.1, 0.3, 0.6]).log()
        samples = maximal_coupling_samples(
            student,
            teacher,
            num_samples=40000,
            epsilon=0.02,
            generator=torch.Generator().manual_seed(7),
        )
        empirical = torch.bincount(samples.final_tokens, minlength=3).float()
        empirical /= samples.final_tokens.numel()
        bridge = trust_region_bridge(student, teacher, epsilon=0.02)
        torch.testing.assert_close(empirical, bridge.q, atol=0.01, rtol=0.0)
        expected_correction = 0.5 * (bridge.p - bridge.q).abs().sum()
        actual_correction = (~samples.accepted).float().mean()
        self.assertLess(abs(float(actual_correction - expected_correction)), 0.01)

    def test_first_rejection_discards_speculative_suffix(self):
        accepted_bridge = Bridge(
            p=torch.tensor([1.0, 0.0]),
            q=torch.tensor([1.0, 0.0]),
            beta=torch.tensor(0.0),
            kl_q_p=torch.tensor(0.0),
        )
        rejected_bridge = Bridge(
            p=torch.tensor([1.0, 0.0]),
            q=torch.tensor([0.0, 1.0]),
            beta=torch.tensor(1.0),
            kl_q_p=torch.tensor(float("inf")),
        )
        result = verify_speculative_block(
            [accepted_bridge, rejected_bridge, accepted_bridge],
            torch.tensor([0, 0, 0]),
            generator=torch.Generator().manual_seed(3),
        )
        self.assertEqual(result.emitted_tokens.tolist(), [0, 1])
        self.assertEqual(result.accepted.tolist(), [True, False])

    def test_q_block_commits_equal_distribution_block(self):
        logits = torch.randn(2, 4, 17, generator=torch.Generator().manual_seed(5))
        proposal = torch.randint(0, 17, (2, 4), generator=torch.Generator().manual_seed(6))
        result = verify_q_blocks(
            logits,
            logits.clone(),
            proposal,
            epsilon=0.02,
            generator=torch.Generator().manual_seed(7),
        )
        self.assertEqual(result.commit_lengths.tolist(), [4, 4])
        self.assertFalse(bool(result.correction_mask.any()))
        torch.testing.assert_close(result.output_tokens, proposal)
        torch.testing.assert_close(result.teacher_mode_ids, logits.argmax(dim=-1))

    def test_q_block_teacher_mode_and_first_rejection_mask(self):
        student_logits = torch.tensor(
            [[[20.0, -20.0], [20.0, -20.0], [20.0, -20.0]]]
        )
        teacher_logits = student_logits.clone()
        teacher_logits[0, 1] = torch.tensor([-20.0, 20.0])
        proposal = torch.tensor([[0, 0, 0]])
        result = verify_q_blocks(
            student_logits,
            teacher_logits,
            proposal,
            epsilon=100.0,
            generator=torch.Generator().manual_seed(11),
        )

        self.assertEqual(result.commit_lengths.tolist(), [2])
        self.assertEqual(result.valid_mask.tolist(), [[True, True, False]])
        self.assertEqual(result.correction_mask.tolist(), [[False, True, False]])
        self.assertEqual(result.output_tokens[0, :2].tolist(), [0, 1])
        self.assertEqual(result.teacher_mode_ids.tolist(), [[0, 1, -1]])
        self.assertEqual(
            int(result.teacher_mode_ids[0, 1]),
            int(teacher_logits.argmax(dim=-1)[0, 1]),
        )


if __name__ == "__main__":
    unittest.main()
