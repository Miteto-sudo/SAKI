import unittest

import torch

from saki_opd.engine import verify_q_blocks
from saki_opd.objectives import coupling_routed_k1_teacher_mode_loss


class QBlockObjectiveIntegrationTest(unittest.TestCase):
    def test_invalid_suffix_teacher_mode_is_safe_for_routed_loss(self):
        student_logits = torch.tensor(
            [[[20.0, -20.0], [20.0, -20.0], [20.0, -20.0]]]
        )
        teacher_logits = student_logits.clone()
        teacher_logits[0, 1] = torch.tensor([-20.0, 20.0])
        proposal_tokens = torch.tensor([[0, 0, 0]])

        verification = verify_q_blocks(
            student_logits,
            teacher_logits,
            proposal_tokens,
            epsilon=100.0,
            generator=torch.Generator().manual_seed(11),
        )
        self.assertEqual(verification.valid_mask.tolist(), [[True, True, False]])
        self.assertEqual(
            verification.correction_mask.tolist(), [[False, True, False]]
        )
        self.assertEqual(verification.teacher_mode_ids.tolist(), [[0, 1, -1]])

        actor_logits = torch.tensor(
            [[[1.4, -0.2], [0.6, 1.1], [0.3, -0.4]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        sampled_token_ids = verification.output_tokens
        current = torch.log_softmax(actor_logits.detach(), dim=-1).gather(
            -1, sampled_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        old_student_logprobs = current.clone()
        teacher_sample_logprobs = current - torch.tensor([[0.4, -0.2, 0.9]])

        output = coupling_routed_k1_teacher_mode_loss(
            actor_logits,
            sampled_token_ids,
            old_student_logprobs,
            teacher_sample_logprobs,
            verification.teacher_mode_ids,
            verification.correction_mask,
            verification.valid_mask,
        )

        self.assertTrue(bool(torch.isfinite(output.total)))
        log_probs = torch.log_softmax(actor_logits, dim=-1)
        expected_accepted = (
            log_probs[0, 0, sampled_token_ids[0, 0]]
            - teacher_sample_logprobs[0, 0]
        ) / 2.0
        expected_teacher_mode = -log_probs[0, 1, 1] / 2.0
        torch.testing.assert_close(output.accepted_rkl, expected_accepted)
        torch.testing.assert_close(output.teacher_mode, expected_teacher_mode)
        torch.testing.assert_close(
            output.total, expected_accepted + expected_teacher_mode
        )

        output.total.backward()
        self.assertTrue(bool(torch.isfinite(actor_logits.grad).all()))
        self.assertGreater(float(actor_logits.grad[0, 0].abs().sum()), 0.0)
        self.assertGreater(float(actor_logits.grad[0, 1].abs().sum()), 0.0)
        torch.testing.assert_close(
            actor_logits.grad[0, 2],
            torch.zeros_like(actor_logits.grad[0, 2]),
            atol=0.0,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
