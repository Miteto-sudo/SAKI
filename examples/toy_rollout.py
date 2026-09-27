"""Small CPU example of the bridge and maximal-coupling rollout."""

import torch

from saki_opd import maximal_coupling_samples, trust_region_bridge


def main() -> None:
    student = torch.tensor([0.7, 0.2, 0.1]).log()
    teacher = torch.tensor([0.1, 0.3, 0.6]).log()
    epsilon = 0.02

    bridge = trust_region_bridge(student, teacher, epsilon=epsilon)
    samples = maximal_coupling_samples(
        student,
        teacher,
        num_samples=16,
        epsilon=epsilon,
        generator=torch.Generator().manual_seed(42),
    )

    print("beta:", float(bridge.beta))
    print("KL(q||p):", float(bridge.kl_q_p))
    print("p:", bridge.p.tolist())
    print("q:", bridge.q.tolist())
    print("proposals:", samples.proposal_tokens.tolist())
    print("final tokens:", samples.final_tokens.tolist())
    print("corrections:", (~samples.accepted).tolist())


if __name__ == "__main__":
    main()
