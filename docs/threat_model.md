# Threat Model

ATE protects the learner-facing transition data plane. The learner accepts an
update only if an approved step relation, deterministic replay, or an enrolled
receipt service vouches for the full transition record.

## Trusted

- Trainer-side verifier.
- Approved simulator closure: core binary, configuration/assets, dependency
  hashes, RNG schedule, observation function, termination function, reward
  inputs, and declarative actuator transform.
- Local software-root launcher and enrolled step-service process in
  software-only mode.

## Untrusted

- Simulator wrapper code.
- Wrapper-reported learner-visible transitions.
- Wrapper-reported executed-action logs.
- Replay-buffer records before admission.

## Boundary Cases

- Malicious approved simulator closure.
- Malicious policy-approved actuator transform.
- Compromised OS, launcher, verifier, or signing authority.
- Hardware-rooted remote attestation claims without TPM/TEE or equivalent
  platform evidence.

