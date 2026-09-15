# Enrollment implementation workgraph

The shared onboarding and trained path live in the isolated
`feature/wake-word-onboarding` worktree.

| Work | Dependencies | Result |
| --- | --- | --- |
| Shared private sample collection and transcript review | Existing ASR providers | Implemented; CLI and terminal approval/cancellation tested |
| Per-word engine profiles | Config model | Implemented; named profiles survive default-backend changes |
| Independent Whisper preprocessing | Original OpenAI feature contract | Implemented; mandatory reference parity test |
| Frozen encoder integration | Preprocessing, public OpenVINO Rust wrapper | Implemented in a supervised persistent worker; real CPU proof |
| Head training/calibration/validation | Embeddings, disjoint labeled clips | Implemented; immutable validated artifacts |
| Concurrent engine routing and shared head scoring | Profiles, worker, head artifacts | Implemented; concurrent-pass and file proofs |
| Retention and re-enrollment | Shared samples, transactional config activation | Implemented; explicit retention and `--reuse-recordings` |
| Regression and license gates | Integrated feature | Existing CI gates retained; controlled-error IPC tests |
| Product accuracy and accelerator qualification | Integrated prototype | Separate follow-up, not a release blocker |

Pi workers were assigned bounded config/routing, encoder, and numerical review
work against the GB10 provider. Checkpoints exposed a LAN connectivity problem,
file writes that replaced existing code, and numerical/native-ABI issues in early
drafts. The final integration uses the existing safe OpenVINO Rust wrapper;
all native and numerical claims were independently checked by the integrating
worker. Unverified draft implementations were not imported.

The trained CLI currently takes a labeled dataset or retained training session.
The guided terminal flow covers transcript onboarding; a full guided
positive/negative/calibration/validation recording flow is a follow-up usability
step for the experimental trained mode. No automatic global backend change
reinterprets already enrolled words.
