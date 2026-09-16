# Wake-word enrollment

This feature adds two ways to adapt a wake word. Transcript aliases remain the
first choice: they use the existing ASR model and need no additional inference.
A trained head is an experimental second choice when transcription does not
reliably represent a phrase. Enrollment never executes the word's action.

## Teach transcript spellings

Start with an existing word:

```sh
omawake word onboard agent
```

Microphone onboarding first offers **Whisper spellings**, **Trainable KWS — learn
from my voice**, or **Trainable KWS with Omaspeak assistance**. No `--engine` flag
or specially named profile is needed. Onboarding reuses configured OpenVINO model
assets, then independently asks for **Training device** and **Finished detector
device**. CPU is the default for training feature extraction; Intel iGPU/NPU are
experimental options. Both encoders are checked before recording. Missing model
or runtime setup is reported before capture.

The Whisper flow records several examples, shows the transcripts, and asks
which exact spellings to accept. Arrow keys and Enter select each choice; Esc
cancels. Unrelated text and hallucinations must be skipped. Nothing changes until
the final Apply step. The existing phrase and command are preserved.

You can create a word during onboarding:

```sh
omawake word onboard assistant --phrase 'Hey unusual name' -- notify-send 'Ready'
```

For silent, repeatable testing, supply WAV files:

```sh
omawake word onboard agent --audio example-1.wav --audio example-2.wav --json
omawake word onboard agent --audio example-1.wav --audio example-2.wav \
  --accept-alias 'the spelling actually observed' --apply
```

Preview does not modify config. Only variants observed in the current examples
can be accepted; onboarding does not invent similar-sounding phrases or widen
matching to arbitrary fuzzy text. Normalization uses the existing matcher's
case, punctuation and word-boundary rules.

`--engine NAME` chooses a configured engine profile. Different models can produce
different spellings, so review examples through the model intended for that word.

## Recording ownership

Recordings stay local. Temporary copies have private permissions and are deleted
when the session ends, including cancellation and errors. Explicitly retained
training datasets are saved before inference and survive a failed or cancelled
activation. Imported source files
are never deleted. Retention requires `--keep-recordings` or the explicit Keep
choice in the terminal flow. Heads do not require retained raw audio to run.

```sh
omawake word recordings agent
omawake word recordings agent --remove SESSION-ID
```

Removing one retained session preserves the word, its command, its heads, and
other sessions. Transcript-onboarding clips are positive examples only; you can
use them when preparing a dataset, but automatic `--reuse-recordings` requires a
previous labeled training session with all three splits. Without retained recordings, changing the encoder may require
new examples. A device/runtime change that preserves the exact encoder contract
can reuse its head.

## Experimental trained heads

The first encoder implementation uses the public OpenVINO C API and the Whisper
base.en encoder. It does not use ONNX Runtime or a patched native runtime. Input
features follow OpenAI Whisper's original preprocessing specification. An
encoder contract hashes actual model XML/BIN bytes and preprocessing/pooling
versions. A head cannot be applied to a different contract just because its
embedding has the same dimensions.

Training uses a small, independently implemented balanced logistic classifier.
Each word has a separate head. All compatible heads score the same normalized
embedding; they do not each load another encoder. Transcript and trained words
can coexist through separate engine groups.

### Guided recording and training

Run `omawake word onboard agent` (or **Teach a wake word** in setup), choose
**Trainable KWS — learn from my voice**, then choose training and deployment
devices. Configured OpenVINO Whisper base.en assets are reused; no specially
named engine or command-line override is needed.
You can also switch to training after reviewing Whisper spellings. CPU is the
qualified device for this experimental path so far.

Onboarding checks that the encoder loads before asking for additional samples.
It collects ten positive and ten negative recordings. If you started with
Whisper spellings, it reuses those positives and asks only for the remainder. Negatives are **other speech without the wake phrase**:
alternate similar-sounding phrases with ordinary everyday sentences. Use a
different utterance each time and vary pace and distance. Each clip should
contain one spoken utterance; silence and clipped audio are retried. The same
VAD used by live detection checks each guided clip immediately. If a pause or
background voice splits it into multiple segments, only that clip is retried;
other examples are kept. Reused transcript examples receive this check too.

The default twenty clips are allocated before fitting:

| Role | Wake phrase | Other speech |
| --- | ---: | ---: |
| Training | 6 | 6 |
| Threshold calibration | 2 | 2 |
| Held-out validation | 2 | 2 |

If more than ten positives were supplied, onboarding collects the same number
of negatives and reserves about 20% of each class for calibration and 20% for
validation. No recording is reused between these roles. The twenty-clip count
is an initial enrollment target, not a reliability guarantee.

Choose whether to retain the recordings. Training then runs locally and shows
held-out misses and false activations. **Apply** saves the word, reviewed aliases,
and validated head together; Cancel leaves the existing word unchanged. An
encoder, segmentation, or validation failure leaves it unchanged too. Collect
a fresh, more representative dataset after addressing a failure; repeatedly
tuning against the same held-out clips does not establish accuracy.

Choosing **Keep recordings locally** saves the complete labeled dataset before
model loading or inference. It remains saved after model-loading, segmentation,
calibration, or validation failure, and after cancelling final activation.
`word train --keep-recordings` also retains data during a preview. The error reports
the manifest path and held-out miss/false-activation counts or calibration
failure. A held-out candidate also needs every score at least 0.025 away from
the threshold; this experimental margin gate can reject a candidate even with
zero classification errors. Replaying the same dataset is a diagnostic, not a
way to improve it or an independent validation.

On successful activation, **Keep recordings locally** retains the dataset for
`word train agent --reuse-recordings`. Otherwise the temporary WAVs and generated
manifest are removed. No dataset editing, audio playback, or wake action is
needed during guided onboarding.

### Optional Omaspeak-assisted examples

After reviewing transcript spellings, choose **Train with Omaspeak examples**.
If Omaspeak is missing, the menu offers **Install Omaspeak for assisted training**
instead. Installation is optional: continue with human recordings, or explicitly
download the checksum-pinned Omaspeak 0.0.1 release to `~/.local/opt` with a launcher
in `~/.local/bin`. The archive's native library and licenses remain bundled. The
installer requires `curl`, `tar`, and `cp`; it never enables a service or silently
installs a model. If synthesis needs configuration, it opens Omaspeak's own setup.

The assisted flow asks for a TTS pronunciation spelling, two confusable negative
phrases, and an ordinary negative sentence. It selects up to three voices from
the configured model. You must explicitly play and approve each voice's
pronunciation before using it. **Edit pronunciation** lets you change the spelling
for that voice, regenerate, and listen again as often as needed. Approval resets
after each edit; other voices retain their own spelling. The accepted per-voice
text is used for all its positive examples and saved in their provenance.
Incorrect voices can be skipped; no approval is
inferred from an automatic transcript. Generation otherwise uses
`omaspeak say --no-play --out ...` and is silent.

Each approved voice supplies three positives at 0.9×, 1.0×, and 1.1× speed, and
three negatives at normal speed. Generated clips must pass the live segmentation
check. Synthetic examples are added **only to training**; their generator,
voice, text, and speed are retained in the dataset. Calibration and validation
remain human recordings, and the loader rejects synthetic provenance in either
of those splits. The existing acceptance checks are unchanged. Synthetic audio
is an experimental supplement, not a guarantee of improved accuracy.

Temperature is not exposed by the current Omaspeak synthesis interface, so this
flow does not pretend to set it. Supertonic's native seed setting is a different
control; this integration does not change the user's TTS configuration.

With recording retention selected, Omawake saves a pre-augmentation checkpoint before
starting optional synthesis, then saves the combined dataset before fitting.
If assistance fails, the user can continue with human-only training. Cancelling
leaves the active configuration unchanged and preserves explicit checkpoints.

### Resume a retained dataset

```sh
omawake word onboard --dataset /path/to/manifest.json
```

For an existing word, simply run `omawake word onboard WORD`. If labeled datasets
were saved, **Add positive and negative examples** offers a session picker; no
manifest path is required. Choose **Add my positive and negative examples** to
append four new positives and four new negatives to the training split, with
eight additional fresh human clips reserved for calibration and validation.
The original session stays intact, and the expanded dataset is retained in a new
session before fitting. Repeat later to add another batch (128 training clips
maximum). The active detector changes only after validation and Apply.

With an explicit manifest, choose or create the intended word, then choose
whether to add human examples, Omaspeak examples, or only refresh evaluation.
The training split is reused without editing the source session. Since previous
validation results have already been inspected, onboarding collects **eight new
human clips**: four wake-phrase and four other-speech recordings, divided evenly
between fresh calibration and fresh validation. This avoids declaring success
merely by tuning to previously examined held-out clips. Existing training
provenance remains attached to every synthetic example.

### File-based training

For scripted experiments, prepare a JSON dataset with three independent splits. Each split needs at least
two positive and two negative recordings; practical enrollment should include
many more speakers, distances, speaking styles and confusable phrases. Every
clip must contain one natural speech utterance. File paths are relative to the
manifest. Example structure (expand each list to the required sample counts):

```json
{
  "training": [
    {"audio": "train-positive.wav", "positive": true},
    {"audio": "train-confusable.wav", "positive": false}
  ],
  "calibration": [
    {"audio": "cal-positive.wav", "positive": true},
    {"audio": "cal-confusable.wav", "positive": false}
  ],
  "validation": [
    {"audio": "heldout-positive.wav", "positive": true},
    {"audio": "heldout-confusable.wav", "positive": false}
  ]
}
```

```sh
omawake word train agent --engine intel --dataset dataset.json --json
omawake word train agent --engine intel --dataset dataset.json --apply --keep-recordings
```

Training and live detection use the same Silero endpoint and pre/post-roll.
Fingerprinting normalized input audio prevents exact recordings from appearing
in multiple splits. Threshold selection sees only calibration recordings;
validation is a separate final check. A failure leaves the current config and
active detector intact.

Passing local clips does **not** establish a background false-activation rate or
prove performance for other speakers. Before relying on a trained phrase, test
held-out real speech, near misses, background conversations and long negative
recordings. Do not tune against the held-out set and then describe it as an
independent accuracy measurement.

## Changing engines

Word identity, action, aliases, recording sessions and compiled heads are
separate from backend selection. Applying a trained head pins a named copy of
its encoder profile, so changing the default backend does not reinterpret the
word. Earlier heads remain content-addressed artifacts, indexed by encoder
contract.

To move a word to a different encoder, train a candidate with `--engine NEW`.
Omawake can select the newest complete retained labeled session automatically:

```sh
omawake word train agent --engine NEW --reuse-recordings --json
omawake word train agent --engine NEW --reuse-recordings --apply
```

If recordings were discarded, it asks for a new `--dataset` and leaves the
working word unchanged. You can also supply a retained manifest explicitly. The stored `manifest.json` from a retained
training session is directly reusable. Activation happens only after validation
and a final check that the config has not changed concurrently. If it fails,
the previous word/engine remains selected. There is no silent fallback from a
trained head to transcript matching.

## Engine profiles

The top-level `[backend]` and `[model]` are the default profile. A named profile
has the same fields under `[engines.NAME.backend]` and `[engines.NAME.model]`.
Use the paths/model selected by setup when defining a named profile; model files
are shared, not downloaded again. A word's optional `engine = "NAME"` chooses
that profile. Omitting `engine` chooses the default. Onboarding presents the
configured profiles as a terminal selector.

Training against the current default needs no manual profile creation: omit
`--engine`, and successful activation pins a named snapshot automatically.
At most eight active transcript/trained engine groups are allowed. Every group
has one persistent owner thread and a bounded mailbox; groups receive the same
PCM buffer before the caller waits for results. Detection and daemon JSON report
each group's runtime separately. A failed group is an explicit error, never an
empty success that silently disables its words.

The feature branch has a real CPU file-based proof and a small
[CPU/iGPU/NPU parity check](../validation/ENROLLMENT-ACCELERATOR-PROOF.md). Broader
accuracy qualification for this new encoder/head pipeline remains separate work;
the existing transcription-provider proofs do not establish trained-head parity.

### Managed enrollment profiles and runtime inference

Applying a trained word creates an `enrolled-WORD-HASH` profile that preserves
its backend and model settings. Engine pickers label these **Managed enrollment
profile: WORD**. They are configuration snapshots, not extra model downloads or
services. Changing the default profile does not change an enrolled word's model.
Successful retraining removes unreferenced generated profiles for that word in
the same config update. Shared profiles (including references from disabled
words), manually modified profiles, and user-named profiles are preserved.
Saved recordings and head artifacts are not deleted by profile cleanup.

Live trained detection uses VAD, Whisper's frozen encoder, and the trained phrase
classifier. It does not run Whisper's text decoder or transcribe the utterance.
Other words configured for transcript matching still use their own transcription
path. Accelerator compatibility and classifier accuracy must be checked for this
encoder pipeline separately from the regular Whisper transcription backend.

### Separate training and deployment devices

For guided collection, device flags skip the corresponding device questions:

```sh
omawake word onboard WORD --training-device cpu --run-device npu
```

For a saved dataset, without collecting new clips:

```sh
omawake word train WORD --reuse-recordings --training-device cpu --run-device npu --apply
```

Both flags accept `cpu`, `gpu` (`igpu` alias), or `npu`. They choose devices within
the existing OpenVINO frozen-encoder pipeline, not arbitrary runtime libraries.
The optional `--engine` flag chooses source model/runtime assets; it does not
need to name a profile for the desired destination device. Low-level training
defaults to CPU feature extraction and the source profile's deployment device.

Training examples are encoded on the training device. Calibration and held-out
examples are encoded on the deployment device, with the same encoder contract.
Classifier fitting runs on CPU; its threshold is calibrated for deployment.
The unchanged held-out gate must pass there before Apply. Only the deployment
settings are pinned in the managed enrollment profile. Failure to load the
selected device, incompatible encoder contracts, or failed held-out validation
leave the current detector unchanged. Guided onboarding loads both devices before
requesting recordings, also preparing the deployment encoder's compilation cache.

### Thresholds and reviewed detection history

`word threshold WORD` reports calibrated and effective thresholds for each bound
head. `word threshold WORD VALUE` sets a per-word override in `(0, 1]`;
`word threshold WORD auto` removes it. These config changes reload the active
daemon. An override applies to all heads bound to that word and survives
retraining; retraining must pass deployment validation at that override too.
The saved artifact retains its calibrated threshold so `auto` remains useful.
Scores are classifier outputs, not calibrated probabilities. Raising the
threshold trades fewer activations for more missed wake phrases.

History is disabled by default and currently supports trained words only:

```sh
omawake word history WORD enable --max-events 100
omawake word history WORD list --json
omawake word history WORD play EVENT_ID
omawake word history WORD label EVENT_ID false-positive
omawake word history WORD label EVENT_ID true-positive
omawake word history WORD label EVENT_ID unreviewed
omawake word history WORD disable
omawake word history WORD clear
```

Only the live daemon records detected VAD utterances; file tests, benchmarks,
evaluation, and training do not populate history. There is no continuous ambient
recording. Clips are mono 16 kHz PCM WAVs, capped at 30 seconds, with event time,
score, effective threshold, encoder contract, head artifact, and device metadata.
They are stored beneath `$XDG_DATA_HOME/omawake/history/WORD` (normally
`~/.local/share/omawake/history/WORD`) with private directories (0700) and files
(0600). The default limit is 100 events per word, configurable from 1 to 1000.
Oldest events, **including labeled events**, expire when new events reach the
limit. Disable stops new capture without deleting existing evidence; clear
removes history and labels, but not copies in retained enrollment datasets.
Playback is explicit. History I/O errors are reported without stopping detection.
Accepted trained detections log their scores and thresholds even with audio
history disabled. Detection text is the configured phrase, not a transcription
of the triggering speech. History cannot recover audio from earlier activations
or show missed phrases that did not trigger.

A false-positive label adds a negative example; a true-positive label adds a
positive example. Unreviewed clips are ignored. Labels never modify a running
head. Guided `word onboard WORD` offers reviewed history, collects fresh human
calibration/validation clips, and asks about retaining the resulting dataset.
Labeled audio is deduplicated against training examples, and changing an event's
label updates its training label on the next merge. Any overlap with held-out
or calibration audio is rejected. Only the training split receives feedback.

For an explicit low-level candidate using saved evaluation clips:

```sh
omawake word train WORD --reuse-recordings --feedback --training-device cpu --run-device npu
# Add --apply only when ready to activate a candidate that passes validation.
```

This reuses the old evaluation split; repeated success is not new independent
accuracy evidence. Prefer guided collection of fresh held-out examples for
feedback-driven changes. The training set remains bounded at 128 examples;
reaching the limit requests curation instead of silently dropping examples.
Local validation is still a small acceptance gate, not an ambient false-activation
rate measurement. Feedback supports reviewed incremental retraining, not
unsupervised online updates to the active detector.
