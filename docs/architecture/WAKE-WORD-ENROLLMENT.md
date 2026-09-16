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

The terminal flow records several examples, shows the transcripts, and asks
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
when the session ends, including cancellation and errors. Imported source files
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

Run `omawake word onboard agent` (or **Teach a wake word** in setup), choose a
configured OpenVINO Whisper base.en engine, and record the phrase as prompted.
After reviewing Whisper spellings, choose **Train this wake phrase**. CPU is the
qualified device for this experimental path so far.

Onboarding checks that the encoder loads before asking for additional samples.
It reuses the initial five positive recordings and asks for five more, then ten
negative recordings. Negatives are **other speech without the wake phrase**:
alternate similar-sounding phrases with ordinary everyday sentences. Use a
different utterance each time and vary pace and distance. Each clip should
contain one spoken utterance; silence and clipped audio are retried.

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

Choosing **Keep recordings locally** also preserves the complete labeled dataset
if calibration or held-out validation rejects the candidate. The error reports
the manifest path and held-out miss/false-activation counts or calibration
failure. A held-out candidate also needs every score at least 0.025 away from
the threshold; this experimental margin gate can reject a candidate even with
zero classification errors. Replaying the same dataset is a diagnostic, not a
way to improve it or an independent validation.

On successful activation, **Keep recordings locally** retains the dataset for
`word train agent --reuse-recordings`. Otherwise the temporary WAVs and generated
manifest are removed. No dataset editing, audio playback, or wake action is
needed during guided onboarding.

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

The feature branch has a real CPU file-based proof. GPU/NPU placement and
accuracy qualification for this new encoder/head pipeline remain separate work;
the existing transcription-provider proofs do not establish trained-head parity.
