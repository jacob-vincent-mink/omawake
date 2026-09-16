# Audio device selection plan

Status: implemented in these worktrees; validation is recorded below.

## Scope and branches

Add microphone selection to Omawake and playback-device selection to Omaspeak.
Each application owns its route; neither changes the system default.
Omawake does not need a speaker setting and Omaspeak does not need a microphone setting.

Worktrees created from local main:
- omawake-audio-devices: plan/audio-device-selection at af25056.
- omaspeak-audio-devices: plan/audio-device-selection at 86064f6.

## Findings from main

- Omawake already persists audio.device (default: "default"), exposes it through
  config get/set/unset/schema, and lists CPAL input names via devices.
  src/audio.rs selects the first case-insensitive name match; src/audio/cpal.rs
  owns enumeration and opening. This is not yet a robust device picker.
- src/app.rs uses that selection for bounded live testing and daemon capture.
  The daemon reopens capture between armed cycles, retains its startup config,
  and currently propagates capture failures out of the daemon.
- Omaspeak has no audio config section or output enumeration. src/main.rs play()
  tries pw-play and then aplay on the default route. Explicit targeting must
  prevent this fallback from accidentally playing through a different output.
- Omaspeak routes synthesis playback through its request handler, including
  local execution when no daemon handles the request. Output-file generation
  and no_play must remain independent of playback-device availability.
- Both applications have setup wizards, machine-readable configuration schemas,
  status JSON, protocol types, example configuration, and CLI/unit tests.
  Neither application repository contains QML. No Omawake/Omaspeak references
  were found in the local Omarchy source checkout; an external settings consumer
  must be located before promising a shell UI patch.

## 1. Establish the device contract and verify Linux routing

First implementation gate: prove enumeration and explicit opening against real
PipeWire/ALSA devices before wiring selectors into setup.

- Offer "System default" plus discovered devices. Keep backend identifiers
  separate from user-facing labels; never persist transient list positions or
  PipeWire numeric object IDs.
- Retain Omawake's audio.device key and existing name-based configurations.
  Detect ambiguous legacy names instead of silently choosing the first.
- Add Omaspeak audio.device with a default of "default", giving both applications
  the same configuration shape and direction-specific labels.
- Prefer retaining Omawake's CPAL capture adapter and Omaspeak's external player
  approach if they can enumerate and address the advertised devices reliably.
  Verify whether CPAL exposes physical microphones on the supported desktop:
  a list containing only an ALSA default/pulse bridge is insufficient.
- For Omaspeak, establish stable PipeWire target names and an explicit,
  distinguishable ALSA identifier namespace if ALSA selection is supported.
  Use structured process arguments. Never translate a PipeWire target into an
  ALSA target by guessing. Default playback may retain the existing fallback;
  explicitly pinned playback must not fall back to another route.
- If CPAL cannot address the advertised microphones, expand the capture adapter
  to a PipeWire-capable implementation before continuing. Record that decision,
  dependency changes, and identity limitations in the plan.
- Define a shared record shape: selector, label, direction, backend, default
  marker, and availability. Distinguish the configured selector from the route
  actually known to be open; do not invent a resolved device for an opaque
  system-default bridge.

Deliverable: documented selector grammar and executable enumeration/opening proof.

## 2. Implement discovery, configuration, and CLI

Omawake: src/audio.rs, src/audio/cpal.rs, src/config.rs, src/app.rs.
Omaspeak: new src/audio.rs module, src/lib.rs, src/config.rs, src/main.rs.

- Add structured discovery for pickers, including refresh and an explicit
  unavailable entry for a saved device that is absent.
- Preserve Omawake's existing devices and devices --json output contract;
  add an explicit detailed mode for device records. Provide the corresponding
  output-device listing in Omaspeak.
- Wire defaults, parsing, serialization, get/set/unset, and schema metadata.
  Separate invalid selector syntax from temporarily disconnected hardware:
  saving a valid offline device remains possible.
- Expose dynamic choices without making schema retrieval fail when the audio
  server is unavailable. Avoid opening a microphone just to build the schema.
- Keep backend.device (CPU/GPU/NPU inference) entirely separate from audio.device.
- Do not introduce per-request routing overrides in the initial feature.

## 3. Route audio consistently and define lifecycle behavior

Omawake:
- Apply the same resolver to daemon capture and bounded live tests.
- On device loss, close capture, discard partial recognition state, and enter
  an observable audio-unavailable state. Retry with bounded backoff while
  continuing to process pause/resume/shutdown.
- A pinned microphone reconnects only to that microphone. System default is
  re-resolved when capture reopens; verify whether the audio server moves an
  already-open default stream and document the actual behavior.

Omaspeak:
- Move player selection, target arguments, and errors into a testable audio
  adapter. Pass the configured route through both daemon and local playback.
- Resolve the route at each new playback. A lost pinned speaker fails that
  request with a useful error; a later request can succeed after reconnection.
- Preserve cancellation, child cleanup, and daemon shutdown behavior.
- Do not replay an utterance automatically after a partial playback failure.
- File-only synthesis, benchmarks, and no_play must work without audio hardware.

Initial apply policy:
- Mark audio.device as restart-required, matching the existing Omawake schema.
- Plain config writes save the choice and report that an active daemon needs
  restart. Setup Apply restarts an already-active service using its existing
  transaction/rollback mechanism; it does not start an inactive service.
- A running process continues using its loaded configuration until restart.
  Report saved versus active selection so a UI cannot imply the change is live.
- Audio-only setup must not re-probe inference providers or rebuild model caches.

## 4. Add setup selection and meaningful verification

Touch both src/setup/wizard.rs implementations, setup reporting, and the
application-level setup dispatch/Apply code.

- Add an Audio entry to setup and a direction-appropriate selector to onboarding.
- Show System default, readable labels, the current selection, and unavailable
  saved devices. Provide refresh and preserve configuration on cancellation.
- Include the selected route in the final Apply summary.
- Provide an explicit microphone level/capture check and an output test sound.
  Run these only when requested; enumeration must not record or play audio.
- Make output testing independent of model installation, so it checks the
  speaker rather than inference setup.
- Keep terminal pickers and future desktop pickers on the same discovery and
  configuration contract.

## 5. Status, desktop consumers, and packaging

Touch Omawake src/app.rs status paths; Omaspeak src/main.rs and src/protocol.rs.

- Report requested route, known effective route, availability/error, and
  restart-needed state consistently while running, paused, unavailable, or stopped.
- Make protocol additions backward-compatible and test old payload decoding.
  Keep audio errors separate from model/provider readiness.
- Locate the active desktop settings consumer and audit its schema rendering,
  device refresh, saved/active display, and service Apply behavior. If a separate
  repository needs changes, create its own worktree from main at that point.
- Audit both repositories' packaging/install scripts and CI dependencies after
  the routing proof determines the required discovery/player tools.
- Update config.example.toml, README.md, INSTALL.md, and CHANGELOG.md with
  selection, default-route behavior, restart requirements, and troubleshooting.

## 6. Validate and deliver

Extend existing tests rather than requiring real audio hardware in CI:
- Defaults and old config compatibility; set/unset/schema round trips.
- Detailed discovery, duplicate labels, invalid identifiers, disconnected saved
  devices, and audio-server enumeration failures.
- Explicit device reaches the capture/player adapter in every live audio path.
- Pinned output never falls back to a different route; safe argument handling;
  playback cancellation and cleanup; file-only synthesis bypasses discovery.
- Omawake loss/reconnect with fresh detector state and responsive controls.
- Running/stopped status, protocol compatibility, setup cancellation,
  audio-only Apply, active-service restart, and transaction failure/rollback.

Manual acceptance on the desktop:
- Select between two microphones and prove only the chosen one triggers Omawake.
- Select speakers/headphones and prove speech/test sound uses the chosen output.
- Change system defaults: default selections follow the documented lifecycle;
  pinned selections stay pinned.
- Unplug/replug devices, restart the audio server, and test a Bluetooth profile
  change where available. Observe explicit errors and recovery without rerouting.
- Save while daemons are running, Apply/restart, and confirm active status agrees.
- Verify no device listing changes the system-wide default.

Run each repository's required format/lint/test checks and targeted hardware
smokes. Deliver coordinated application changes, any confirmed consumer change,
and documentation together.

## Suggested implementation order

1. Device identity and Linux routing proof.
2. Discovery/configuration/CLI contracts in both apps.
3. Capture/playback routing, recovery, and status.
4. Setup pickers and route checks.
5. Confirmed desktop integration, packaging, documentation, and acceptance tests.


## Implementation decisions

- Pinned routes use pipewire:<node.name>. Default/legacy input remains CPAL;
  explicit ALSA output selection is not added.
- Config writes already restart active services transactionally on main. That
  established behavior is preserved instead of the proposed save-only policy.
- Dynamic choices are included directly in config schemas plus a refresh command.
  Local main checkouts of Omarchy and omarchy-pkgs contain no application settings
  consumer/package definitions to update. Installed desktop config is untouched.
- Runtime dependencies are documented; CI uses fake PipeWire tools and no audio
  hardware. WirePlumber routing policy reference:
  https://pipewire.pages.freedesktop.org/wireplumber/policies/linking.html

## Validation completed

- All-target test suite: 259 passed, none failed.
- Required line-coverage gate (90.01%): 90.40%.
- cargo fmt --check, strict all-target clippy, cargo deny licenses, and
  cargo about generation passed. Updated workflow YAML parses successfully.
- Device fixtures cover duplicate display names, missing/malformed inventory,
  offline configuration, explicit targeting, and absence of pinned fallback.
- Omawake additionally exercises a real terminal's test/apply/cancel flow and
  a live daemon's missing microphone, pause/resume, loss/reconnect, saved/active
  disagreement, and shutdown with an isolated native-provider fixture.
- Omaspeak exercises playback timeout/cleanup, legacy status decoding, and
  file-only synthesis with an invalid audio selector.
- Desktop smoke: both built CLIs discovered the physical input/output nodes;
  Omawake captured 47,104 microphone samples in its three-second level check
  (RMS 0.0023, peak 0.0090), and Omaspeak's selected laptop-speaker test exited
  successfully. No device preference or system default was changed.
- Hardware unplug/replug, multiple physical microphones, and Bluetooth profile
  changes were not exercised on this desktop; disconnect recovery was simulated.
- Release archives now include docs/AUDIO-DEVICES.md, linked from README/INSTALL.
