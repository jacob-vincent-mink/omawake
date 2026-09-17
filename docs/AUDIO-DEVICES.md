## Audio device selection

Run `omawake setup audio` to select System default or a specific device. The Audio
screen offers refresh, an optional test, and Apply. Full setup includes this
selection. Testing needs no model and does not save configuration: Omawake
measures three seconds of microphone levels without recording a file; Omaspeak
plays a short, quiet tone.

```bash
omawake audio-devices --detailed --json
omawake setup audio --device 'pipewire:<node.name>' --test
omawake setup audio --device 'pipewire:<node.name>' --apply
omawake config get audio.device
omawake config unset audio.device
```

Copy a selector from the listing instead of the example placeholder. PipeWire
selectors persist node names, never numeric IDs or display labels. A hardware
or Bluetooth profile change may change the name; refresh and reselect if needed.
Unavailable saved devices stay visible and can be kept.

The selection applies only to this application and never changes the desktop
default. Pinned routes cannot fall back, reconnect, or move to another device.

Omawake retains CPAL for System default and legacy device names. Its original
`audio-devices --json` array remains compatible; `--detailed` adds structured
records and physical PipeWire sources. Duplicate legacy names are rejected.
Default is resolved when capture opens; routing of an already-open default
stream depends on the audio server. On capture failure, the daemon reports
`audio_unavailable`, discards partial recognition state, and retries one second
after each failed open. Controls remain available between bounded open attempts.

Omaspeak resolves its pinned output before each playback. A disconnect fails
the current request; a later request can succeed after reconnection. Pinned
playback times out after the WAV duration plus five seconds and preserves
cancellation and child cleanup. System default retains the existing pw-play /
aplay behavior. File-only synthesis, no-play requests, and benchmarks do not
require a speaker. There are no per-request route overrides.

Apply and config set/unset preserve the existing transaction: save and restart
an already-active systemd service for this configuration, restoring the previous
configuration if restart fails. They do not start inactive services. Manually
launched daemons need manual restart. Status JSON reports requested and saved
selections and whether they differ; an unknown effective physical route is null.

Pinned routing requires pw-dump and pw-record (input) or pw-play (output), plus
a running PipeWire session and session manager. On Arch these tools come from
pipewire and pipewire-audio. Default routing retains its existing dependencies.
No Rust dependencies were added.

Config schemas expose audio.device as an enum with labels, availability,
discovery errors, and a choices_command for refresh. Discovery never starts recording or plays audio. Settings still load when discovery is unavailable.
