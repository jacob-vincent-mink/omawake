# Playback pause ownership (W10)

IPC v1 now accepts `{"protocol":1,"id":"caller-request-id","type":"hold_pause"}`.
The caller keeps this Unix control connection open for the duration of playback.
The response echoes the ID and has `type: "state"`, `state: "paused"` only after
Omawake has dropped capture. A caller must wait for this acknowledgement before
starting audio. This command is additive; older daemons reject it explicitly.

Each connection owns one hold. EOF, reset or extra request bytes release only
that connection's hold. At most 32 holds are accepted; excess requests receive
`busy`. Status includes `details.pause.manual` and `details.pause.owners`.
The existing `pause` and `resume` commands control the independent manual pause.
`resume` reports `paused` while playback holds remain. The last hold disappearing
re-arms capture only when manual pause is clear. Process death and disconnect do
not require a compensating resume command or persistent lease recovery.

The control loop processes at most 32 incoming connections per poll, with bounded
read/write timeouts. Holds are checked each paused poll (normally 50 ms); this is
not a hard real-time release guarantee under system load or malicious slow clients.
Shutdown closes all holds. A playback client must stop if its hold connection
closes, since a restarted daemon cannot inherit the old ownership.

Omaspeak's companion change acquires this hold before ordinary and preview
playback. Its player also inherits the hold descriptor, so killing the parent
cannot release the pause while the audio process remains alive. Playback without
a running Omawake is allowed; an incompatible or unresponsive running daemon
produces an explicit error instead of using an unsafe pause/resume fallback.

Validation uses Unix socket pairs and isolated listeners: capture-state transition
before acknowledgement, nested holds, manual pause, disconnect before/after ack,
owner limits, malformed requests, status and shutdown. No microphone is opened.
The companion CLI test uses a silent fake player. Real-room acoustic playback
qualification remains an open W01/W10 gate.

This delivers the pause-ownership portion of W10. Bounded Omaspeak admission,
prompt synthesis cancellation, request cancellation and queue lifecycle remain
tracked S06 work; passing these tests does not claim that whole milestone complete.
