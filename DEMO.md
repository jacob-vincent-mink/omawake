# Demo

From an unpacked release, run the guided setup and its checks:

```bash
./omawake setup
./omawake setup check
```

Test without opening a microphone:

```bash
./omawake test --audio /path/to/recording.wav --json
./omawake benchmark --warmup 1 --iterations 5 /path/to/recording.wav
```

Add or replace phrase-to-action mappings with direct argument vectors:

```bash
./omawake wake-word add --id computer --phrase "Computer" -- notify-send "Wake phrase heard"
./omawake wake-word remove computer
```

Run the resident process only when desired:

```bash
./omawake daemon
./omawake status --json
./omawake pause
./omawake resume
./omawake stop
```

Ordinary setup does not install a service. Use `omawake setup systemd` as a
separate, explicit step if a persistent user service is wanted.
