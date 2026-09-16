# Provider discovery and installation boundaries

This implements the family-discovery and installer-protection slice of W04/W05.
The required audio.cpp families are `moonshine_asr` and `silero_vad`.

The runtime probe now queries the pinned audio.cpp registry before treating a
provider as usable for the application's default. An ABI-compatible library
that omitted a required family is rejected with the missing family name and
its registered alternatives. Listing the bundled catalog remains offline;
registry discovery runs only in the existing native runtime-probe path.
Registration is not a claim about device placement, inference or output quality.

Installers take a nonblocking per-profile filesystem lock before verifying an
existing installation or touching its downloads/staging directory. A second
writer gets a clear busy error and can retry. Different profiles retain
independent locks. Lock files remain in `models/.locks`: removing the inode on
unlock would allow two simultaneous owners. Locks release on normal completion,
error or process exit, without stale PID-based ownership checks.

Before transferring or staging assets, setup checks filesystem free space for
the complete additional staging copy plus unverified download objects and a
16 MiB metadata reserve. Verified cached files need no new download allocation.
The budgets are combined when models and downloads share a filesystem and
checked separately otherwise. Existing models/caches already consume reported
free space and remain in place while the replacement is prepared. Preflight is
a point-in-time check, not a disk reservation; concurrent unrelated writes can
still exhaust storage and follow the existing failure/rollback path.

SIGINT/Ctrl-C and SIGTERM request cancellation during installation. Hashing,
copying and download loops check between 128 KiB chunks. Network requests have
a 10-second connect timeout and a five-second idle read timeout. Cancellation
cleans this install's staging and partial transfer, preserves verified reusable
cache objects and the active model, and releases the lock. The final atomic
publication/rollback sequence is allowed to finish rather than abandoning a
half-published target. Normal termination signals still work after installation.
Forced termination such as SIGKILL cannot run cleanup; the filesystem lock
still releases, but abandoned staging/partial files may require later cleanup.

Existing license acceptance, pinned hashes, provenance manifests, replacement
rollback and separately fingerprinted compiled caches retain their roles. This
change does not estimate unknown device-cache compilation sizes, implement
range-request download resume, or qualify additional hardware. Cache capacity
estimation and power/quality qualification remain explicit follow-ups.

Regression coverage includes competing installers, lock reuse, insufficient
space and overflow, cancellation cleanup, real signals in disposable processes,
registry enumeration failures and missing required families. Existing corruption,
license and activation tests continue to cover the transaction boundaries.
