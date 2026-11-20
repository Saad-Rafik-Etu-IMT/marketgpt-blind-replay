# Training infrastructure

The local API has one generalized training subprocess. Opt-in
`POST /api/experiments?enqueue=true` persists a waiting job even when busy;
the supervisor dispatches the next queued job when the process exits. If the
application restarts, use Dispatch waiting queue or POST `/queue/dispatch`.
The original non-enqueue API retains its conflict behavior. The process-local
supervisor is not a multi-process scheduler.

Spec fingerprints lock inputs/resources/search parameters. Search candidates
are deterministic: shared candidate IDs, seeds and feature profiles; bounded
grid or random sampling; a total cap of 60 runs. Cost estimation is a rough
CPU operation estimate. Each epoch publishes measured losses to a progress
artifact. Cancellation and time budget are checked at epoch boundaries.
Resume skips verified completed reports, retrains incomplete/failed runs from
the beginning and refuses modified development data. It does not serialize an
in-flight optimizer or resume at the exact interrupted epoch.

NumPy trains on CPU. The optional reference Transformer supports CPU, auto,
or explicitly requested CUDA, with resolved device recorded. The existing
checkpoint integrity checks apply; no accelerator availability is claimed.
The generalized CLI runs on a remote host when the repo, Python environment
and data are available; the legacy SSH wrapper is not a distributed scheduler
for the generalized campaign format.

Optional automatic evaluation generates inner-validation reports after
training. Historical validation remains an explicit recorded developmental
read. Protected evaluation requires a separate CLI and frozen authorization ID.
