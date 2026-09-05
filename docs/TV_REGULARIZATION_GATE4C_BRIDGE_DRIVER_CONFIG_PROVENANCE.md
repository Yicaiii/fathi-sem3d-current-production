# Gate 4C bridge driver-config provenance fix

The certified exact reverse uses a frozen historical external-driver runtime
config from the derived certified reference. The regularized production
wrapper uses a different CURRENT runtime config.

The corrected-gradient bridge previously treated these two config files as
"equivalent paths" and required their file SHA256 values to be identical.
That requirement is invalid: the configs intentionally have different
provenance and different bytes.

The bridge now validates them separately:

- `input_hashes.runtime_config` must equal the CURRENT regularized runtime
  config exactly.
- `input_hashes.driver_assets.config` must equal the frozen historical driver
  config named by `certified_reference.immutable_input_assets.runtime_config`
  and must match its certified SHA256.

No reverse output, gradient array, TV mathematics, Mtilde solve, optimizer, or
SEM3D execution is changed.
