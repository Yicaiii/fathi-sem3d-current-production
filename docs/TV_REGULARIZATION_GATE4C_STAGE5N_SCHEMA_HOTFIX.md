# Gate 4C Stage5N receiver-schema hotfix

The certified dt-contract repair passed its tests but the first
`--certify-existing` attempt stopped before recertification because it assumed
the Stage5N bundle stored the certified current receiver specifically at
`objective.current`.

The Stage5N bundle is historical certification provenance, so Gate 4C must not
rewrite it or depend on one undocumented JSON location.

The hotfix resolves string leaves whose basename is
`current_external_receiver.npy`, resolves them through the runtime path layer,
hashes the referenced files, and accepts only a referenced artifact whose
SHA256 equals the already-completed regularized parent receiver SHA256.

This preserves strict bitwise certification and does not lower any tolerance.
No forward, reverse, SEM3D, or optimizer rerun is required.
