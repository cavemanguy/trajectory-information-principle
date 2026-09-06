# R9-H1 Artifact Schema

Each seed artifact must contain a single JSON file named `family_result.json` with:

- `seed`
- `protocol_version`
- `architectures`
- `anomaly_metrics`
- `controls`
- `validity`
- `target_leak_audit`

The aggregate classifier consumes only this frozen schema.
