# Local service and publication boundary

This Python service is designed for a trusted local environment. It is **not hardened for direct internet exposure**: it has no built-in authentication or TLS, serves history/export endpoints, permits cross-origin API access, and includes a refresh endpoint that triggers an upstream fetch. Do not forward its port from a router or expose it through a public tunnel without a separate security design.

The public package binds to `127.0.0.1:8765` by default. For an intentionally shared, trusted LAN, an operator can set `BUKIT_KIARA_HOST=0.0.0.0` and apply appropriately scoped firewall rules. That exposes the service to devices able to reach the listener; it is not an access-control mechanism. This repository does not change those settings for you.

Use one collector/server instance per local database. Keep backups of data you care about, and allow the host to stay awake if uninterrupted collection matters. A process restart or machine sleep may produce a real gap; never fill it with invented sensor readings.

Do not commit databases, logs, exports, environment files, credentials, reviewer bundles or personal screenshots. `.gitignore` reduces accidents but is not a secret scanner. Review the actual staged file list before every push.

Do not put secrets or private records in a public issue. For a report that can be shared safely, use a synthetic example and omit credentials, private endpoints and personal data. No response-time commitment or security certification is implied.
