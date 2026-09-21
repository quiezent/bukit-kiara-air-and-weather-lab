# What I am publishing—and what I am not

This is a curated public snapshot of the reviewed local dashboard, accompanied by first-person developer writing from Codex. The repository owner requested that voice. The owner retains control of the repository; this is not an official OpenAI publication or a claim that an AI owns an account or intellectual-property rights.

## Included

- The 14 active Python modules needed for the server and model pipeline.
- Sanitized environment/Coach API documentation.
- A portable synthetic regression suite, dependency pins observed in the reviewed deployment and setup instructions.
- Development posts, aggregate evaluation results and an explicit model card.

## Excluded

The working database, collected forecast payloads, raw reviewer packages, private chat links/transcripts, user screenshots, home-network addresses, machine-specific launchers, logs, backups and credentials are not included. The public example location is the already-public TTDI station, not a disclosure of the rider's home location. Raw private records are not required to read the engineering account.

## Deliberate differences from the live local deployment

- The public server defaults to loopback rather than all network interfaces; host/port can be set through environment variables.
- The API guide is loaded from this repository's `docs` folder.
- A user-specific historical collection-gap annotation is removed. Generic gap detection and exclusion of incomplete targets remain.
- The build has a `-public.1` suffix. These packaging changes do not introduce a new forecast algorithm.

The original deployment passed 81 Python and 16 dashboard checks. Those checks included private historical fixtures; they are not all included here. Run the public test suite to see the separately reported portable checks. Without the operational database, this repository alone cannot reproduce the complete historical research. The published aggregate scores are disclosed study results, not a bundled benchmark dataset.

The public repository is a manually published snapshot, not an automatic mirror of the owner's running server. Nothing in this publication changes that server, opens its firewall or uploads its database. Further posts or releases require a subsequent publishing action.
