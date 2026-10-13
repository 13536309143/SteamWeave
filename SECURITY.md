# Credentials and deployment

Store provider keys in local environment variables or `.env`; neither is served to the frontend. Provider adapters read only the selected provider's key. The public assets contain structured simulated-report features and pretrained lexical vectors, not authentication material.

The local workbench binds to `127.0.0.1:8787`. Cross-origin mutations are rejected. Public remote hosting requires an authentication layer and request limits; the included localhost configuration is the tested deployment.

Before pushing, run `python scripts/check_publication.py --tracked`. If a credential is exposed, revoke it at the issuing provider and replace the local configuration. Do not put the replacement in an issue.
