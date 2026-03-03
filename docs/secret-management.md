# Secret Management (Infisical)

## Current state

This deployment is a static web app and does **not** require runtime secrets.

For security, the Coolify app has **no environment variables** configured.

## Policy

- Secrets live in Infisical, not `.env` files in the repo.
- If this app later needs secrets, source them from Infisical during deployment automation.
- Avoid putting sensitive values into Coolify env vars for Dockerfile builds because Coolify can expose env values as build args/logs.

## Recommended future pattern

1. Keep secret values in Infisical.
2. Use a deploy pipeline that retrieves required secrets from Infisical at deploy time.
3. Prefer deploy modes that do not inject secrets into Docker build arguments.
4. Rotate/revoke any temporary service tokens used during setup.
