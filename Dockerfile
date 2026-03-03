FROM oven/bun:1.2.20 AS build
WORKDIR /app

COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

COPY . .
RUN bun run build

FROM oven/bun:1.2.20
WORKDIR /app

ENV PORT=80
ENV INFISICAL_ENV=dev
ENV INFISICAL_SECRET_PATH=/

COPY --from=build /app/dist ./dist
COPY scripts/server.mjs ./scripts/server.mjs

EXPOSE 80

CMD ["sh", "-lc", "if [ -n \"$INFISICAL_TOKEN\" ] && [ -n \"$INFISICAL_PROJECT_ID\" ]; then bunx infisical run --token \"$INFISICAL_TOKEN\" --projectId \"$INFISICAL_PROJECT_ID\" --env \"$INFISICAL_ENV\" --path \"$INFISICAL_SECRET_PATH\" -- bun ./scripts/server.mjs; else bun ./scripts/server.mjs; fi"]
