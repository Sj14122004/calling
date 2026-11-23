# Edesy Lead Caller

Outbound calling agent, no LLM. Rule-based yes/no flow on the Edesy Voice API.

## Flow
1. "Website design karwana chahte hain?"  No -> removed. Yes -> 2
2. "Website hai?"  No -> LEAD ("boss call karenge"). Yes -> 3
3. "Improvement chahiye?"  Yes -> LEAD. No -> removed.

Caller presses **1 = haan**, **2 = nahi**. Edit script/branching in `src/flow.js`.

## Setup
```
npm install
cp .env.example .env     # fill SID, token, from-number, PUBLIC_BASE_URL
ngrok http 3000          # put the https URL in PUBLIC_BASE_URL
npm start
```
Open `http://localhost:3000/?key=<ADMIN_TOKEN>`, paste numbers, Add, Start calling.

## Prerequisites on Edesy
KYC approved, wallet topped up, an Indian number bought (used as `from`).

## First test
Add only YOUR number. After the call, check `data/debug.log` - it records exactly
what Edesy sent to `/edesy/gather` and `/edesy/status`. The docs do not fully document
those two payloads, so if the keypress or status is not detected, adjust `digitOf()`
or the regex in `src/routes.js` to match the logged field names.

## Structure
```
src/config.js  env        src/edesy.js   Edesy API client
src/flow.js    script     src/dialer.js  concurrency-limited caller
src/store.js   leads DB   src/routes.js  webhooks + dashboard API
public/index.html  dashboard
```
Statuses: pending, calling, lead, removed, no_response, no_answer, failed.
