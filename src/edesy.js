
'use strict';

const { edesy } = require('./config');

async function placeCall({ to, referenceId, statusCallbackUrl }) {
  const phoneNumber = String(to ?? '').trim();

  if (!phoneNumber) {
    throw new Error('Phone number is required');
  }

  const idempotencyKey =
    referenceId ||
    `lead-call-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

  const requestBody = {
    flow_id: edesy.flowId,
    to: phoneNumber,
  };

  if (referenceId) {
    requestBody.reference_id = referenceId;
  }

  if (statusCallbackUrl) {
    requestBody.status_callback_url = statusCallbackUrl;
  }

  const startedAt = Date.now();

  console.log(`[Dialer] Sending call request for ${phoneNumber}`);

  let res;

  try {
    res = await fetch(edesy.base, {
      method: 'POST',
      headers: {
        'X-API-Key': edesy.apiKey,
        'Content-Type': 'application/json',
        'Idempotency-Key': idempotencyKey,
      },
      body: JSON.stringify(requestBody),
    });
  } catch (error) {
    console.error(
      `[Dialer] Network error after ${Date.now() - startedAt}ms:`,
      error.message
    );
    throw error;
  }

  const rawBody = await res.text();
  const requestDuration = Date.now() - startedAt;

  let json = {};

  try {
    json = rawBody ? JSON.parse(rawBody) : {};
  } catch {
    json = { rawResponse: rawBody };
  }

  console.log(
    `[Dialer] Edesy responded: HTTP ${res.status} in ${requestDuration}ms`
  );

  if (!res.ok) {
    const details = {
      httpStatus: res.status,
      message: json.message || json.error || 'Edesy call request failed',
      details: json.details || json.errors || null,
    };

    console.error('[Dialer] Edesy API rejected call:', details);

    throw new Error(JSON.stringify(details));
  }

  const callData = json.data || json;

  console.log(
    `[Dialer] Call accepted in ${requestDuration}ms`,
    callData.call_sid || callData.sid || callData.id || ''
  );

  return callData;
}

module.exports = { placeCall };
