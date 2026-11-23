
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

  const res = await fetch(edesy.base, {
    method: 'POST',
    headers: {
      'X-API-Key': edesy.apiKey,
      'Content-Type': 'application/json',
      'Idempotency-Key': idempotencyKey,
    },
    body: JSON.stringify(requestBody),
  });

  const rawBody = await res.text();

  let json = {};

  try {
    json = rawBody ? JSON.parse(rawBody) : {};
  } catch {
    json = { rawResponse: rawBody };
  }

  if (!res.ok) {
    const details = {
      httpStatus: res.status,
      message: json.message || json.error || 'Edesy call request failed',
      details: json.details || json.errors || null,
    };

    console.error('Edesy API rejected call:', details);
    throw new Error(JSON.stringify(details));
  }

  return json.data || json;
}

module.exports = { placeCall };
