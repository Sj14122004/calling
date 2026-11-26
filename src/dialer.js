
'use strict';

// Places calls with a concurrency cap (CONCURRENCY).

const cfg = require('./config');
const store = require('./store');
const { placeCall } = require('./edesy');

const STUCK_MS = 3 * 60 * 1000;
const DISPATCH_DELAY_MS = 100;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

let running = false;

function activeCount() {
  let count = 0;

  for (const lead of store.list()) {
    if (lead.status !== 'calling') continue;

    // Safety net for calls that never receive an outcome.
    if (lead.calledAt && Date.now() - lead.calledAt > STUCK_MS) {
      console.warn(`[Dialer] Call stuck: ${lead.phone}. Marking no_response`);

      store.update(lead.id, {
        status: 'no_response',
      });
    } else {
      count++;
    }
  }

  return count;
}

async function dispatchCall(lead) {
  const dispatchStartedAt = Date.now();

  console.log(`[Dialer] Dispatching call to ${lead.phone}`);

  try {
    const call = await placeCall({
      to: lead.phone,
      referenceId: String(lead.id),
    });

    store.update(lead.id, {
      callSid: call.call_sid || call.sid || call.id || null,
    });

    console.log(
      `[Dialer] Dispatch completed for ${lead.phone} in ${
        Date.now() - dispatchStartedAt
      }ms`
    );
  } catch (error) {
    store.update(lead.id, {
      status: 'failed',
      error: error.message,
    });

    console.error(`[Dialer] Failed for ${lead.phone}:`, error.message);
  }
}

async function start() {
  if (running) {
    console.warn('[Dialer] Already running');
    return false;
  }

  running = true;

  console.log('[Dialer] Queue started');
  console.log(`[Dialer] Concurrency limit: ${cfg.concurrency}`);

  try {
    while (true) {
      const active = activeCount();
      const next = store.list().find((lead) => lead.status === 'pending');

      // Queue finished when no pending or active calls remain.
      if (!next && active === 0) {
        console.log('[Dialer] Queue finished');
        break;
      }

      // Wait if queue is empty or concurrency is full.
      if (!next || active >= cfg.concurrency) {
        await sleep(250);
        continue;
      }

      // Mark before dispatch to prevent duplicate calls.
      store.update(next.id, {
        status: 'calling',
        calledAt: Date.now(),
        error: null,
      });

      // Fire the call request without blocking the queue loop.
      dispatchCall(next);

      await sleep(DISPATCH_DELAY_MS);
    }
  } finally {
    running = false;
    console.log('[Dialer] Queue stopped');
  }

  return true;
}

module.exports = { start };
