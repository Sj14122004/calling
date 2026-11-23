
// Places calls with a cap on simultaneous calls (CONCURRENCY).
const cfg = require('./config');
const store = require('./store');
const { placeCall } = require('./edesy');

const STUCK_MS = 3 * 60 * 1000; // Safety net if a call never receives an outcome
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

let running = false;

function activeCount() {
  let count = 0;

  for (const lead of store.list()) {
    if (lead.status !== 'calling') continue;

    if (lead.calledAt && Date.now() - lead.calledAt > STUCK_MS) {
      store.update(lead.id, { status: 'no_response' });
    } else {
      count++;
    }
  }

  return count;
}

async function start() {
  if (running) return false;

  running = true;

  try {
    while (true) {
      const active = activeCount();
      const next = store.list().find((lead) => lead.status === 'pending');

      // No pending leads and no active calls: queue is finished.
      if (!next && active === 0) break;

      // Wait if there are no pending leads, or the concurrency limit is reached.
      if (!next || active >= cfg.concurrency) {
        await sleep(500);
        continue;
      }

      // Mark before sending the request to avoid duplicate dispatch.
      store.update(next.id, {
        status: 'calling',
        calledAt: Date.now(),
        error: null,
      });

      placeCall({
        to: next.phone,
        referenceId: String(next.id),
      })
        .then((call) => {
          store.update(next.id, {
            callSid: call.call_sid || call.sid || call.id || null,
          });
        })
        .catch((error) => {
          store.update(next.id, {
            status: 'failed',
            error: error.message,
          });
        });

      // Give the loop a little breathing room after dispatching a call.
      await sleep(200);
    }
  } finally {
    running = false;
  }

  return true;
}

module.exports = { start };
