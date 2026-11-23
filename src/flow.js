
'use strict';

// ==========================================
// Edesy Lead Caller - Hindi Audio Flow
// Voice: Mithi v2 Flash (Ahana)
// Audio: Pre-recorded WAV files
// ==========================================

const AUDIO_FILES = {
  q1: 'q1.wav',
  q2: 'q2.wav',
  q3: 'q3.wav',

  lead: 'lead.wav',
  removed: 'removed.wav',
  no_response: 'no-response.wav',
};

// ==========================================
// Call branching logic
// 1 = YES
// 2 = NO
// ==========================================

const STEPS = {
  q1: {
    yes: { next: 'q2' },
    no: { end: 'removed' },
  },

  q2: {
    yes: { next: 'q3' },
    no: { end: 'lead' },
  },

  q3: {
    yes: { end: 'lead' },
    no: { end: 'removed' },
  },
};

// ==========================================
// Resolve public audio URL
// ==========================================

function getBaseUrl(base) {
  const url = (
    base ||
    process.env.PUBLIC_BASE_URL ||
    ''
  ).trim().replace(/\/+$/, '');

  if (!url) {
    throw new Error('PUBLIC_BASE_URL is missing');
  }

  if (!url.startsWith('https://')) {
    throw new Error('PUBLIC_BASE_URL must use HTTPS');
  }

  return url;
}

// ==========================================
// Create prerecorded audio instruction
// ==========================================

function play(base, key) {
  const filename = AUDIO_FILES[key];

  if (!filename) {
    throw new Error(`Unknown audio key: ${key}`);
  }

  const publicBase = getBaseUrl(base);

  return {
    verb: 'play',
    url: `${publicBase}/audio/${filename}`,
  };
}

// ==========================================
// Finish the call
// ==========================================

function finish(outcome, base) {
  const validOutcomes = [
    'lead',
    'removed',
    'no_response',
  ];

  if (!validOutcomes.includes(outcome)) {
    outcome = 'no_response';
  }

  return {
    instructions: [
      play(base, outcome),
      {
        verb: 'hangup',
      },
    ],
  };
}

// ==========================================
// Ask a question and collect keypad input
// ==========================================

function ask(base, leadId, key, attempt = 0) {
  if (!STEPS[key]) {
    return finish('no_response', base);
  }

  const publicBase = getBaseUrl(base);

  const safeLeadId = encodeURIComponent(leadId);

  const actionUrl =
    `${publicBase}/edesy/gather` +
    `?lead=${safeLeadId}` +
    `&step=${encodeURIComponent(key)}` +
    `&try=${attempt}`;

  return {
    instructions: [
      // Play the prerecorded question.
      play(publicBase, key),

      // Collect one keypad digit.
      {
        verb: 'gather',
        num_digits: 1,
        timeout: 8,
        action: actionUrl,
      },

      // Played if no digit is entered.
      play(publicBase, 'no_response'),

      // End the call.
      {
        verb: 'hangup',
      },
    ],
  };
}

// ==========================================
// Handle keypad response
// ==========================================

function handle({
  base,
  leadId,
  key,
  digit,
  attempt = 0,
}) {
  const step = STEPS[key];

  if (!step) {
    return {
      voiceml: finish('no_response', base),
      outcome: 'no_response',
    };
  }

  const input = String(digit ?? '')
    .trim()
    .charAt(0);

  const answer =
    input === '1'
      ? 'yes'
      : input === '2'
        ? 'no'
        : null;

  // ----------------------------------------
  // Invalid input: repeat the question once.
  // ----------------------------------------

  if (!answer) {
    if (attempt < 1) {
      return {
        voiceml: ask(
          base,
          leadId,
          key,
          attempt + 1
        ),
      };
    }

    return {
      voiceml: finish('no_response', base),
      outcome: 'no_response',
    };
  }

  const rule = step[answer];

  // ----------------------------------------
  // Lead qualification completed.
  // ----------------------------------------

  if (rule.end) {
    return {
      voiceml: finish(rule.end, base),
      outcome: rule.end,
      answer,
    };
  }

  // ----------------------------------------
  // Continue to the next question.
  // ----------------------------------------

  return {
    voiceml: ask(
      base,
      leadId,
      rule.next,
      0
    ),
    answer,
  };
}

// ==========================================
// Export functions used by routes.js
// ==========================================

module.exports = {
  ask,
  handle,
  finish,
};
