
'use strict';

const fs = require('fs');
const path = require('path');
const router = require('express').Router();
const store = require('./store');
const dialer = require('./dialer');

const DEBUG = path.join(__dirname, '..', 'data', 'debug.log');

// ==========================================
// Debug logging
// ==========================================

const dbg = (tag, req) => {
  fs.appendFile(
    DEBUG,
    JSON.stringify({
      t: new Date().toISOString(),
      tag,
      query: req.query,
      body: req.body,
    }) + '\n',
    () => {}
  );
};

const merged = (req) => ({ ...req.query, ...(req.body || {}) });

// ==========================================
// CSV parser (numbers only)
// ==========================================

function parseCsv(text) {
  const lines = String(text || '')
    .replace(/^\uFEFF/, '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (!lines.length) return [];

  const first = lines[0]
    .replace(/^["']|["']$/g, '')
    .trim()
    .toLowerCase();

  const hasHeader = [
    'phone',
    'number',
    'mobile',
    'mobile number',
    'phone number',
  ].includes(first);

  const dataLines = hasHeader ? lines.slice(1) : lines;

  return dataLines.map((line) => ({
    name: '',
    phone: line.split(',')[0].replace(/^["']|["']$/g, '').trim(),
  }));
}

// ==========================================
// Edesy qualified lead webhook
// ==========================================

// Permanently save qualified lead from Edesy Studio HTTP Request node
router.post('/edesy/lead', (req, res) => {
  dbg('qualified-lead', req);

  const body = merged(req);
  const phone = String(body.phone || body.caller || '').trim();

  if (!phone) {
    return res.status(400).json({
      success: false,
      error: 'Phone number is required',
    });
  }

  const result = store.addQualified(phone);

  if (!result.success) {
    return res.status(400).json({
      success: false,
      phone,
      outcome: 'lead',
      ...result,
    });
  }

  return res.json({
    success: true,
    phone,
    outcome: 'lead',
    ...result,
  });
});

// ==========================================
// Edesy call status webhook
// ==========================================

router.all('/edesy/status', (req, res) => {
  dbg('status', req);

  const lead = store.get(req.query.lead);
  const body = merged(req);

  const status = String(
    body.status ||
      body.CallStatus ||
      body.call_status ||
      body.event ||
      ''
  ).toLowerCase();

  if (lead && lead.status === 'calling') {
    if (/no[-_ ]?answer|busy|failed|cancel/.test(status)) {
      store.update(lead.id, {
        status: 'no_answer',
        finishedAt: Date.now(),
      });
    } else if (/complete|ended|hangup/.test(status)) {
      store.update(lead.id, {
        status: 'no_response',
        finishedAt: Date.now(),
      });
    }
  }

  return res.sendStatus(200);
});

// ==========================================
// Dashboard API
// Authentication removed
// ==========================================

// Get active leads and counts
router.get('/api/leads', (req, res) => {
  return res.json({
    counts: store.counts(),
    leads: store.list().reverse(),
  });
});

// Add manually entered numbers
router.post('/api/leads', (req, res) => {
  return res.json(store.addMany(req.body.numbers || ''));
});

// Import CSV
router.post('/api/leads/import', (req, res) => {
  try {
    const csv = req.body.csv;

    if (typeof csv !== 'string' || !csv.trim()) {
      return res.status(400).json({
        error: 'CSV text is required',
      });
    }

    const rows = parseCsv(csv);
    const result = store.addCsvRows(rows);

    return res.json({
      success: true,
      ...result,
      totalRows: rows.length,
    });
  } catch (error) {
    return res.status(400).json({
      error: error.message,
    });
  }
});

// Start outbound calling
router.post('/api/start', (req, res) => {
  const pending = store.counts().pending || 0;

  dialer.start().catch((error) => {
    console.error('dialer:', error);
  });

  return res.json({
    started: true,
    pending,
  });
});

// ==========================================
// Export leads
// ==========================================

router.get('/api/export', (req, res) => {
  const status = String(req.query.status || 'lead');

  const phones =
    status === 'lead'
      ? store.getQualified().map((lead) => lead.phone)
      : store.list()
          .filter((lead) => lead.status === status)
          .map((lead) => lead.phone);

  const csv = ['phone', ...phones].join('\r\n');

  res.setHeader(
    'Content-Disposition',
    `attachment; filename="${status}-leads.csv"`
  );

  return res.type('text/csv').send(csv);
});

// ==========================================
// Export router
// ==========================================

module.exports = router;
