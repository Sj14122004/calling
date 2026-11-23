
'use strict';

const fs = require('fs');
const path = require('path');
const router = require('express').Router();
const cfg = require('./config');
const store = require('./store');
const flow = require('./flow');
const dialer = require('./dialer');

const DEBUG = path.join(__dirname, '..', 'data', 'debug.log');

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

const digitOf = (body) =>
  String(body.Digits ?? body.digits ?? body.Digit ?? body.digit ?? body.dtmf ?? '')
    .trim()
    .charAt(0);

// ---------- CSV parser (numbers only) ----------

function parseCsv(text) {
  const lines = String(text || '')
    .replace(/^\uFEFF/, '')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (!lines.length) return [];

  const first = lines[0].replace(/^["']|["']$/g, '').trim().toLowerCase();

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

// ---------- Edesy callbacks ----------

// Permanently save qualified lead from Edesy HTTP Request node
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

  res.json({
    success: true,
    phone,
    outcome: 'lead',
    ...result,
  });
});

// ---------- Legacy Edesy callbacks (kept) ----------

router.all('/edesy/answer', (req, res) => {
  dbg('answer', req);

  const lead = store.get(req.query.lead);

  if (!lead) return res.json(flow.finish('removed', cfg.baseUrl));

  res.json(flow.ask(cfg.baseUrl, lead.id, 'q1'));
});

router.all('/edesy/gather', (req, res) => {
  dbg('gather', req);

  const lead = store.get(req.query.lead);

  if (!lead) return res.json(flow.finish('removed', cfg.baseUrl));

  const key = req.query.step;

  const result = flow.handle({
    base: cfg.baseUrl,
    leadId: lead.id,
    key,
    digit: digitOf(merged(req)),
    attempt: Number(req.query.try) || 0,
  });

  const patch = {
    answers: {
      ...lead.answers,
      ...(result.answer ? { [key]: result.answer } : {}),
    },
  };

  if (result.outcome) {
    patch.status = result.outcome;
    patch.finishedAt = Date.now();
  }

  store.update(lead.id, patch);
  res.json(result.voiceml);
});

router.all('/edesy/status', (req, res) => {
  dbg('status', req);

  const lead = store.get(req.query.lead);
  const body = merged(req);

  const status = String(
    body.status || body.CallStatus || body.call_status || body.event || ''
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

  res.sendStatus(200);
});

// ---------- Dashboard API ----------

const guard = (req, res, next) =>
  !cfg.adminToken || req.get('x-admin-token') === cfg.adminToken
    ? next()
    : res.status(401).json({ error: 'unauthorized' });

router.get('/api/leads', guard, (req, res) => {
  res.json({
    counts: store.counts(),
    leads: store.list().reverse(),
  });
});

router.post('/api/leads', guard, (req, res) => {
  res.json(store.addMany(req.body.numbers || ''));
});

router.post('/api/leads/import', guard, (req, res) => {
  try {
    const csv = req.body.csv;

    if (typeof csv !== 'string' || !csv.trim()) {
      return res.status(400).json({ error: 'CSV text is required' });
    }

    const rows = parseCsv(csv);
    const result = store.addCsvRows(rows);

    res.json({
      success: true,
      ...result,
      totalRows: rows.length,
    });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

router.post('/api/start', guard, (req, res) => {
  const pending = store.counts().pending || 0;

  dialer.start().catch((error) => console.error('dialer:', error));

  res.json({
    started: true,
    pending,
  });
});

// Export qualified leads as numbers only
router.get('/api/export', guard, (req, res) => {
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

  res.type('text/csv').send(csv);
});

module.exports = router;
