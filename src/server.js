'use strict';

const express = require('express');
const path = require('path');
const cfg = require('./config');

const app = express();

// ─── Middleware ──────────────────────────────────────────────
app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true }));

// ─── Health check ─────────────────────────────────────────
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'ok',
    service: 'edesy-lead-caller',
    timestamp: new Date().toISOString(),
  });
});

// ─── Static dashboard ──────────────────────────────────────
app.use(express.static(path.join(__dirname, '..', 'public')));

// ─── API & Webhooks ───────────────────────────────────────
app.use(require('./routes'));

// ─── Error handling ────────────────────────────────────────
app.use((err, req, res, next) => {
  console.error(`[Server Error] ${req.method} ${req.path}`, err);

  if (res.headersSent) {
    return next(err);
  }

  res.status(err.status || 500).json({
    error: 'Internal server error',
  });
});

// ─── Start server ──────────────────────────────────────────
const server = app.listen(cfg.port, '0.0.0.0', () => {
  console.log('────────────────────────────────────────');
  console.log(`Lead Caller running on port ${cfg.port}`);
  console.log(`Public URL: ${cfg.baseUrl}`);
  console.log(`Health: ${cfg.baseUrl}/health`);
  console.log('Server ready to receive requests');
  console.log('────────────────────────────────────────');
});

// ─── Graceful shutdown ────────────────────────────────────
function shutdown(signal) {
  console.log(`\n${signal} received. Shutting down...`);

  server.close(() => {
    console.log('Server closed.');
    process.exit(0);
  });

  // Force exit if shutdown hangs.
  setTimeout(() => {
    console.error('Shutdown timeout. Forcing exit.');
    process.exit(1);
  }, 10000).unref();
}

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));