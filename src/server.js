'use strict';

const express = require('express');
const path = require('path');
const cfg = require('./config');

const app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve dashboard and audio files from public/
app.use(express.static(path.join(__dirname, '..', 'public')));

// Register API and Edesy webhook routes
app.use(require('./routes'));

// Start server
app.listen(cfg.port, () => {
  console.log(
    `Lead caller on :${cfg.port} | Edesy webhooks -> ${cfg.baseUrl}`
  );
});