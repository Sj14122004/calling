'use strict';

require('dotenv').config();

const need = (key) => {
  if (!process.env[key]) {
    throw new Error(`Missing env var ${key} (see .env.example)`);
  }
  return process.env[key];
};

module.exports = {
  port: Number(process.env.PORT) || 3000,

  baseUrl: need('PUBLIC_BASE_URL').replace(/\/$/, ''),

  concurrency: Number(process.env.CONCURRENCY) || 5,

  adminToken: process.env.ADMIN_TOKEN || '',

  edesy: {
    base: process.env.EDESY_API_URL || 'https://voice-api.edesy.in/v1/ivr/calls',
    apiKey: need('EDESY_API_KEY'),
    flowId: need('EDESY_FLOW_ID'),
  },
};