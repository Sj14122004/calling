
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const DATA_DIR = path.join(__dirname, '..', 'data');
const LEADS_FILE = path.join(DATA_DIR, 'leads.json');
const QUALIFIED_FILE = path.join(DATA_DIR, 'qualified-leads.json');

fs.mkdirSync(DATA_DIR, { recursive: true });

const leads = new Map();
const byPhone = new Map();
const qualified = new Map();

function loadFile(file) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return [];
  }
}

// Load active leads
for (const lead of loadFile(LEADS_FILE)) {
  leads.set(lead.id, lead);
  byPhone.set(lead.phone, lead.id);
}

// Load permanently saved qualified numbers
for (const item of loadFile(QUALIFIED_FILE)) {
  qualified.set(item.phone, item);
}

let leadsTimer;
let qualifiedTimer;

function saveLeads() {
  clearTimeout(leadsTimer);

  leadsTimer = setTimeout(() => {
    const temp = LEADS_FILE + '.tmp';

    fs.writeFile(temp, JSON.stringify([...leads.values()], null, 2), (err) => {
      if (!err) fs.rename(temp, LEADS_FILE, () => {});
    });
  }, 200);
}

function saveQualified() {
  clearTimeout(qualifiedTimer);

  qualifiedTimer = setTimeout(() => {
    const temp = QUALIFIED_FILE + '.tmp';

    fs.writeFile(temp, JSON.stringify([...qualified.values()], null, 2), (err) => {
      if (!err) fs.rename(temp, QUALIFIED_FILE, () => {});
    });
  }, 200);
}

// Normalize Indian phone numbers
function normalize(raw) {
  const digits = String(raw ?? '').replace(/\D/g, '');

  if (String(raw).trim().startsWith('+') && digits.length >= 11 && digits.length <= 14) {
    return '+' + digits;
  }

  if (digits.length === 10) {
    return '+91' + digits;
  }

  if (digits.length === 11 && digits.startsWith('0')) {
    return '+91' + digits.slice(1);
  }

  if (digits.length === 12 && digits.startsWith('91')) {
    return '+' + digits;
  }

  return null;
}

// Add manually pasted numbers
function addMany(text) {
  let added = 0;
  let duplicate = 0;
  let invalid = 0;

  for (const raw of String(text)
    .split(/[\n,;]+/)
    .map((value) => value.trim())
    .filter(Boolean)) {

    const phone = normalize(raw);

    if (!phone) {
      invalid++;
      continue;
    }

    if (byPhone.has(phone) || qualified.has(phone)) {
      duplicate++;
      continue;
    }

    const id = crypto.randomUUID();

    const lead = {
      id,
      name: '',
      phone,
      status: 'pending',
      answers: {},
      createdAt: Date.now(),
    };

    leads.set(id, lead);
    byPhone.set(phone, id);
    added++;
  }

  saveLeads();
  return { added, duplicate, invalid };
}

// Import numbers from CSV
function addCsvRows(rows) {
  let added = 0;
  let duplicate = 0;
  let invalid = 0;

  for (const row of rows) {
    const phone = normalize(row.phone);

    if (!phone) {
      invalid++;
      continue;
    }

    if (byPhone.has(phone) || qualified.has(phone)) {
      duplicate++;
      continue;
    }

    const id = crypto.randomUUID();

    const lead = {
      id,
      name: '',
      phone,
      status: 'pending',
      answers: {},
      createdAt: Date.now(),
    };

    leads.set(id, lead);
    byPhone.set(phone, id);
    added++;
  }

  saveLeads();
  return { added, duplicate, invalid };
}

// Directly save qualified number from Edesy webhook
function addQualified(rawPhone) {
  const phone = normalize(rawPhone);

  if (!phone) {
    return {
      success: false,
      added: 0,
      duplicate: 0,
      invalid: 1,
    };
  }

  // Prevent duplicate webhook retries
  if (qualified.has(phone)) {
    return {
      success: true,
      added: 0,
      duplicate: 1,
      invalid: 0,
    };
  }

  // Save permanently
  qualified.set(phone, {
    phone,
    qualifiedAt: Date.now(),
  });

  // Remove from active dashboard if present
  const activeId = byPhone.get(phone);

  if (activeId) {
    leads.delete(activeId);
    byPhone.delete(phone);
    saveLeads();
  }

  saveQualified();

  return {
    success: true,
    added: 1,
    duplicate: 0,
    invalid: 0,
  };
}

// Get active lead
const get = (id) => leads.get(id);

// Get active dashboard leads
const list = () => [...leads.values()];

// Update lead
function update(id, patch) {
  const lead = leads.get(id);

  if (!lead) return null;

  Object.assign(lead, patch);

  // Final call statuses
  const finishedStatuses = [
    'lead',
    'removed',
    'no_response',
    'no_answer',
    'failed',
  ];

  if (finishedStatuses.includes(lead.status)) {
    // Save qualified number permanently
    if (lead.status === 'lead') {
      qualified.set(lead.phone, {
        phone: lead.phone,
        qualifiedAt: Date.now(),
      });

      saveQualified();
    }

    // Remove from active dashboard storage
    leads.delete(id);
    byPhone.delete(lead.phone);

    saveLeads();
    return lead;
  }

  saveLeads();
  return lead;
}

// Dashboard counts
function counts() {
  const result = {};

  for (const lead of leads.values()) {
    result[lead.status] = (result[lead.status] || 0) + 1;
  }

  return result;
}

// Get permanently saved qualified numbers
function getQualified() {
  return [...qualified.values()];
}

module.exports = {
  addMany,
  addCsvRows,
  addQualified,
  get,
  list,
  update,
  counts,
  getQualified,
};
