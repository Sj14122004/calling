'use strict';

const cfg = require('./config');

function playGather(audioUrl, actionUrl, maxDigits = 1) {
  return {
    action: actionUrl,
    method: 'POST',
    maxDigits,
    timeout: 8,
    play: audioUrl,
  };
}

function play(audioUrl) {
  return {
    play: audioUrl,
  };
}

function hangup() {
  return {
    hangup: true,
  };
}

function q1() {
  return playGather(
    cfg.audio.q1,
    `${cfg.baseUrl}/edesy/q1`
  );
}

function q2() {
  return playGather(
    cfg.audio.q2,
    `${cfg.baseUrl}/edesy/q2`
  );
}

function lead() {
  return play(cfg.audio.lead);
}

module.exports = {
  playGather,
  play,
  hangup,
  q1,
  q2,
  lead,
};