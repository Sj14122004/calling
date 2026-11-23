const express = require('express');
const path = require('path');
const cfg = require('./config');

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, '..', 'public')));
app.use(require('./routes'));

app.listen(cfg.port, () => console.log(`Lead caller on :${cfg.port}  |  Edesy webhooks -> ${cfg.baseUrl}`));
