/*
 * ergo_rule.js - Mesin aturan EAR ERGO-SENSE (v0.1). Logika sama persis dengan ergo_rule.py.
 *
 * Browser : <script src="ergo_rule.js"></script>  ->  window.ErgoRule
 * Node    : const { RuleEngine } = require('./ergo_rule.js');
 *
 *   const cfg = await (await fetch('ergo_rule_config.json')).json();
 *   const engine = new ErgoRule.RuleEngine(cfg);          // atau (cfg, 'high')
 *   // tiap frame:
 *   const ear = ErgoRule.earFromMediapipe(result.faceLandmarks[0], video.videoWidth, video.videoHeight); // null bila tak ada wajah
 *   const out = engine.update(performance.now() / 1000, ear);   // detik; out = objek per detik, atau null
 *
 * Waktu HARUS dalam detik dan berasal dari timestamp frame, bukan jumlah frame.
 * Koordinat landmark dikonversi ke PIKSEL (x * lebar, y * tinggi) sebelum menghitung jarak.
 * Parameter bertanda 'provisional' di ergo_rule_config.json adalah tebakan desain, bukan hasil data.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.ErgoRule = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const RIGHT_EYE = [33, 160, 158, 133, 153, 144];
  const LEFT_EYE = [362, 385, 387, 263, 373, 380];

  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const earOneEye = (p) => (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * dist(p[0], p[3]));
  const earTwoEyes = (right, left) => (earOneEye(right) + earOneEye(left)) / 2.0;

  // landmarks: array objek {x, y} ternormalisasi 0..1 (keluaran MediaPipe Tasks Vision). Mengembalikan EAR atau null.
  function earFromMediapipe(landmarks, width, height) {
    if (!landmarks || landmarks.length < 400) return null;
    const pt = (i) => [landmarks[i].x * width, landmarks[i].y * height];
    const e = earTwoEyes(RIGHT_EYE.map(pt), LEFT_EYE.map(pt));
    return Number.isFinite(e) ? e : null;
  }

  function percentile(values, q) {
    const xs = values.slice().sort((a, b) => a - b);
    const pos = (xs.length - 1) * q / 100.0;
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo);
  }

  class RuleEngine {
    constructor(cfg, sensitivity) {
      if (!cfg) throw new Error('cfg (isi ergo_rule_config.json) wajib diberikan');
      this.cfg = cfg;
      this.setSensitivity(sensitivity || cfg.default_sensitivity);
      this.reset();
    }
    setSensitivity(name) { this.sensitivity = name; this.k = this.cfg.sensitivity[name]; }
    reset() {
      this.t0 = null; this.baseline = null; this.calibratedAt = null;
      this.calibEars = []; this.calibValidTime = 0.0;
      this.prev = null; this.win = [];
      this.nextTick = null; this.lastValidT = null; this.lastEar = null; this.lastRatio = null;
      this.redSince = null; this.armed = true; this.lastAlertT = -Infinity;
    }
    recalibrate(t) { const keep = this.sensitivity; this.reset(); this.sensitivity = keep; this.t0 = t; }

    update(t, ear) {
      const cfg = this.cfg;
      const events = [];
      if (this.t0 === null) { this.t0 = t; this.nextTick = t + cfg.tick_s; }
      if (ear !== null && ear !== undefined && !Number.isFinite(ear)) ear = null;
      if (ear === undefined) ear = null;
      if (ear !== null) { this.lastValidT = t; this.lastEar = ear; }

      if (this.prev !== null) {
        const [pt, pe] = this.prev;
        const dt = Math.min(t - pt, cfg.dt_cap_s);
        if (dt > 0) this._finalize(pt, dt, pe);
      }
      this.prev = [t, ear];

      if (this.baseline === null && t - this.t0 >= cfg.calibration_s) {
        const validFrac = this.calibValidTime / cfg.calibration_s;
        if (validFrac >= cfg.calibration_min_valid_frac && this.calibEars.length > 0) {
          this.baseline = percentile(this.calibEars, cfg.baseline_percentile);
          this.calibratedAt = t;
          this.win = [];
          events.push({ type: 'calibration_done', baseline_ear: this.baseline, valid_frac: Math.min(validFrac, 1.0) });
        } else {
          events.push({ type: 'calibration_failed', valid_frac: Math.min(validFrac, 1.0) });
          const keepNext = this.nextTick;
          this.recalibrate(t);
          this.nextTick = keepNext;
          this.prev = [t, ear];
        }
      }
      if (this.baseline !== null && ear !== null) this.lastRatio = ear / this.baseline;

      if (t >= this.nextTick) {
        const out = this._tick(t, events);
        while (this.nextTick <= t) this.nextTick += cfg.tick_s;
        return out;
      }
      if (events.length) return this._tick(t, events);
      return null;
    }

    _finalize(pt, dt, pe) {
      if (this.baseline === null) {
        if (this.t0 !== null && pt >= this.t0 && pe !== null) { this.calibEars.push(pe); this.calibValidTime += dt; }
        return;
      }
      if (pe === null) this.win.push([pt, dt, 'invalid', null]);
      else { const r = pe / this.baseline; this.win.push([pt, dt, r < this.cfg.closed_ratio ? 'closed' : 'open', r]); }
    }

    _tick(now, events) {
      const cfg = this.cfg, W = cfg.window_s;
      while (this.win.length && this.win[0][0] <= now - W) this.win.shift();
      const out = {
        t: now, calibrated: this.baseline !== null, baseline_ear: this.baseline,
        ear: this.lastEar, ear_ratio: this.lastRatio, sensitivity: this.sensitivity,
        p50_30: null, maxclose_30_s: null, valid_frac_30: null, ikm: null,
        status: 'uncertain', reason: null, alert: false, events: events,
      };
      if (this.baseline === null) { out.reason = 'calibrating'; this.redSince = null; return out; }

      let validTime = 0, closedTime = 0, run = 0, longest = 0;
      for (const [, dt, st] of this.win) {
        if (st === 'invalid') { run = 0; continue; }
        validTime += dt;
        if (st === 'closed') { closedTime += dt; run += dt; longest = Math.max(longest, run); }
        else run = 0;
      }
      const validFrac = Math.min(validTime / W, 1.0);
      out.valid_frac_30 = validFrac;
      out.maxclose_30_s = longest;
      out.p50_30 = validTime > 0 ? closedTime / validTime : null;

      if (now - this.calibratedAt < cfg.warmup_s) out.reason = 'warming_up';
      else if (this.lastValidT === null || now - this.lastValidT > cfg.no_face_uncertain_s) out.reason = 'no_face';
      else if (validFrac < cfg.min_valid_frac || out.p50_30 === null) out.reason = 'low_valid';
      else {
        const ikm = Math.max(0.0, Math.min(100.0, 100.0 * out.p50_30 / (cfg.ikm_divisor * this.k)));
        out.ikm = ikm;
        out.status = ikm >= cfg.ikm_red ? 'red' : (ikm >= cfg.ikm_yellow ? 'yellow' : 'green');
      }

      const a = cfg.alert;
      if (out.status === 'red') {
        if (this.redSince === null) this.redSince = now;
        if (this.armed && now - this.redSince >= a.red_hold_s && now - this.lastAlertT >= a.cooldown_s) {
          out.alert = true; this.armed = false; this.lastAlertT = now;
        }
      } else this.redSince = null;
      if (out.ikm !== null && out.ikm < a.rearm_ikm) this.armed = true;
      return out;
    }
  }

  return { RuleEngine, earFromMediapipe, earTwoEyes, earOneEye, percentile, RIGHT_EYE, LEFT_EYE };
});

// CLI: node ergo_rule.js input.csv [high|medium|low]   (kolom: t,ear ; ear kosong = wajah tidak terdeteksi)
if (typeof require !== 'undefined' && typeof module !== 'undefined' && require.main === module) {
  const fs = require('fs'), path = require('path');
  if (process.argv.length < 3) { console.error('pemakaian: node ergo_rule.js input.csv [sensitivitas]'); process.exit(1); }
  const cfg = JSON.parse(fs.readFileSync(path.join(__dirname, 'ergo_rule_config.json'), 'utf8'));
  const { RuleEngine } = module.exports;
  const eng = new RuleEngine(cfg, process.argv[3]);
  const lines = fs.readFileSync(process.argv[2], 'utf8').split(/\r?\n/).slice(1);
  for (const line of lines) {
    if (!line.trim()) continue;
    const i = line.indexOf(',');
    const t = parseFloat(line.slice(0, i)), e = line.slice(i + 1).trim();
    const out = eng.update(t, e === '' ? null : parseFloat(e));
    if (out) console.log(JSON.stringify(out));
  }
}
