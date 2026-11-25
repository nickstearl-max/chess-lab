// End-to-end check of the JavaScript vision port on the sample screenshots (same ground
// truth as tests/test_vision.py).  Run:  node iphone_app/tests/run.mjs   (after build.py)

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { readPosition } from "../pipeline.js";
import {
  FINE_WARP, calibrateFromStart, decodeTemplateSet, detectBoard, encodeTemplateSet, makePyramid,
  prepareTemplates, warpBoard,
} from "../vision.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = join(HERE, "..");
const fixtures = JSON.parse(readFileSync(join(HERE, "fixtures", "manifest.json"), "utf8"));
const index = JSON.parse(readFileSync(join(APP, "templates", "index.json"), "utf8"));

function loadSet(id) {
  const e = index.find(s => s.id === id);
  return decodeTemplateSet(e.label, e.symbols, new Uint8Array(readFileSync(join(APP, e.file))));
}

function loadImage(f) {
  const data = new Uint8ClampedArray(gunzipSync(readFileSync(join(HERE, "fixtures", f.raw))));
  return { w: f.width, h: f.height, data };
}

let failures = 0;
const check = (cond, msg) => { if (!cond) { failures++; console.log("   FAIL", msg); } };

const prepared = prepareTemplates(loadSet("default"));
for (const f of fixtures) {
  const t0 = performance.now();
  const pyr = makePyramid(loadImage(f));
  const r = readPosition(pyr, prepared);
  const ms = Math.round(performance.now() - t0);
  console.log(`${f.file.padEnd(32)} ${r.ok ? "OK     " : "REFUSED"} ${String(ms).padStart(5)} ms  `
    + (r.ok ? r.fen : r.errors[0]));
  check(r.ok === f.expect_ok, `${f.file}: ok=${r.ok}, expected ${f.expect_ok}`);
  if (f.expect_ok) {
    check(r.placement === f.placement, `${f.file}: placement ${r.placement} != ${f.placement}`);
    check(r.orientation.whiteAtBottom === f.white_at_bottom, `${f.file}: orientation`);
    if (f.white_to_move !== null && r.fen) check((r.fen.split(" ")[1] === "w") === f.white_to_move, `${f.file}: turn`);
    const side = r.detection.corners.reduce((a, p, i, c) => a + Math.hypot(p[0] - c[(i + 1) % 4][0], p[1] - c[(i + 1) % 4][1]), 0) / 4;
    const err = Math.max(...r.detection.corners.flatMap((p, i) => [Math.abs(p[0] - f.corners[i][0]), Math.abs(p[1] - f.corners[i][1])]));
    check(err < 0.12 * side / 8, `${f.file}: corners off by ${err.toFixed(1)} px`);
  }
}

// unfamiliar pieces: refused with the default set, read after calibrating from the start position
const start = fixtures.find(f => f.file.startsWith("07"));
const pyr = makePyramid(loadImage(start));
const det = detectBoard(pyr);
const { set, whiteAtBottom } = calibrateFromStart(warpBoard(pyr, det.corners, FINE_WARP), FINE_WARP);
check(whiteAtBottom, "calibration: white at bottom");
const roundTrip = encodeTemplateSet(set);
const calibrated = prepareTemplates(decodeTemplateSet("site", roundTrip.symbols, roundTrip.bytes));
const r = readPosition(pyr, calibrated, {}, det);
console.log(`calibrated ${start.file.padEnd(21)} ${r.ok ? "OK     " : "REFUSED"}        ${r.ok ? r.fen : r.errors[0]}`);
check(r.ok && r.placement === start.placement, "calibrated set reads the starting position");

console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
