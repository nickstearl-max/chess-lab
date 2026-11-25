// Runs the board vision off the UI thread. Messages: {id, type, ...} -> {id, result | error}.

import { readPosition } from "./pipeline.js";
import {
  FINE_WARP, calibrateFromStart, decodeTemplateSet, detectionFromQuad, encodeTemplateSet, makePyramid,
  prepareTemplates, warpBoard,
} from "./vision.js";

const sets = new Map();          // id -> prepared template set
let pyramid = null;
let lastDetection = null;

function toRGBA(F, n) {
  const out = new Uint8ClampedArray(n * n * 4);
  for (let i = 0; i < n * n; i++) {
    out[i * 4] = F[i * 3];
    out[i * 4 + 1] = F[i * 3 + 1];
    out[i * 4 + 2] = F[i * 3 + 2];
    out[i * 4 + 3] = 255;
  }
  return out;
}

const handlers = {
  "load-set"({ setId, label, symbols, bytes }) {
    sets.set(setId, prepareTemplates(decodeTemplateSet(label, symbols, new Uint8Array(bytes))));
    return { setId };
  },

  image({ w, h, data }) {
    pyramid = makePyramid({ w, h, data: new Uint8ClampedArray(data) });
    lastDetection = null;
    return { w, h };
  },

  read({ setId, options, corners }) {
    if (!pyramid) throw new Error("No image loaded.");
    const prepared = sets.get(setId);
    if (!prepared) throw new Error(`Piece set ${setId} is not loaded.`);
    const detection = corners ? detectionFromQuad(pyramid, corners) : null;
    const r = readPosition(pyramid, prepared, options, detection);
    if (r.detection) lastDetection = r.detection;
    const transfer = [];
    let warped = null;
    if (r.warped) {
      warped = toRGBA(r.warped, FINE_WARP);
      transfer.push(warped.buffer);
    }
    const out = {
      ok: r.ok, errors: r.errors || [], warnings: r.warnings || [], fen: r.fen || "", placement: r.placement || "",
      detection: r.detection || null, confidence: r.confidence || 0, uncertain: r.uncertain || [],
      orientation: r.orientation || null, turnMethod: r.turnMethod || "", gameOver: r.gameOver || null,
      warped, warpedSize: FINE_WARP,
    };
    return [out, transfer];
  },

  calibrate({ name, orientation }) {
    if (!pyramid || !lastDetection) throw new Error("Load a screenshot of a board in the starting position first.");
    const warped = warpBoard(pyramid, lastDetection.corners, FINE_WARP);
    const wab = orientation === "white" ? true : orientation === "black" ? false : null;
    const { set, whiteAtBottom } = calibrateFromStart(warped, FINE_WARP, wab, name);
    const { symbols, bytes } = encodeTemplateSet(set);
    return [{ symbols, bytes: bytes.buffer, whiteAtBottom }, [bytes.buffer]];
  },
};

self.onmessage = e => {
  const { id, type, ...args } = e.data;
  try {
    const res = handlers[type](args);
    const [result, transfer] = Array.isArray(res) ? res : [res, []];
    self.postMessage({ id, result }, transfer);
  } catch (err) {
    self.postMessage({ id, error: err && err.message ? err.message : String(err) });
  }
};
