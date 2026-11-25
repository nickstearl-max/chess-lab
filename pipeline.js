// image -> board -> pieces -> orientation/turn -> validated FEN (port of chesslab/pipeline.py).
// `ok` is only true when every safety check passed; otherwise `errors` says why and the
// app must not ask the engine for a move.

import {
  FINE_WARP, detectBoard, readSquares, readingsGrid, resolveOrientation, turnFromHighlights, warpBoard,
} from "./vision.js";
import { gameOverMessage, gridToPieceMap, gridToSquare, placementFen, validatePosition } from "./rules.js";

export const DEFAULT_OPTIONS = {
  orientation: "auto",            // auto | white | black   (side at the bottom)
  turn: "auto",                   // auto | white | black   (side to move)
  sticky: null,
  confidenceThreshold: 0.70,
  minBoardScore: 0.45,
  ambiguousThreshold: 0.35,
};

export function readPosition(pyr, prepared, options = {}, detection = null) {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  // 1. board
  if (!detection) detection = detectBoard(pyr);
  if (!detection) return { ok: false, errors: ["No chessboard found. Drag the corners onto the board."] };
  if (detection.score < opts.minBoardScore) {
    return { ok: false, detection, confidence: detection.score,
             errors: [`Board detection confidence too low (${Math.round(detection.score * 100)}%). `
                      + "Drag the corners onto the board."] };
  }

  // 2. pieces
  const warped = warpBoard(pyr, detection.corners, FINE_WARP);
  const squares = readSquares(warped, FINE_WARP, prepared);
  const grid = readingsGrid(squares);
  const confs = squares.map(s => s.confidence);
  const pieceConf = 0.6 * confs.reduce((a, v) => a + v, 0) / 64 + 0.4 * Math.min(...confs);
  const boardFactor = Math.min(1, Math.max(0, detection.score / 0.75)) ** 0.5;
  const confidence = pieceConf * boardFactor;
  const ambiguous = squares.filter(s => s.confidence < opts.ambiguousThreshold).map(s => [s.row, s.col]);

  // 3. orientation
  const orientation = resolveOrientation(grid, opts.orientation, opts.sticky);

  // 4. FEN + legality + side to move
  const map = gridToPieceMap(grid, orientation.whiteAtBottom);
  const visual = turnFromHighlights(squares);
  const report = opts.turn === "white" || opts.turn === "black"
    ? validatePosition(map, opts.turn === "white", true)
    : validatePosition(map, visual.whiteToMove, false, visual.method, visual.confidence);

  const errors = [...report.errors];
  if (ambiguous.length) errors.unshift(`${ambiguous.length} square(s) could not be read reliably (marked '?').`);
  if (confidence < opts.confidenceThreshold) {
    errors.push(`Recognition confidence ${Math.round(confidence * 100)}% is below the `
      + `${Math.round(opts.confidenceThreshold * 100)}% threshold. Try a sharper screenshot, drag the corners, `
      + "or calibrate this piece set.");
  }
  const warnings = [...report.warnings];
  if (orientation.method === "pieces" && orientation.confidence < 0.35) {
    errors.push("Board orientation is ambiguous - choose White at bottom or Black at bottom.");
  } else if (orientation.method === "sticky") {
    warnings.push("Orientation unclear from pieces; reusing the previous orientation.");
  }
  const ok = !errors.length && report.ok;
  return {
    ok, detection, warped, squares, orientation, confidence, pieceConfidence: pieceConf,
    placement: placementFen(map), fen: report.fen,
    uncertain: ambiguous.map(([r, c]) => gridToSquare(r, c, orientation.whiteAtBottom)),
    turnMethod: report.turnMethod, errors, warnings,
    gameOver: ok ? gameOverMessage(report.fen) : null,
  };
}
