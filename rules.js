// Position building and legality checks: a port of chesslab/chesscore (fen_builder.py,
// validation.py). Never hand an impossible position to the engine.

import { Chess } from "./vendor/chess.js";

const FILES = "abcdefgh";

export function squareName(file, rank) {
  return FILES[file] + (rank + 1);
}

/** Image-grid cell (row 0 = top) -> square name. */
export function gridToSquare(row, col, whiteAtBottom) {
  return whiteAtBottom ? squareName(col, 7 - row) : squareName(7 - col, row);
}

/** Square name -> image-grid [row, col]. */
export function squareToGrid(sq, whiteAtBottom) {
  const f = FILES.indexOf(sq[0]), r = Number(sq[1]) - 1;
  return whiteAtBottom ? [7 - r, f] : [r, 7 - f];
}

/** grid of symbols -> {square: symbol} */
export function gridToPieceMap(grid, whiteAtBottom) {
  const map = {};
  for (let row = 0; row < 8; row++)
    for (let col = 0; col < 8; col++)
      if (grid[row][col]) map[gridToSquare(row, col, whiteAtBottom)] = grid[row][col];
  return map;
}

export function placementFen(map) {
  const rows = [];
  for (let rank = 7; rank >= 0; rank--) {
    let row = "", empty = 0;
    for (let file = 0; file < 8; file++) {
      const sym = map[squareName(file, rank)];
      if (sym) { if (empty) row += empty; empty = 0; row += sym; } else empty++;
    }
    rows.push(row + (empty ? empty : ""));
  }
  return rows.join("/");
}

/** Castling rights wherever king and rook stand on their home squares; no en passant. */
export function buildFen(map, whiteToMove) {
  let castling = "";
  if (map.e1 === "K") { if (map.h1 === "R") castling += "K"; if (map.a1 === "R") castling += "Q"; }
  if (map.e8 === "k") { if (map.h8 === "r") castling += "k"; if (map.a8 === "r") castling += "q"; }
  return `${placementFen(map)} ${whiteToMove ? "w" : "b"} ${castling || "-"} - 0 1`;
}

/** Problems with the position, as python-chess' Board.status() would report them. */
export function positionErrors(map, whiteToMove) {
  const errors = [], count = {};
  for (const sym of Object.values(map)) count[sym] = (count[sym] || 0) + 1;
  const total = colour => Object.entries(count)
    .filter(([s]) => (s === s.toUpperCase()) === (colour === "w")).reduce((a, [, n]) => a + n, 0);
  if (!Object.keys(map).length) return ["The board is empty"];
  if (!count.K) errors.push("No white king found");
  if (!count.k) errors.push("No black king found");
  if ((count.K || 0) > 1 || (count.k || 0) > 1) errors.push("More than one king of a colour");
  if ((count.P || 0) > 8) errors.push("More than 8 white pawns");
  if ((count.p || 0) > 8) errors.push("More than 8 black pawns");
  if (Object.entries(map).some(([sq, s]) => s.toLowerCase() === "p" && (sq[1] === "1" || sq[1] === "8"))) {
    errors.push("Pawn on the first or eighth rank");
  }
  if (total("w") > 16) errors.push("More than 16 white pieces");
  if (total("b") > 16) errors.push("More than 16 black pieces");
  if (errors.length) return errors;

  const board = new Chess(buildFen(map, whiteToMove), { skipValidation: true });
  const kingSq = sym => Object.keys(map).find(sq => map[sq] === sym);
  const mover = whiteToMove ? "w" : "b", waiting = whiteToMove ? "b" : "w";
  if (board.isAttacked(kingSq(waiting === "w" ? "K" : "k"), mover)) errors.push("The side that just moved is in check");
  if (board.attackers(kingSq(mover === "w" ? "K" : "k"), waiting).length > 2) {
    errors.push("Impossible number of checking pieces");
  }
  return errors;
}

/**
 * Build and validate a position, resolving the side to move (validation.validate_position).
 * Returns {ok, fen, errors, warnings, turnMethod, turnConfidence}.
 */
export function validatePosition(map, preferredWhiteToMove, manual = false, preferredMethod = "highlight",
                                 preferredConf = 0) {
  if (manual && preferredWhiteToMove !== null) {
    const errors = positionErrors(map, preferredWhiteToMove);
    return { ok: !errors.length, fen: buildFen(map, preferredWhiteToMove), errors, warnings: [],
             turnMethod: "manual", turnConfidence: 1 };
  }
  const options = {};
  for (const wtm of [true, false]) options[wtm] = positionErrors(map, wtm);
  const valid = [true, false].filter(wtm => !options[wtm].length);
  if (!valid.length) {
    const wtm = preferredWhiteToMove ?? true;
    return { ok: false, fen: buildFen(map, wtm), errors: options[wtm], warnings: [], turnMethod: "unknown", turnConfidence: 0 };
  }
  if (valid.length === 1) {
    const wtm = valid[0], warnings = [];
    if (preferredWhiteToMove !== null && preferredWhiteToMove !== wtm) {
      warnings.push("Highlight suggested the other side to move, but that position is illegal.");
    }
    return { ok: true, fen: buildFen(map, wtm), errors: [], warnings, turnMethod: "check-rule", turnConfidence: 0.95 };
  }
  if (preferredWhiteToMove !== null) {
    return { ok: true, fen: buildFen(map, preferredWhiteToMove), errors: [], warnings: [],
             turnMethod: preferredMethod, turnConfidence: preferredConf };
  }
  return { ok: true, fen: buildFen(map, true), errors: [], turnMethod: "default", turnConfidence: 0.3,
           warnings: ["Side to move could not be detected - assuming White. Set it under Turn if wrong."] };
}

export function gameOverMessage(fen) {
  const board = new Chess(fen, { skipValidation: true });
  if (board.isCheckmate()) return `Checkmate - ${board.turn() === "w" ? "Black" : "White"} has won. No move to recommend.`;
  if (board.isStalemate()) return "Stalemate - no legal moves.";
  if (board.isInsufficientMaterial()) return "Draw by insufficient material.";
  return null;
}

const PIECE_NAMES = { p: "Pawn", n: "Knight", b: "Bishop", r: "Rook", q: "Queen", k: "King" };

/** UCI moves -> SAN line, plus a description of the first move. */
export function describeLine(fen, uciMoves) {
  const board = new Chess(fen, { skipValidation: true });
  const san = [];
  let first = null;
  for (const uci of uciMoves) {
    let mv;
    try {
      mv = board.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
    } catch {
      break;
    }
    if (!mv) break;
    if (!first) {
      first = {
        uci, from: mv.from, to: mv.to, san: mv.san, piece: mv.color === "w" ? mv.piece.toUpperCase() : mv.piece,
        desc: `${mv.color === "w" ? "White" : "Black"} ${PIECE_NAMES[mv.piece]} ${mv.from} → ${mv.to}`
          + (mv.promotion ? ` (promotes to ${PIECE_NAMES[mv.promotion]})` : ""),
      };
    }
    san.push(mv.san);
  }
  return { first, san };
}

export function isLegalFen(fen) {
  try {
    const parts = fen.trim().split(/\s+/);
    if (parts.length < 2) return "FEN needs at least the placement and the side to move.";
    const map = {};
    const rows = parts[0].split("/");
    if (rows.length !== 8) return "FEN placement must have 8 ranks.";
    rows.forEach((row, i) => {
      let file = 0;
      for (const ch of row) {
        if (/\d/.test(ch)) file += Number(ch);
        else { if (!/[pnbrqkPNBRQK]/.test(ch) || file > 7) throw new Error(); map[squareName(file, 7 - i)] = ch; file++; }
      }
      if (file !== 8) throw new Error();
    });
    const errors = positionErrors(map, parts[1] !== "b");
    return errors.length ? errors.join("; ") : null;
  } catch {
    return "That FEN could not be read.";
  }
}
