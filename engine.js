// Stockfish 19 (lite, single-threaded WASM) in a Web Worker, spoken to over UCI.
// A new analyse() call stops the previous search first; superseded calls resolve to null.

export class Engine {
  constructor(url) {
    this.url = url;
    this.worker = null;
    this.listeners = new Set();
    this.ready = null;
    this.running = null;
    this.gen = 0;
    this.name = "Stockfish";
  }

  init() {
    if (this.ready) return this.ready;
    this.worker = new Worker(this.url);
    this.worker.onmessage = e => {
      const text = typeof e.data === "string" ? e.data : String(e.data);
      for (const line of text.split("\n")) for (const f of [...this.listeners]) f(line.trim());
    };
    this.ready = new Promise((resolve, reject) => {
      this.worker.onerror = err => reject(new Error(err.message || "Stockfish failed to load"));
      const f = line => {
        if (line.startsWith("id name ")) this.name = line.slice(8);
        if (line === "uciok") this.worker.postMessage("isready");
        if (line === "readyok") { this.listeners.delete(f); resolve(); }
      };
      this.listeners.add(f);
      this.worker.postMessage("uci");
    });
    return this.ready;
  }

  /** Resolves to {bestmove, depth, cp, mate, pv} with cp/mate from White's point of view. */
  async analyse(fen, { depth = 16, skill = 20, onInfo } = {}) {
    await this.init();
    const gen = ++this.gen;
    if (this.running) {
      this.worker.postMessage("stop");
      await this.running;
    }
    if (gen !== this.gen) return null;
    const sign = fen.split(" ")[1] === "b" ? -1 : 1;
    let last = { depth: 0, cp: null, mate: null, pv: [] };
    this.running = new Promise(resolve => {
      const f = line => {
        if (line.startsWith("info ") && line.includes(" pv ") && !/ (lower|upper)bound /.test(line)) {
          const t = line.split(" ");
          const at = key => t.indexOf(key);
          const info = { depth: Number(t[at("depth") + 1]) || 0, cp: null, mate: null, pv: t.slice(at("pv") + 1) };
          const s = at("score");
          if (s >= 0 && t[s + 1] === "cp") info.cp = sign * Number(t[s + 2]);
          if (s >= 0 && t[s + 1] === "mate") info.mate = sign * Number(t[s + 2]);
          last = info;
          if (onInfo && gen === this.gen) onInfo(info);
        } else if (line.startsWith("bestmove")) {
          this.listeners.delete(f);
          const best = line.split(" ")[1];
          resolve({ ...last, bestmove: best && best !== "(none)" ? best : null });
        }
      };
      this.listeners.add(f);
      this.worker.postMessage(`setoption name Skill Level value ${skill}`);
      this.worker.postMessage("ucinewgame");
      this.worker.postMessage(`position fen ${fen}`);
      this.worker.postMessage(`go depth ${depth}`);
    });
    const result = await this.running;
    this.running = null;
    return gen === this.gen ? result : null;
  }
}

export function formatEval(cp, mate) {
  if (mate !== null && mate !== undefined) {
    if (mate === 0) return "Mate";
    return `Mate in ${Math.abs(mate)} (${mate > 0 ? "White" : "Black"})`;
  }
  if (cp === null || cp === undefined) return "n/a";
  return (cp >= 0 ? "+" : "") + (cp / 100).toFixed(2);
}

export function shortEval(cp, mate) {
  if (mate !== null && mate !== undefined) return mate === 0 ? "#" : `${mate > 0 ? "+" : "-"}M${Math.abs(mate)}`;
  if (cp === null || cp === undefined) return "";
  return (cp >= 0 ? "+" : "") + (cp / 100).toFixed(2);
}

/** White's share of the eval bar, 0..1 (same curve as the desktop overlay). */
export function winFraction(cp, mate) {
  if (mate !== null && mate !== undefined) return mate > 0 ? 1 : 0;
  if (cp === null || cp === undefined) return 0.5;
  const c = Math.max(-1000, Math.min(1000, cp));
  return 0.5 + 0.5 * (2 / (1 + Math.exp(-0.00368208 * c)) - 1);
}
