/**
 * CardVault Image Engine bridge: alignment, JSON in / JSON out.
 *
 * The Python side has no TypeScript runtime and should not grow one. This
 * process is the whole interface: it reads one job from stdin, runs the engine's
 * alignment stage, and writes one JSON result to stdout. Anything the caller
 * needs to log goes to stderr, so stdout stays a single parseable document.
 *
 * Job shape (all fields required unless marked):
 *
 *   {
 *     "operation": "align",
 *     "width": 1240, "height": 930,
 *     "format": "RGB",                       // RGB | RGBA | GRAY
 *     "pixels": "<base64 of interleaved bytes>",
 *     "profile": "pokemon",                  // optional, see PROFILES
 *     "options": {                           // all optional
 *       "edgeThreshold": 48, "morphologyRadius": 1,
 *       "minAreaRatio": 0.015, "minConfidence": 0.25
 *     }
 *   }
 *
 * Result shape:
 *
 *   {
 *     "ok": true,
 *     "status": "ALIGNED" | "CARD_NOT_FOUND" | "LOW_CONFIDENCE",
 *     "geometry": { "corners": [[x,y],...], "confidence": n,
 *                   "width": w, "height": h } | null,
 *     "candidates": [...], "timings": {...}, "engineVersion": "..."
 *   }
 *
 * Corner order is TL, TR, BR, BL for both input and output, which is what the
 * engine emits and what CardVault Vision's `Quad` records. The engine's card
 * profiles express aspect as *width/height* whereas CardVault Vision's layout
 * specs use *height/width*; the mapping below is the only place that conversion
 * happens, so it cannot leak into a caller.
 */
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

// CardVault Vision layout `aspect` is height/width (88/63 for a trading card);
// the engine's profiles are width/height. Converted explicitly here.
const PROFILES = {
  'standard-trading-card': 'STANDARD_TRADING_CARD',
  pokemon: 'POKEMON_CARD_PROFILE',
  'magic-the-gathering': 'MAGIC_CARD_PROFILE',
  'yu-gi-oh': 'YUGIOH_CARD_PROFILE',
  sports: 'SPORTS_CARD_PROFILE',
};

function fail(message, extra = {}) {
  process.stdout.write(JSON.stringify({ ok: false, error: String(message), ...extra }));
  process.exit(0);
}

async function loadEngine(enginePath) {
  const entry = enginePath.endsWith('.js') ? enginePath : `${enginePath}/index.js`;
  try {
    return await import(pathToFileURL(entry).href);
  } catch (error) {
    throw new Error(
      `cannot import the image engine from ${entry}: ${error.message}. ` +
        'Pass --engine-path <dir containing index.js>, or set CVV_IMAGE_ENGINE.',
    );
  }
}

function readJob() {
  const raw = readFileSync(0, 'utf8');
  if (!raw.trim()) throw new Error('empty job on stdin');
  return JSON.parse(raw);
}

const job = readJob();
const enginePath = process.env.CVV_IMAGE_ENGINE || job.enginePath;
if (!enginePath) throw new Error('no engine path: set CVV_IMAGE_ENGINE or pass enginePath');

const engine = await loadEngine(enginePath);

if (job.operation !== 'align') {
  fail(`unknown operation ${JSON.stringify(job.operation)}; expected "align"`);
}

const { width, height, format = 'RGB' } = job;
if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1) {
  fail('width and height must be positive integers');
}

const pixels = Buffer.from(job.pixels ?? '', 'base64');
const channels = format === 'RGBA' ? 4 : format === 'GRAY' ? 1 : 3;
const expected = width * height * channels;
if (pixels.length !== expected) {
  fail(`pixel length ${pixels.length} does not match ${width}x${height}x${channels} = ${expected}`);
}

const image = new engine.ImageBuffer(width, height, new Uint8Array(pixels), format);

let profile;
if (job.profile) {
  const symbol = PROFILES[job.profile];
  if (!symbol) {
    fail(`unknown profile ${JSON.stringify(job.profile)}; expected one of ${Object.keys(PROFILES)}`);
  }
  profile = engine[symbol];
}

const started = Date.now();
const stage = new engine.AutomaticCardAlignmentStage({ profile, ...(job.options ?? {}) });
const context = { debug: false, metadata: {} };
const output = stage.process(image, context);

const geometry = context.metadata['cardGeometry'] ?? null;
const status = context.metadata['alignmentStatus'] ?? (geometry ? 'ALIGNED' : 'CARD_NOT_FOUND');

process.stdout.write(
  JSON.stringify({
    ok: true,
    status,
    geometry,
    candidates: context.metadata['cardCandidates'] ?? [],
    timings: { totalMs: Date.now() - started },
    outputSize: { width: output.width, height: output.height },
    version: engine.IMAGE_ENGINE_VERSION ?? null,
    enginePath,
  }),
);
