# Dataset card — synthetic card captures

## What this dataset is

Card images produced by simulating **camera captures of physical cards**, together
with automatically generated ground truth for every card field.

Each sample composes:

| Stage | What is simulated | Why it matters |
|---|---|---|
| Geometry | pitch, yaw, roll, distance, edge warp, cropping | real photos are never taken square-on |
| Materials | card finish (matte, holo, foil), wear, scratches | holos reflect the room; matte cards do not |
| Lighting | key/fill lights, colour temperature, vignetting, unevenness, exposure | the same card looks different in a kitchen and a lightbox |
| Optics | lens distortion, chromatic aberration, defocus, motion blur | phone lenses are imperfect and hands move |
| Sensor | shot noise, read noise, chroma noise, ISO, resolution | small sensors are noisy, especially in low light |
| Scene | background, cast shadows, occluders (fingers, sleeves, other cards) | people hold cards, and their hands get in the way |
| Encoding | JPEG quality, double compression, resizing | uploads are recompressed at least once |

## What this dataset is **not**

- **Not photographs.** It is a simulation. It will not reproduce every failure
  mode a real capture can produce, and the simulation's fidelity is bounded by
  the models it uses.
- **Not real card data.** Demo source cards carry invented text per game
  (`src/cardvault_vision/demo.py`). No copyrighted card art or real card
  database is included.
- **Not a substitute for evaluation on real uploads.** The honest test of this
  work is a held-out set of genuine CardVault captures. Until that exists, all
  reported numbers are numbers about the *simulator*.

## Composition

The `realistic` profile mix deliberately mirrors a plausible upload
distribution rather than a uniform one, because a uniform mix would over-train
on conditions that rarely occur:

| Profile | Share | Character |
|---|---|---|
| `phone_normal` | 55% | mid-range phone, decent light, mild perspective |
| `phone_bad_lighting` | 20% | indoor tungsten, high ISO, noise, soft focus |
| `phone_low_end` | 12% | small sensor, strong sharpening, heavy compression |
| `scanner` | 8% | flatbed, high resolution, minimal distortion |
| `child_camera` | 5% | very unsteady, close cropping, extreme angles |

Degradations are targeted rather than uniform: `problem_rate` controls how often
a named problem (glare, blur, low light, perspective, occlusion, compression) is
applied on purpose, so there are always enough hard examples to learn from.

## Quality control

A sample a human could not read is not a valid label, so it is rejected rather
than written with ground truth nobody could have produced. Rejection is
recorded in the generation report (not silently dropped):

- minimum sharpness
- maximum clipped / crushed pixel fraction
- minimum card visibility in frame
- minimum per-field legibility
- maximum per-field occlusion

## Provenance

Every image records: seed, index, resolved profile, per-stream RNG seeds,
geometry, camera, lighting, surface and artefact parameters, and the config
fingerprint. Any single image can be regenerated without regenerating the
dataset in order.

The manifest additionally carries a content hash over `(path, sha256)` pairs and
the config fingerprint, so a dataset is identified by what is in it, not by when
it was built.

## Leakage policy

Splits are assigned per **card id**. A card appearing in two photographs cannot
have those photographs split across train and test. Near-duplicate images are
detected by perceptual hash and grouped for the same reason.

## Known limitations

1. The occlusion and background vocabularies are small. Real clutter is more
   varied than the generator's object set.
2. The simulator has no model of card sleeves' specular behaviour at grazing
   angles beyond a simple haze plus band approximation.
3. Real cards have physical wear patterns (edge whitening, creases) that are
   only approximated.
4. Language coverage is limited to the characters in `universal_charset()`.
