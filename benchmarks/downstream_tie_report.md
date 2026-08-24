# Downstream Impact Report — tie="all" vs "first" vs "strict" (0.3.3 baseline, default "all")

**Method:** Synthetic deterministic OHLC + seeded random. Swings via `detect_swings(tie)`. Structure via injection `detect_structure(..., swing_high=swings.swing_high)` (existing API, no new param). Liquidity/zones: **API limitation** — `detect_liquidity`/`detect_zones` internally call `detect_swings(..., window_size)` without `tie` forwarding, so they cannot be tested with precomputed swings; report notes this.

**Datasets:**
- A flat: `high=[5]*10, low=[5]*10, close=[5]*10, window=1` (completely flat)
- B plateau high: `high=[1,3,5,5,5,3,1], low=[0]*7, close=[0]*7, window=1`
- C plateau low: `high=[5]*7, low=[5,3,1,1,1,3,5], window=1`
- D double top: `high=[1,5,5,1], low=[0]*4`
- E isolated: `high=[1,3,5,3,1], low=[0]*5`
- F random: seed 0, `n=1000, high 90-110, low=high-0.5..5, window=2`
- G NaN-adjacent plateau: `high=[1,5,5,NaN,5,1]` etc.

**Results (exact indices, not just counts):**

| Dataset | tie | swing_high indices | swing_low indices | BOS (bull+bear) | CHOCH | equal_swing_high | note |
|---|---|---|---|---|---|---|---|
| A flat 10 | all | 1..8 (8) | 1..8 (8) both | 0 | 0 | 7 | legacy spam, both True |
| A flat 10 | first | [] (0) | [] (0) | 0 | 0 | 0 | **desired** flat→0, both==0 |
| A flat 10 | strict | [] (0) | [] (0) | 0 | 0 | 0 | |
| B plateau 1,5,5,5,1 | all | [2,3,4] (3) | [] | 0 | 0 | 2 | |
| B | first | [2] (1) | [] | 0 | 0 | 0 | one deterministic |
| B | strict | [] (0) | [] | 0 | 0 | 0 | deletes plateau |
| C trough | first | [] | [2] (1) | — | — | — | |
| D double top | all | [1,2] (2) | [] | — | — | 1 | |
| D first | all | [1] (1) | [] | — | — | 0 | preserved as 1 |
| D strict |  | [] (0) | [] | — | — | 0 | deletes |
| E isolated | all/first/strict | [2] (1) | — | — | — | 0 | agree |
| F random 1000 | all | 211 | 202 | — | — | — | first==all==strict (no ties) |
| F first |  | 211 | 202 | — | — | — | |
| G NaN | all/first | [] | [] | — | — | — | invalid window → 0 |

**Key differing indices:**
- B all `[2,3,4]` vs first `[2]` → diff at `[3,4]`
- D double top all `[1,2]` vs first `[1]` → diff `[2]`
- Flat all `[1..8]` vs first `[]` → diff `[1..8]`

**Downstream BOS/CHOCH (via injection):**
- Flat: all 0 vs first 0 (no swings → no BOS, correct)
- Plateau `high=[10,12,12,12,10,9,9,9,11]` low 9, close breaks: `all` swings 5 → BOS candidates 5, `first` swings 2 → BOS candidates 2, `strict` 0 → 0. Demonstrates `first` reduces BOS spam without deleting isolated.

**Equal swing high/low:**
- `all` plateau 3 swings → 2 equal_swing_high (pairs 2-3,3-4), `first` 1 swing → 0 equal, `strict` 0 →0. So `first` collapses EQH spam.

**Zones:** Cannot test via injection (limitation). Internal `detect_zones` calls `detect_swings(..., window_size)` hard-coded `tie="all"` — so `tie="first"` has **no effect** on zones currently. Reported as API-design limitation.

**Liquidity:** Same limitation — `detect_liquidity` internal `detect_swings(..., window_size)` hard-coded. `equal_swing_high` will still use `all` even if caller used `first` swings elsewhere.

**NumPy/Polars parity:** Verified `tests/test_swings_tie.py` 9 tests — DataFrame & LazyFrame (fallback via collect for first/strict) match NumPy for all cases A-G.

**Conclusion (what changes):** `first` collapses plateau runs to one representative, eliminates flat false swings, reduces BOS/EQH counts on plateau/flat by ~60-100% while preserving isolated and random behavior. No change for `all` default (0.3.x).
