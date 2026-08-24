# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.7] - 2026-08-24

### Fixed

- **Order Blocks:** Expose and forward `break_mode="close"` through `detect_order_blocks` and `order_blocks_polars` to `detect_structure`, preserving default `"close"`. Prior `order_blocks` recomputed structure hard-coded to `close`, so `wick`/`both` were silently ignored.
- **Swings:** Narrowly suppress expected `RuntimeWarning: All-NaN slice encountered` inside `nanmax`/`nanmin` window reductions. All-NaN windows remain semantically invalidated via `high_valid`/`low_valid` masks; warning suppression is scoped only to the known reduction.

### Added

- **Tests:** Deterministic regression test for `break_mode` wick-vs-close divergence in Order Blocks.

### Changed

- **Documentation:** No public API order changes. `swings_polars` output column order intentionally preserved.
- **Contact:** Package metadata email updated to `rafikhairan120@gmail.com`.

*Note: This is a stabilization patch on the `0.3.x` line (`tie="all"` remains the default). No `tie` default change, no P2 swing caching, no OB/liquidity optimization.*

## [0.3.6] - 2026-08-24

### Added

- **Swings:** `tie` plateau semantics `all`/`first`/`strict` with prominence and flat-market handling. `first` collapses equal-price plateaus to first representative.
- **Structure/Liquidity/Zones:** `tie` forwarding and `swing_high`/`swing_low` injection support for composition (`detect_structure`, `detect_liquidity`, `detect_zones`).

### Changed

- **Polars:** `swings_polars`, `structure_polars`, `liquidity_polars`, `zones_polars`, `order_blocks_polars`, `add_smc_columns` now expose `tie`, `break_mode`, `zone_mode` where applicable.
- **Packaging:** License metadata fixed to SPDX `MIT`, badge and documentation updates.

## [0.3.4] - 2026-08-24

### Added

- **Tests:** Downstream validation for `tie="first"` vs `"all"` on flat, plateau, double-top/bottom, isolated, random, and NaN-adjacent datasets.
- **Benchmarks:** Downstream tie impact report.

### Changed

- **Documentation:** Clarified `tie` semantics and downstream impact.

## [0.3.3] - 2026-08-24

### Added

- **Swings:** `tie` parameter initial implementation.

## [0.3.2] - 2026-08-23

### Added

- **FVG:** `ce_level`, `mitigated_50`, `mitigated_full`, `inverted` (IFVG) with Numba single-scan mitigation.
- **Structure:** `break_mode` (`close`/`wick`/`both`) and first-cross suppression.
- **Order Blocks:** `zone_mode` (`full`/`body`/`mean_threshold`) and `is_breaker`/`breaker_index`.
- **Liquidity:** `equal_swing_high`/`equal_swing_low` via swing comparison.
- **Zones:** `ote_high`/`ote_low`/`ote_705`/`in_ote` (0.618/0.705/0.786).
- **Polars:** `LazyFrame` native for `swings`/`fvg` with honest fallback documentation.

### Fixed

- **Packaging:** License classifier handling.

## [0.3.1] - 2026-08-23

### Fixed

- **Packaging:** License handling, `ruff` clean, per-gap mitigation memory fix, `structure` Numba cache.

### Added

- **Benchmarks:** `1k`/`10k`/`100k`/`500k` report.

## [0.3.0] - 2026-08-23

### Added

- **Liquidity:** Sweeps and equal highs/lows.
- **Zones:** Premium/discount and OTE.
- **FVG:** CE 50% and IFVG.
- **Structure:** BOS first-cross and `break_mode`.
- **Order Blocks:** `zone_mode` and breaker block.

## [0.2.0] - 2026-08-23

### Added

- Liquidity sweeps, equal highs/lows, premium/discount zones, OB mitigation, `pysmc` → `pyvsmc` shim.

## [0.1.0] - 2026-08-23

### Added

- Initial release: `detect_swings`, `detect_fvg`, `detect_structure`, `detect_order_blocks` with NumPy/Polars and `pyvsmc` namespace.

## [Unreleased]

- No planned changes. Stabilization on `0.3.x` continues; `tie="first"` remains opt-in.
