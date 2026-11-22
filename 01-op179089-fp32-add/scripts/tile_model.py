#!/usr/bin/env python3
"""Enumerate the tiles AddKernel::Run executes, to pick and interpret memcheck cases.

This is a faithful Python mirror of the partition/tile arithmetic in op179089.asc
(AddKernel::Run).  It exists because the local CPU simulation in tests/mock_headers
models DataCopy/Add as byte-exact memcpy loops, so it cannot show how a tile maps onto
32 B DMA granules and 256 B vector repeats -- which is what the instrumented Atlas A3
runs are reporting on.

Property P ("partial repeat"): a tile whose element count is >= 64 and not a multiple
of 64, i.e. a multi-repeat vector op whose final 256 B repeat is partial.  The Add for
such a tile covers lanes that the preceding DataCopy/DataCopyPad did not fill.
P separates every instrumented A3 result recorded so far:

    N=1, 7, 9        no P    memcheck clean
    N=8193, 40961    has P   memcheck reports UB access errors

Usage:
    python3 scripts/tile_model.py                 # scan the suite's cases
    python3 scripts/tile_model.py 8193 40961      # detail specific lengths
    python3 scripts/tile_model.py --candidate 8193
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

# Mirrors the constants in op179089.asc.
DMA_ALIGNMENT_ELEMENTS = 8        # kDmaAlignmentElements: 32 B / sizeof(float)
TILE_ELEMENTS = 4096              # kTileElements
ELEMENTS_PER_BLOCK = 1024         # kElementsPerBlock
MAX_BLOCKS = 64                   # kMaxBlocks
VECTOR_REPEAT_ELEMENTS = 64       # 256 B vector repeat / sizeof(float)

Tile = Tuple[int, int, int, int, str]  # (block, gm_start, offset, count, path)


def tiles(length: int, core_limit: int = 48, candidate: bool = False) -> Tuple[int, List[Tile]]:
    """Return (blocks, tiles) for one launch.

    candidate=False mirrors the shipped op179089.asc.
    candidate=True mirrors reports/candidates/op179089-tail-repeat-split.asc, which cuts
    each block tail on the vector-repeat boundary.
    """
    blocks = min(core_limit, (length + ELEMENTS_PER_BLOCK - 1) // ELEMENTS_PER_BLOCK)
    blocks = min(blocks, MAX_BLOCKS)
    groups = (length + DMA_ALIGNMENT_ELEMENTS - 1) // DMA_ALIGNMENT_ELEMENTS
    groups_per_block, extra_groups = divmod(groups, blocks)

    out: List[Tile] = []
    for block in range(blocks):
        preceding_extras = block if block < extra_groups else extra_groups
        start = (block * groups_per_block + preceding_extras) * DMA_ALIGNMENT_ELEMENTS
        end = start + (groups_per_block + (1 if block < extra_groups else 0)) * DMA_ALIGNMENT_ELEMENTS
        stop = min(end, length)
        if start >= stop:
            continue

        block_length = stop - start
        offset = 0
        while offset + TILE_ELEMENTS <= block_length:
            out.append((block, start, offset, TILE_ELEMENTS, "aligned"))
            offset += TILE_ELEMENTS

        tail = block_length - offset
        if candidate:
            whole_repeats = tail - tail % VECTOR_REPEAT_ELEMENTS
            if whole_repeats:
                out.append((block, start, offset, whole_repeats, "aligned"))
                offset += whole_repeats
                tail -= whole_repeats
        if tail:
            path = "aligned" if tail % DMA_ALIGNMENT_ELEMENTS == 0 else "pad"
            out.append((block, start, offset, tail, path))
    return blocks, out


def partial_repeat_tiles(ts: List[Tile]) -> List[Tile]:
    """Tiles with property P: count >= one repeat, but not a whole number of repeats."""
    return [t for t in ts
            if t[3] >= VECTOR_REPEAT_ELEMENTS and t[3] % VECTOR_REPEAT_ELEMENTS]


def check_cover(length: int, ts: List[Tile]) -> None:
    """Assert the tiles cover [0, length) exactly once (no gap, no overlap)."""
    ranges = sorted((start + offset, start + offset + count) for _, start, offset, count, _ in ts)
    assert ranges[0][0] == 0, f"N={length}: first tile starts at {ranges[0][0]}"
    assert ranges[-1][1] == length, f"N={length}: last tile ends at {ranges[-1][1]}"
    for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:]):
        assert prev_end == next_start, f"N={length}: gap/overlap at {prev_end} vs {next_start}"


SUITE_CASES = [
    ("1", 1), ("7", 7), ("8", 8), ("9", 9), ("511", 511), ("512", 512), ("513", 513),
    ("1023", 1023), ("1024", 1024), ("1025", 1025), ("4095", 4095), ("4096", 4096),
    ("4097", 4097), ("8192", 8192), ("8193", 8193), ("40959", 40959), ("40960", 40960),
    ("40961", 40961), ("1048575", 1048575), ("1048576", 1048576), ("4M", 4 * 1024 * 1024),
    ("2x3", 6), ("16x257", 4112), ("32x64x129", 264192), ("4x8x16x33", 16896),
    ("bench 16M", 16 * 1024 * 1024), ("bench 32M", 32 * 1024 * 1024),
]


def detail(length: int, core_limit: int, candidate: bool) -> None:
    blocks, ts = tiles(length, core_limit, candidate)
    check_cover(length, ts)
    bad = partial_repeat_tiles(ts)
    print(f"N={length}  blocks={blocks}  tiles={len(ts)}  partial-repeat tiles={len(bad)}")
    shown = 0
    for block, start, offset, count, path in ts:
        flag = "  <-- partial repeat" if (count >= VECTOR_REPEAT_ELEMENTS
                                          and count % VECTOR_REPEAT_ELEMENTS) else ""
        if shown < 6 or flag:
            print(f"    block={block:<3} gm_start={start:<9} offset={offset:<7} count={count:<6} "
                  f"path={path:<7} repeats={count / VECTOR_REPEAT_ELEMENTS:g}{flag}")
            shown += 1
    if len(ts) > shown:
        print(f"    ... {len(ts) - shown} more tiles")


def scan(core_limit: int) -> int:
    print(f"{'case':<12} {'N':>10} {'blocks':>6} {'tiles':>6} {'P':>4} "
          f"{'cand tiles':>10} {'cand P':>6}")
    for label, length in SUITE_CASES:
        _, cur = tiles(length, core_limit, False)
        blocks, cand = tiles(length, core_limit, True)
        check_cover(length, cur)
        check_cover(length, cand)
        print(f"{label:<12} {length:>10} {blocks:>6} {len(cur):>6} "
              f"{len(partial_repeat_tiles(cur)):>4} {len(cand):>10} "
              f"{len(partial_repeat_tiles(cand)):>6}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lengths", nargs="*", type=int, help="element counts to detail")
    parser.add_argument("--cores", type=int, default=48,
                        help="vector cores reported by the device (default 48 = Atlas A3)")
    parser.add_argument("--candidate", action="store_true",
                        help="model the tail-repeat-split candidate instead of the shipped source")
    args = parser.parse_args()

    if not args.lengths:
        return scan(args.cores)
    for length in args.lengths:
        detail(length, args.cores, args.candidate)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
