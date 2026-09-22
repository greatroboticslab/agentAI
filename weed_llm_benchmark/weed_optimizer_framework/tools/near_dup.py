"""When two images are the same photograph, by 64-bit dHash.

A re-exported copy -- Roboflow resize, JPEG re-encode -- moves a few bits of the
hash, so an exact-match check lets it through. On the lab registry (2026-09-21)
422 harvested images sat 1-3 bits from a sealed cwd12 holdout image, every one
from a dataset that contains cwd12 photographs, and all past the exact guard;
cwd12's own train and test photos are more than three bits apart in all but 2
of 3,671 cases.

Where it must NOT be used: a 64-bit dHash is a 9x8 thumbnail, and images that
share a large uniform region -- black letterbox padding, one studio wall --
land within 3 bits while showing different plants (checked by eye on the lab
registry: Roboflow exports padded to a black square, potted plants in front of
the same wall). So the near test protects natural field photographs we must
not train on twice or at all (the holdout, the verified cwd12 copies); general
cross-dataset dedup stays exact.

The sealed holdout gets a wider threshold, HOLDOUT_NEAR_DUP_BITS (v3.60.0).
A registry-wide lab scan found two Roboflow re-exports of holdout photographs
4 and 5 bits from their original (rf_zig-zag Morningglory_891, 4 bits from
20210820_iPhoneSE_YL_1480; rf_karthikeya SpottedSpurge_86, 5 bits from
20210806_iPhoneSE_YL_353), both 15+ bits from every train photograph. Only 4
of cwd12's 3,671 train photographs sit 5-6 bits from a holdout image, so the
wider test costs almost nothing, and it errs toward not training.

No third-party imports.
"""

NEAR_DUP_BITS = 3
HOLDOUT_NEAR_DUP_BITS = 6

# Owners that denote a sealed holdout image: mega_trainer's
# HOLDOUT_HASH_SENTINEL (alone or as (sentinel, image)), and the "holdout"
# kind of merge_roboflow_projects.cwd12_photo_index / synth_cutpaste's guard.
_HOLDOUT_OWNERS = frozenset({"__HOLDOUT__", "holdout"})


def is_holdout_owner(who):
    if isinstance(who, tuple):
        who = who[0] if who else None
    return isinstance(who, str) and who in _HOLDOUT_OWNERS


def _blocks(nbits):
    """(shift, mask) of nbits+1 disjoint blocks covering 64 bits: two hashes at
    most nbits apart agree exactly on at least one block (pigeonhole)."""
    n = nbits + 1
    widths = [64 // n + (1 if i < 64 % n else 0) for i in range(n)]
    out, shift = [], 0
    for w in widths:
        out.append((shift, (1 << w) - 1))
        shift += w
    return tuple(out)


class NearHashIndex:
    """dHash -> first owner, answering "is a stored hash within range?".

    The range is NEAR_DUP_BITS, or HOLDOUT_NEAR_DUP_BITS for a holdout owner
    (is_holdout_owner) or an explicit add(..., max_bits=). Pigeonhole: two
    64-bit hashes that differ in at most 3 bits agree exactly on at least one
    of their four 16-bit blocks, so a lookup only compares against hashes
    sharing a block with it; wide-range hashes (the ~2,000 holdout images) sit
    in a second, 7-block index."""

    _BLOCKS = _blocks(NEAR_DUP_BITS)
    _WIDE_BLOCKS = _blocks(HOLDOUT_NEAR_DUP_BITS)

    def __init__(self):
        self.owner = {}
        self._max = {}
        self._by_block = [dict() for _ in self._BLOCKS]
        self._wide_by_block = [dict() for _ in self._WIDE_BLOCKS]

    def add(self, h, who, max_bits=None):
        if h in self.owner:
            return
        if max_bits is None:
            max_bits = HOLDOUT_NEAR_DUP_BITS if is_holdout_owner(who) else NEAR_DUP_BITS
        if not 0 <= max_bits <= HOLDOUT_NEAR_DUP_BITS:
            raise ValueError("max_bits must be 0..%d" % HOLDOUT_NEAR_DUP_BITS)
        self.owner[h] = who
        self._max[h] = max_bits
        blocks, table = ((self._WIDE_BLOCKS, self._wide_by_block)
                         if max_bits > NEAR_DUP_BITS else (self._BLOCKS, self._by_block))
        for (shift, mask), t in zip(blocks, table):
            t.setdefault((h >> shift) & mask, []).append(h)

    def _near(self, h):
        """{stored hash: bits} within each stored hash's own range."""
        found = {}
        if h in self.owner:
            found[h] = 0
        for blocks, table in ((self._BLOCKS, self._by_block),
                              (self._WIDE_BLOCKS, self._wide_by_block)):
            for (shift, mask), t in zip(blocks, table):
                for c in t.get((h >> shift) & mask, ()):
                    if c not in found:
                        d = bin(c ^ h).count("1")
                        if d <= self._max[c]:
                            found[c] = d
        return found

    def find(self, h):
        """(owner, bits) of the nearest stored hash within range, else None."""
        found = self._near(h)
        if not found:
            return None
        c = min(found, key=lambda k: found[k])
        return self.owner[c], found[c]

    def matches(self, h):
        """Every (owner, bits) stored within range of h, nearest first."""
        return sorted(((self.owner[c], d) for c, d in self._near(h).items()),
                      key=lambda t: t[1])

    def __len__(self):
        return len(self.owner)
