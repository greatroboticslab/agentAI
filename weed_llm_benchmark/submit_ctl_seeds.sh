#!/bin/bash
# Put an error bar on the warm-start gap by repeating BOTH arms over seeds.
#
# The gap the poster quotes -- fresh start 0.58053 against warm start 0.55180 on
# round 15's own data, same 30 epochs and same complete cosine both sides -- is
# one run per arm. Its sigma is borrowed from a different recipe's seed spread
# (the round recipe, 0.0040 over three seeds). Repeating each arm at seeds 102
# and 103 gives the gap a spread measured on the arms themselves.
#
# Walltime: the seed-101 pair finished in 6:08 and 6:10, so 8 h carries 30%
# headroom. The earlier submissions asked for 12 h and sat in a 2,900-deep
# GPU-shared queue with no start estimate; nothing backfills a 12 h hole.
#
# CTL_SEED is passed explicitly. run_ctl_chain_seed.sh defaults it to 101, so a
# submission that forgets it silently repeats the seed already measured and the
# whole job buys nothing -- the output file name carries the seed, which is the
# only reason that would ever be noticed.
set -uo pipefail
cd "$(dirname "$0")" || exit 1

for SEED in 102 103; do
  CTL_SEED="$SEED" sbatch --export=ALL,CTL_SEED="$SEED" \
      --job-name="ctlS$SEED" --time=08:00:00 --array=1-2 run_ctl_chain_seed.sh
done
