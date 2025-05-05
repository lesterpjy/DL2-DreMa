#!/bin/bash
#SBATCH -p cbuild
#SBATCH -t 02:00:00
#SBATCH --job-name=move_data
#SBATCH --output=work/move_data_%A.out

set -euo pipefail

SRC="/scratch-shared/tmp.lUdVGE8VOd/rlbench/train/slide_block_to_color_target/all_variations/episodes"
DST="/scratch-shared/tmp.lUdVGE8VOd/mix_data/slide_block_to_color_target/all_variations/episodes"

# Make sure destination exists
mkdir -p "$DST"

echo '--- final GPU check ---'

for src_ep in "$SRC"/episode*; do
  epname=$(basename "$src_ep")
  dest_ep="$DST/$epname"

  if [ -d "$dest_ep" ]; then
    # Conflict: compute a unique name (episode17 → episode17_orig, _orig_1, etc.)
    base="${epname}_orig"
    newname="$base"
    counter=1
    while [ -e "$DST/$newname" ]; do
      newname="${base}_$counter"
      ((counter++))
    done
    echo "Conflict: $epname → moving as $newname"
    mv "$src_ep" "$DST/$newname"
  else
    echo "Moving: $epname"
    mv "$src_ep" "$dest_ep"
  fi
done
