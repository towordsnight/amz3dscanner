#!/bin/bash

echo "== Step 1: Generate convex hull as initial mesh (.off) =="
PYTHONPATH=. python scripts/process_data/convex_hull.py \
  --i test/chip.ply \
  --faces 3000 \
  --o test/init_chip.off

echo "== Step 2: Convert .off to .obj using trimesh =="
python -c "import trimesh; m=trimesh.load('test/init_chip.off'); m.export('test/init_chip.obj')"

echo "== Step 3: Run Point2Mesh reconstruction =="
python main.py \
  --input-pc test/chip.ply \
  --initial-mesh test/init_chip.obj \
  --save-path output/chip \
  --iterations 3000 \
  --upsamp 1

echo "== Done! Final mesh saved to output/chip/final_mesh.obj =="
