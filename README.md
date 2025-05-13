#!/bin/bash

# Step 1: 生成 initial mesh（直接导出 .obj）
PYTHONPATH=. python scripts/process_data/convex_hull.py \
  --i test/chip.ply \
  --faces 3000 \
  --o test/init_chip.obj \
  --manifold-path ./Manifold/build

echo "== Step 2: Run Point2Mesh reconstruction =="

# Step 2: 跑 Point2Mesh 重建
python main.py \
  --input-pc test/chip.ply \
  --initial-mesh test/init_chip.obj \
  --save-path output/chip \
  --iterations 3000 \
  --upsamp 1

echo "== Done! Final mesh saved to output/chip/final_mesh.obj =="