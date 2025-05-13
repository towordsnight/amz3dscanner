#!/bin/bash

# 设置Manifold路径（根据你目录结构）
MANIFOLD_PATH="./Manifold/build"

# Step 1: 生成初始 .off 网格（Convex Hull）
PYTHONPATH=. python scripts/process_data/convex_hull.py \
    --i test/chip.ply \
    --faces 3000 \
    --o test/init_chip.off \
    --manifold-path $MANIFOLD_PATH

# Step 2: 将 .off 转换为 .obj
echo "== Step 2: Convert .off to .obj using trimesh =="
python -c "import trimesh; m=trimesh.load('test/init_chip.off'); m.export('test/init_chip.obj')"

# Step 3: 运行 Point2Mesh 重建
echo "== Step 3: Run Point2Mesh reconstruction =="
python main.py \
    --input-pc test/chip.ply \
    --initial-mesh test/init_chip.obj \
    --save-path output/chip \
    --iterations 3000 \
    --upsamp 1

echo "== Done! Final mesh saved to output/chip/final_mesh.obj =="