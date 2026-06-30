#!/bin/bash

if [[ -e MeshCNN ]]; then
  echo "MeshCNN directory already exists. Exiting."
  exit
fi

# 1. Get MeshCNN
git clone https://github.com/ranahanocka/MeshCNN.git
cd MeshCNN

# 2. Update environment.yml directly (Avoids fragile git diffs)
cat << 'EOF' > environment.yml
name: meshcnn
channels:
  - pytorch
  - defaults
dependencies:
  - python=3.9
  - cython
  - pytorch=1.12.1
  - torchvision=0.13.1
  - torchaudio=0.12.1
  - numpy
  - matplotlib
  - pip
  - pip:
    - git+https://github.com/lanpa/tensorboardX.git
    - pytest
EOF

# 3. Patch the python source code files
git apply --ignore-whitespace << 'MyEOF'
diff --git a/models/layers/mesh_pool.py b/models/layers/mesh_pool.py
index 394d0fc..1fa9a2f 100644
--- a/models/layers/mesh_pool.py
+++ b/models/layers/mesh_pool.py
@@ -44,7 +44,7 @@ class MeshPool(nn.Module):
         # recycle = []
         # last_queue_len = len(queue)
         last_count = mesh.edges_count + 1
-        mask = np.ones(mesh.edges_count, dtype=np.bool)
+        mask = np.ones(mesh.edges_count, dtype=np.bool_)
         edge_groups = MeshUnion(mesh.edges_count, self.__fe.device)
         while mesh.edges_count > self.__out_target:
             value, edge_id = heappop(queue)
diff --git a/models/layers/mesh_prepare.py b/models/layers/mesh_prepare.py
index 47e827c..0167bd0 100644
--- a/models/layers/mesh_prepare.py
+++ b/models/layers/mesh_prepare.py
@@ -159,7 +159,7 @@ def build_gemm(mesh, faces, face_areas):
     mesh.sides = np.array(sides, dtype=np.int64)
     mesh.edges_count = edges_count
     mesh.edge_areas = np.array(mesh.edge_areas, dtype=np.float32) / np.sum(face_areas) #todo whats the difference between edge_areas and edge_lenghts?
-
+    mesh.ve = np.array(mesh.ve, dtype=list)  # convert to array of lists

 def compute_face_normals_and_areas(mesh, faces):
     face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
diff --git a/util/writer.py b/util/writer.py
index 6c0b475..9c45266 100644
--- a/util/writer.py
+++ b/util/writer.py
@@ -52,10 +52,10 @@ class Writer:
             for name, param in model.net.named_parameters():
                 self.display.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

-    def print_acc(self, epoch, acc):
+    def print_acc(self, epoch, acc, phase='TEST'):
         """ prints test accuracy to terminal / file """
-        message = 'epoch: {}, TEST ACC: [{:.5} %]\n' \
-            .format(epoch, acc * 100)
+        message = ('epoch: {}, {} ACC: [{:.5} %]\n'
+                   .format(epoch, phase, acc * 100))
         print(message)
         with open(self.testacc_log, "a") as log_file:
             log_file.write('%s\n' % message)
MyEOF

echo "MeshCNN is ready. Create conda environment with:"
echo "conda env create -f environment.yml"
