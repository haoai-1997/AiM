for name in "blade1/103706/4.8"; do
for i in {0..199}; do
for j in train test; do
echo "Running with i=$i, j=$j";
python blender_cam.py "$name" $i $j;
done;
done;
done;
for name in "fridge/10905"; do
for i in {0..199}; do
for j in train test;  # j 的范围，比如 A, B, C
do
python blender_depth.py "$name" $i $j;
done;
done;
done

for name in "table/31249/3.5"; do
for i in {0..199}; do
for j in train test; do
echo "Running with i=$i, j=$j";
python blender_cam_m.py "$name" $i $j;
done;
done;
done;
for name in "table/31249"; do
for i in {0..199}; do
for j in train test;  # j 的范围，比如 A, B, C
do
python blender_depth_m.py "$name" $i $j;
done;
done;
done