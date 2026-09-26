# "object category/object id/camera distance" -> "scissor/11100/4"
for name in "scissor/11100/4"; do
for i in {0..199}; do
for j in train test; do
echo "Running with i=$i, j=$j";
python blender_cam2.py "$name" $i $j;
done;
done;
done

for name in "scissor/11100"; do
for i in {0..199}; do
for j in train test;  # j 的范围，比如 A, B, C
do
python blender_depth2.py "$name" $i $j;
done;
done;
done

