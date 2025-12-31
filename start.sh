for num in {0..9}; do
for i in "fridge/10905"; do
    left=$(echo "$i" | awk -F'[/_]' '{print $1}')
    right=$(echo "$i" | awk -F'[/_]' '{print $2}')
        echo "Running: $left / $right / $num "
        python train_main.py \
            --source_path ../AIM_dataset_200_Final/paris/$left/$right/ \
            --model_path output_final/paris/$num/test_${left}_${right}/ \
            --is_blender --eval --random_bg_color
    done
done

for num in {0..9}; do
for i in "fridge/10905" ; do
    # 用awk把前后分开（支持 / 和 _ 分隔符）
    left=$(echo "$i" | awk -F'[/_]' '{print $1}')
    right=$(echo "$i" | awk -F'[/_]' '{print $2}')
        echo "Running: $left / $right / $num "
        python seg_main.py \
            --source_path ../AIM_dataset_OPEN/paris/$left/$right/ \
            --model_path output_open/paris/$num/test_${left}_${right}/ \
            --is_blender --eval --random_bg_color
    done
done

for num in {0..9}; do
for i in "fridge/10905"; do
    # 用awk把前后分开（支持 / 和 _ 分隔符）
    left=$(echo "$i" | awk -F'[/_]' '{print $1}')
    right=$(echo "$i" | awk -F'[/_]' '{print $2}')
        echo "Running: $left / $right / $num "
        python render_main.py \
            --source_path ../AIM_dataset_200/paris/$left/$right/ \
            --model_path output/paris/$num/test_${left}_${right}/ \
            --is_blender --eval --random_bg_color
    done
done
