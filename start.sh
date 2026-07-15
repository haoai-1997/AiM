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
