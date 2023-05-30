#!/bin/bash

# 定义待合并的 .json 文件路径
json_files=(
    "output/deit_tiny_patch16_224/A_1_50_epochs/pred_all.json"
    "output/deit_tiny_patch16_224/A_2_50_epochs/pred_all.json"
    "output/deit_tiny_patch16_224/A_3_50_epochs/pred_all.json"
    "output/deit_tiny_patch16_224/A_4_50_epochs/pred_all.json"
    "output/deit_tiny_patch16_224/A_5_50_epochs/pred_all.json"
)

# 初始化最大的 n_parameters
max_n_parameters=-1

# 循环处理每个 .json 文件
for file_path in "${json_files[@]}"; do
    # 从当前文件中获取 n_parameters 的值
    n_parameters=$(cat $file_path | jq '.n_parameters')

    # 如果当前文件的 n_parameters 值比最大值要大，则更新最大值
    if [ $n_parameters -gt $max_n_parameters ]; then
        max_n_parameters=$n_parameters
    fi

    # 把当前文件的内容追加到最终的 .json 文件中
    cat $file_path >> merged.json
done

# 把最大的 n_parameters 值更新到最终的 .json 文件中
jq ".n_parameters = $max_n_parameters" merged.json > merged_temp.json
mv merged_temp.json merged.json