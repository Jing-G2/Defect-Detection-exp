#!/bin/bash
export result_path='/nfs4-p1/gj/DEFECT2022/results/'

# export model_name='GAT'
export model_name='GCN'
# export model_name='SAGE'

# export data_name='NEU-CLS'
# export in_channels=4
# export num_classes=6
# export num_epochs=500

export data_name='NEU-CLS-64'
export in_channels=4
export num_classes=9


export exp_name=${model_name}'_gc_'${data_name}
export model_dir=${result_path}${exp_name}'/models'
export data_dir='/nfs4-p1/gj/DEFECT2022/data'
export log_dir='/nfs4-p1/gj/DEFECT2022/runs/'${exp_name}
export device_index='1'

python test_gc.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --in_channels ${in_channels} \
  --num_classes ${num_classes} \
  --model_dir ${model_dir} \
  --data_dir ${data_dir} \
  --log_dir ${log_dir} \
  --device_index ${device_index}
