#!/bin/bash
export result_path='/nfs4-p1/gj/DEFECT2022/results/'

# export model_name='GAT'
export model_name='GCN'
# export model_name='SAGE'


# export data_name='NEU-CLS'
# export in_channels=4
# export n_class=6
# export epochs=500

# export data_name='NEU-CLS-64'
# export in_channels=4
# export hidden_channels=16
# export n_class=9
# export epochs=200

# export data_name='KSDD'

export data_name='CrackForest'
export in_channels=4
# export hidden_channels=16
export n_class=2
export epochs=200

export seed=100
export batch_size=32

export exp_name=${model_name}'_gc_'${data_name}
export model_dir=${result_path}${exp_name}'/models'
export data_dir='/nfs4-p1/gj/DEFECT2022/data'
export log_dir='/nfs4-p1/gj/DEFECT2022/runs/'${exp_name}
export device_index='1'

python train_gc.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --seed ${seed} \
  --batch_size ${batch_size} \
  --epochs ${epochs} \
  --in_channels ${in_channels} \
  --n_class ${n_class} \
  --model_dir ${model_dir} \
  --data_dir ${data_dir} \
  --log_dir ${log_dir} \
  --device_index ${device_index} \
  >> /nfs4-p1/gj/DEFECT2022/log/${model_name}/${exp_name}_train.log
