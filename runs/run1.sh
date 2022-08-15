python3 -u fisher_overlap/overlap_task_evec.py --dataset cifar10 --model_name all_cnn --model_init_path runtime_generated/all_cnn_10_96_144_55000_0_cifar10/tensors/model_init.pt \
--model_fin_path runtime_generated/all_cnn_10_96_144_55000_0_cifar10/tensors/model.pt --dest_path runtime_generated/all_cnn_10_96_144_55000_0_cifar10/tensors/ \
--window_size 30 --tasks 0,1 2,3 4,5 6,7 8,9
