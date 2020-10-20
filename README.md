python train.py --model_name PSPNet --lstm False --noise False --data_dir

python train.py --model_name PSPNet --lstm True --use_pre True --noise False --data_dir

python train.py --model_name PSPNet --lstm True --use_pre True --noise False --noise_type extra --noise_ratio 50 --data_dir --data_extra
