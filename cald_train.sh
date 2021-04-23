
path='/data01/zxl/KeyVehicleDetection/dataset/BITVehicle_Dataset/'
dataset='coco'

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
	python cald_train.py -p $path --dataset=$dataset


