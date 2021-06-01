#!/usr/local/bin/python3.7
print('qweqwe')
path='/data01/zyh/ALDataset/BITVehicle_Dataset/'
dataset='coco'

python cald_train.py -p $path --dataset $dataset
# cd到data01/gpl/ALDataset/BITxxx  python utils.py
# /data01/gpl/ALDataset/BITVehicle_Dataset/utils.sh
# tmux new -s <session-name>
# tmux detach
# tmux ls
# Ctrl+b d
# tmux attach -t <session-name>
# tmux attach -t cald_python
# nvidia-smi
# cp -R dir1 dir2
# tmux capture-pane -S -5557 -E -642 -t 3
# tmux save-buffer output.log
# tmux capture-pane -S -
# show-buffer   显示当前缓存区内容
# capture-pane  捕捉指定面板的可视内容并复制到一个新的缓存区
# list-buffers  列出所有的粘贴缓存区
# choose-buffer 显示粘贴缓存区并粘贴选择的缓存区内的内容
# save-buffer [filename]    保存缓存区的内容到指定文件里
# scp -r czz@192.168.8.123:/home/czz/AL/CALD/output.log /d/Dian/download-from-43005
# git push origin main