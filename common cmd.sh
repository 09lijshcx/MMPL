nohup python3 -u train.py > ./log/9_26.log 2>&1 &
nohup python3 -u preprocess.py > ./log/10_10_preprocess_logd.log 2>&1 &
nohup python3 -u herg_multi_prompts_train.py > ./log/10_10_mtrain.log 2>&1 &


sudo apt-get install psmisc
fuser -v /dev/nvidia*

sudo apt-get install htop

source ~/.bashrc

nohup torchrun moltex_distributed_train.py > /home/jovyan/prompts_learning/log/10_24_text_pre.log 2>&1 &
nohup python3 -u moltex_train.py > ./log/10_24_pre.log 2>&1 &