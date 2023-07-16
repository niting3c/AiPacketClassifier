#   This is the most basic QSUB file needed for this cluster.
#   Further examples can be found under /share/apps/examples
#   Most software is NOT in your PATH but under /share/apps
#
#   For further info please read http://hpc.cs.ucl.ac.uk
#   For cluster help email cluster-support@cs.ucl.ac.uk
#
#   NOTE hash dollar is a scheduler directive not a comment.


# These are flags you must include - Two memory and one runtime.
# Runtime is either seconds or hours:min:sec

#$ -l tmem=30G
#$ -l h_rt=48:00:00
#$ -l gpu=true

#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N Snoozy

#The code you want to run now goes here.

cd ./AiPacketClassifier
git pull
mkdir -p ~/.cache/huggingface
echo "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%" > ~/.cache/huggingface/token

conda activate AiPacketClassifier
python3 run.py
git add .
git add output