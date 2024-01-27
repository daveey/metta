#!/bin/bash -e

label=$1
machine=$(
    vastai search offers num_gpus=1 \
    gpu_name=RTX_4090 rented=False "dph<0.4" -o dph- \
    | tail -n1 \
    | awk '{print $1}')

echo "Reserving $machine with label $label"

cmd="vastai create instance $machine \
   --image daveey/metta:latest \
   --disk 60 \
   --onstart-cmd 'bash' \
   --label $label \
   --ssh --direct \
   --args --ulimit nofile=unlimited --ulimit nproc=unlimited -c 'echo hello; sleep infinity;' \
   "


echo $cmd
$cmd
