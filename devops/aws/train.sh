#/bin/bash -e

 ./trainers/a100_100x100_simple.sh --num_workers=10 --batch_size=16384 "$@"

