#!/bin/bash

vastai copy "$(vastai show instances  | tail -n1 | awk '{print $1}'):/workspace/metta/train_dir/$1" ./train_dir/
