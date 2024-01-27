#!/bin/bash -e

label=$1
get_info=$(vastai show instances --raw)
host=$(echo $get_info | jq -r ".[] | select(.label==\"$label\") | .ssh_host")
port=$(echo $get_info | jq -r ".[] | select(.label==\"$label\") | .ssh_port")
ssh_cmd="ssh -p $port root@$host"
echo $ssh_cmd
$ssh_cmd
