#!/bin/bash
# change the loop value according to num_clients in config.yaml file
for i in `seq 0 2`; do
    echo "Starting Agent : $i "
    python dqnrun.py --config="config.yaml" --vehicle=${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait