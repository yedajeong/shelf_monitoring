#! /bin/bash

for i in {1..12}; do
    sync && sudo -S <<< 'qwe1!' sysctl -w vm.drop_caches=3;
    sleep 5;
done