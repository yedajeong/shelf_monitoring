import os, sys
from time import sleep

# EXPORT = 'sync && sudo -S <<< 'qwe1!' sysctl -w vm.drop_caches=3'

EXPORT = "sync && echo 'qwe1!' | sudo -kS sysctl -w vm.drop_caches=3"

for i in range(10):
    result = os.popen('docker ps').read()
    result = result.split()

    if result.count('Up') == 4:
        os.system(EXPORT)

    else:
        print('else')
    
    sleep(6)

