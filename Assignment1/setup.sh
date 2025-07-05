#!/bin/bash

main() {
    local -r cmddir=$(dirname $(readlink -f ${0}))
    if [[ ! -d /tmp/data/ || ! -e /tmp/data/CO2_sensor_measurements.csv ]]; then
	mkdir -m 777 -p -- /tmp/data
        pushd /tmp
        curl -L "https://drive.switch.ch/index.php/s/zv4Zuv0AFZJx42B/download" -o $$.homework1.zip
        unzip -n -o $$.homework1.zip||true
        rm -rf $$.homework1.zip
        popd
    fi
    mkdir -p ~/shared
    ln -sf /tmp/data ~/shared
    echo "Data files can be found in ${HOME}/shared/data (please, use the shorthand  ~/shared/data)"
}

main "$@"
