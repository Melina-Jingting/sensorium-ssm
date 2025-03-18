scp  -P 14510 -i ~/.ssh/id_ed25519 data/sensorium_all_2023/*.zip root@78.130.201.2:sensorium/data/sensorium_all_2023/dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20.zip
sshfs root@209.53.88.242:sensorium/data/sensorium_all_2023 /mnt/runpod
tailscale up
