#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/../ai-engd
echo Current Directory:
pwd

#$BASEDIR/chat-with-pci-dss-v4/scripts/start.sh

pm2 stop all
pm2 start "PORT=64301 yarn start"
pm2 start "ngrok http --host-header=rewrite 64301 --log=stdout"

