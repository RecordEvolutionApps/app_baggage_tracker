#!/bin/bash

mkdir -p /data/ssh 
mkdir -p ~/.ssh
if [ -f /data/ssh/id_rsa ]; then
  echo "SSH key already exists at /data/ssh/id_rsa"
else
  # Generate a new key if it doesn't exist
  ssh-keygen -t rsa -b 4096 -f /data/ssh/id_rsa -N ""
fi

cp -a /data/ssh/* ~/.ssh

# sleep infinity
exec bun run /app/backend/index.ts
