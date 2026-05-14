#!/usr/bin/env bash
set -e

echo "[startup] fixing permissions..."

chown -R claude-user:claude-user /root/capsule/.claude
chown -R claude-user:claude-user /root/capsule/scratch
chown -R claude-user:claude-user /root/capsule/code
chown -R claude-user:claude-user /root/capsule/results
chmod o+rx /root/capsule/data
