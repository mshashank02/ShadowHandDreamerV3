# ShadowHandDreamerV3
# 🖥️ Remote Script Execution via CoE VPN

This guide walks you through running a Python script remotely on a server via SSH and `tmux`. It is intended for use with the CoE VPN and the remote server at `169.237.227.156`.

---

## 🔐 Step 1: Connect to the CoE VPN

Make sure you're connected to the **CoE VPN** before proceeding.

---

## 🔗 Step 2: SSH into the Remote Server

Open your terminal and run:

```bash
ssh shashank@169.237.227.156

When prompted, enter the password:
#lara2024

Start tmux session:
tmux new -s session_name

Run script:
/home/shashank/miniconda3/bin/python /path/to/your_script.py

Detach from tmux session:
Ctrl + B, then press D

Reconnect to session:
tmux attach-session -t session_name
