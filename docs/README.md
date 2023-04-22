# Documentation
## Useful Commands
### Screen
It is recommended to submit jobs via a screen session. This will allow you to exit the terminal without terminating the jobs. 

Start a new screen session:
```
screen -S <session_name>
```
Re-attach to an existing screen:
```
screen -ls
screen -r <screen number/name>
```

Press `(ctrl+a) + ?` to list all key-binding options available within a screen session.

### Rsync
It was used to copy local files to a remote host.

```
rsync -av ./<src-folder>/ <hostname>:<dst-path>
```
**Note:** Download [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) to use `rsync` with windows. 

### SLURM
#### Interactive job
```
srun --gres=gpu:0 --partition=gpuA100 --pty bash
```
