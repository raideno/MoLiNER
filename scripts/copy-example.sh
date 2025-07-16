# NOTE: to copy content from a remote server to the local machine.
scp -r nadirkichou@34.130.85.248:/home/nadirkichou/MoLiNER/out/training.2025-07-15_11-18-53/logs/plots ./

# NOTE: an alternative using rsync for better performance and overall progress.
rsync -avz -e ssh user@host:/remote/dir/ ./local/dir/