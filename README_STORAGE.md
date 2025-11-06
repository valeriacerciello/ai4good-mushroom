README — storage recommendations for clusters

Goal
----
Keep large, long-lived data off your home directory on shared clusters. Use the cluster's persistent project storage for canonical datasets and small configs, and use scratch for ephemeral, high-throughput files.

Recommended layout
------------------
- Persistent project area (example): /cluster/work/<group>/<user>/mushroom_clip
- Scratch / fast area (example): /scratch/<user>/mushroom_clip_run_<jobid>

Example commands
----------------
Create a persistent project folder and clone this repo there:

    mkdir -p /cluster/work/<group>/<user>/mushroom_clip
    cd /cluster/work/<group>/<user>
    git clone <your-repo-url> mushroom_clip

Use scratch for training runs and temporary data:

    mkdir -p /scratch/<user>/mushroom_clip_run_$(date +%s)
    rsync -a --exclude '.git' /cluster/work/<group>/<user>/mushroom_clip/ /scratch/<user>/mushroom_clip_run_<jobid>/

After training, copy back only artifacts you want to keep (models, logs, evaluation results):

    rsync -a /scratch/<user>/mushroom_clip_run_<jobid>/checkpoints /cluster/work/<group>/<user>/mushroom_clip/artifacts/

Best practices
--------------
- Don't store large datasets in your home directory.
- Keep code under version control in the persistent project area (or on Git remote).
- Clean scratch after jobs to avoid quota issues.
- Compress or deduplicate datasets when possible.

Policies and quota
------------------
Check your cluster's documentation or contact system administrators for exact quotas, available filesystems, and backup/retention policies.

Contact
-------
If you'd like, I can create the folders in a detected persistent location and move the workspace files there. Reply with which option you prefer.
