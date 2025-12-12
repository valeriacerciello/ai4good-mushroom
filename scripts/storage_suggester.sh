#!/usr/bin/env bash
# storage_suggester.sh
# Scans the system for common cluster storage locations and prints recommendations

set -euo pipefail

USER_NAME="${USER:-$(whoami)}"

echo "Scanning common cluster storage locations to recommend where to store project data..."

declare -a PREFER_PERSIST=("/cluster/work" "/work" "/projects" "/project" "/data" "/srv")
declare -a SCRATCH_LOC=("/scratch" "/cluster/scratch" "/tmp" "/local/scratch")

found_persist=""
for p in "${PREFER_PERSIST[@]}"; do
  if [ -d "$p" ]; then
    found_persist="$p"
    echo "Found persistent-ish storage: $p"
  fi
done

found_scratch=""
for s in "${SCRATCH_LOC[@]}"; do
  if [ -d "$s" ]; then
    found_scratch="$s"
    echo "Found scratch/fast storage: $s"
  fi
done

echo
echo "Recommendation (best practices):"
if [ -n "$found_persist" ]; then
  echo "- Use the persistent shared storage (example: $found_persist) for the canonical project repository, small configs, and datasets that must be kept long-term."
  echo "  Suggested project path: $found_persist/<group-or-user>/${USER_NAME}/mushroom_clip"
  echo "  Example commands (no sudo expected on cluster):"
  echo "    mkdir -p $found_persist/<group-or-user>/${USER_NAME}/mushroom_clip"
  echo "    cd $found_persist/<group-or-user>/${USER_NAME} && git clone <your-repo-url> mushroom_clip"
else
  echo "- No obvious persistent shared storage found in common locations. If you have a project filesystem provided by your cluster (ask sysadmins), prefer using that instead of your home directory for large/long-lived data."
fi

if [ -n "$found_scratch" ]; then
  echo "- Use the scratch space ($found_scratch) for large, ephemeral data: model checkpoints during training, temporary preprocessing files, and fast I/O workloads."
  echo "  Example: create a job-specific folder and clean it up after the job completes."
  echo "    mkdir -p $found_scratch/${USER_NAME}/mushroom_clip_run_\$(date +%s)"
  echo "    # after job finishes, move artifacts you want to keep to persistent storage and rm -rf the scratch folder"
else
  echo "- No obvious scratch directory found; you can still use a local fast disk if your cluster exposes one, or use temporary subfolders on the persistent volume, but be mindful of quotas and I/O performance."
fi

echo
echo "Quick policy notes:" 
echo "- Avoid storing large datasets or many model checkpoints in your home directory; home is often small and backed up, and not intended for heavy I/O."
echo "- Keep code in a git repo and store the canonical repo on persistent storage or a remote host (GitHub/GitLab). Clone into the persistent project directory."
echo "- Use symlinks from your working directory to data in the persistent or scratch locations when useful."
echo
echo "If you'd like, I can (A) create the recommended project folder under a found persistent location (if it exists), (B) add a README in your repo documenting the storage layout, or (C) generate commands to move the current workspace contents to the recommended location. Tell me which you'd prefer." 

exit 0
