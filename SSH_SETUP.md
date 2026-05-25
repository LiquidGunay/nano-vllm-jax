# Directory-Scoped SSH Setup

The previous `~/.ssh/nano_vllm_jax` instructions are stale for this workspace. Keep repository authentication scoped under `/mountpoint/.exp/.ssh` so this forked workspace does not depend on or mutate global SSH state.

## Expected layout

```text
/mountpoint/.exp/.ssh/
  nano_vllm_jax
  nano_vllm_jax.pub
  known_hosts
```

## Create or reuse the key

```bash
mkdir -p /mountpoint/.exp/.ssh
chmod 700 /mountpoint/.exp/.ssh
ssh-keygen -t ed25519 -C nano-vllm-jax -f /mountpoint/.exp/.ssh/nano_vllm_jax
chmod 600 /mountpoint/.exp/.ssh/nano_vllm_jax
chmod 644 /mountpoint/.exp/.ssh/nano_vllm_jax.pub
```

Add the public key from `/mountpoint/.exp/.ssh/nano_vllm_jax.pub` to GitHub as an authentication key.

## Scope Git to this directory

Configure only this repository to use the workspace key and workspace-scoped `known_hosts` file:

```bash
git config --local core.sshCommand "ssh -F /dev/null -i /mountpoint/.exp/.ssh/nano_vllm_jax -o IdentitiesOnly=yes -o UserKnownHostsFile=/mountpoint/.exp/.ssh/known_hosts"
```

If the remote is HTTPS and SSH pushes are required, switch only this repository:

```bash
git remote set-url origin git@github.com:LiquidGunay/nano-vllm-jax.git
```

Test with:

```bash
ssh -F /dev/null -i /mountpoint/.exp/.ssh/nano_vllm_jax -o IdentitiesOnly=yes -o UserKnownHostsFile=/mountpoint/.exp/.ssh/known_hosts -T git@github.com
```

Do not put this repository's private key in `~/.ssh`, and do not configure global Git SSH settings for this workspace.
