from __future__ import annotations

import argparse
import os
import posixpath
import time
from pathlib import Path

import paramiko


def ensure_remote_dir(sftp: paramiko.SFTPClient, path: str) -> None:
    parts = path.strip("/").split("/")
    current = ""
    for part in parts:
        current += "/" + part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def remote_size(sftp: paramiko.SFTPClient, path: str) -> int | None:
    try:
        return sftp.stat(path).st_size
    except FileNotFoundError:
        return None


def iter_files(local_root: Path) -> list[Path]:
    return sorted(path for path in local_root.rglob("*") if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload local dataset cache to AutoDL via SFTP.")
    parser.add_argument("--host", default="connect.bjb2.seetacloud.com")
    parser.add_argument("--port", type=int, default=29372)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password-env", default="AUTODL_PASSWORD")
    parser.add_argument("--local-root", type=Path, default=Path("data/cache"))
    parser.add_argument("--remote-root", default="/root/autodl-fs/GPT3-small-V1/data/cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = os.environ.get(args.password_env)
    if not password:
        raise RuntimeError(f"Set {args.password_env} before running this script")

    local_root = args.local_root.resolve()
    files = iter_files(local_root)
    total_bytes = sum(path.stat().st_size for path in files)
    print(f"local root: {local_root}", flush=True)
    print(f"remote root: {args.remote_root}", flush=True)
    print(f"files: {len(files)}", flush=True)
    print(f"bytes: {total_bytes}", flush=True)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=args.host,
        port=args.port,
        username=args.user,
        password=password,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    sftp = client.open_sftp()
    ensure_remote_dir(sftp, args.remote_root)

    uploaded_bytes = 0
    skipped_bytes = 0
    start_all = time.time()
    try:
        for index, local_path in enumerate(files, start=1):
            rel = local_path.relative_to(local_root).as_posix()
            remote_path = posixpath.join(args.remote_root, rel)
            size = local_path.stat().st_size
            ensure_remote_dir(sftp, posixpath.dirname(remote_path))

            existing_size = remote_size(sftp, remote_path)
            if existing_size == size:
                skipped_bytes += size
                print(f"[{index}/{len(files)}] skip complete {rel} ({size} bytes)", flush=True)
                continue

            if existing_size is not None and existing_size != size:
                print(
                    f"[{index}/{len(files)}] overwrite partial {rel} "
                    f"(remote={existing_size}, local={size})",
                    flush=True,
                )
            else:
                print(f"[{index}/{len(files)}] upload {rel} ({size} bytes)", flush=True)

            start = time.time()
            sftp.put(str(local_path), remote_path)
            elapsed = max(0.001, time.time() - start)
            uploaded_bytes += size
            mbps = size / 1024 / 1024 / elapsed
            total_done = skipped_bytes + uploaded_bytes
            overall_elapsed = max(0.001, time.time() - start_all)
            overall_mbps = total_done / 1024 / 1024 / overall_elapsed
            print(
                f"done {rel}: {mbps:.2f} MB/s; "
                f"total {total_done}/{total_bytes} bytes ({overall_mbps:.2f} MB/s overall)",
                flush=True,
            )
    finally:
        sftp.close()
        client.close()

    print("upload complete", flush=True)


if __name__ == "__main__":
    main()
