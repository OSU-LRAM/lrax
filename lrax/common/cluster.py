# Copyright 2026, Laboratory for Robotics and Applied Mechanics
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class Resources:
    """Compute resources requested for a single Slurm job.

    Bundling the partition alongside the resource counts keeps the two in sync, e.g.,
    a job requesting a GPU should also request the partition that has GPUs.
    """

    partition: str
    time: str
    gpus: int = 0
    cpus: int = 1
    mem: str = "8G"


@dataclass
class Cluster:
    """Configuration for a Slurm cluster, shared across every job submitted to it."""

    account: str
    work_dir: str
    default_resources: Resources
    modules: Sequence[str] = field(default_factory=tuple)
    venv: Optional[str] = None
    mail_user: Optional[str] = None
    mail_type: str = "BEGIN,END,FAIL"


@dataclass
class Job:
    """A single program invocation to submit to Slurm.

    `command` is run as-is, so it can express any job type, e.g., `["python", "-m",
    "train", ...]` or a call to a standalone script. To sweep over a parameter (e.g.,
    a random seed), build one `Job` per value at the call site rather than modeling
    the sweep here.
    """

    name: str
    command: Sequence[str]
    resources: Optional[Resources] = None


def make_sbatch_script(
    job: Job, cluster: Cluster, output_dir: str | Path = "run_output"
) -> str:
    """Render the `sbatch` script for `job` on `cluster`.

    Parameters
    ----------
    - `job`: The job to build a script for.
    - `cluster`: The cluster the job will run on.
    - `output_dir`: The directory `sbatch` writes stdout/stderr logs to. Defaults to
        `"run_output"`.

    Returns
    -------
    The contents of the `sbatch` script as a string.
    """
    resources = job.resources or cluster.default_resources

    headers = [
        "#!/bin/bash",
        f"#SBATCH -J {job.name}",
        f"#SBATCH -A {cluster.account}",
        f"#SBATCH -p {resources.partition}",
        f"#SBATCH -o {output_dir}/%x-%j.out",
        f"#SBATCH -e {output_dir}/%x-%j.err",
        f"#SBATCH -t {resources.time}",
        f"#SBATCH -c {resources.cpus}",
        f"#SBATCH --mem={resources.mem}",
    ]
    if resources.gpus > 0:
        headers.append(f"#SBATCH --gres=gpu:{resources.gpus}")
    if cluster.mail_user is not None:
        headers.append(f"#SBATCH --mail-type={cluster.mail_type}")
        headers.append(f"#SBATCH --mail-user={cluster.mail_user}")

    body = [f"module load {module}" for module in cluster.modules]
    body.append(f"cd {cluster.work_dir} || exit 1")
    if cluster.venv is not None:
        body.append(f"source {cluster.venv}")
    body.append(" ".join(job.command))

    return "\n".join(headers) + "\n\n" + "\n".join(body) + "\n"


def submit_slurm_job(
    job: Job,
    cluster: Cluster,
    *,
    output_dir: str | Path = "run_output",
    dry_run: bool = False,
) -> None:
    """Submit `job` to `cluster` via `sbatch`.

    Parameters
    ----------
    - `job`: The job to submit.
    - `cluster`: The cluster to submit the job to.
    - `output_dir`: The directory `sbatch` writes stdout/stderr logs to. Defaults to
        `"run_output"`. (Keyword only argument.)
    - `dry_run`: If `True`, print the rendered script instead of submitting it.
        Defaults to `False`. (Keyword only argument.)
    """
    script = make_sbatch_script(job, cluster, output_dir)

    if dry_run:
        print(f"\n{'=' * 60}\nDRY RUN - Would submit: {job.name}\n{'=' * 60}\n{script}")
        return

    print(f"Submitting: {job.name}")
    result = subprocess.run(["sbatch"], input=script, text=True, capture_output=True)

    if result.returncode == 0:
        print(f"  -> Submitted: {result.stdout.strip()}")
    else:
        print(f"  -> ERROR: {result.stderr.strip()}")
