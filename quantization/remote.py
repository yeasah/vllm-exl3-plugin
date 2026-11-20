#!/usr/bin/env python3
"""Drive quant.py's pipeline on a remote (cloud) box over ssh, from a machine that stays up.

Each stage (traces, recall slice, YAQA, baseline, measure, probe, quantize, qbench, card, upload) runs remotely
under nohup with its exit status written to a file, so a dropped ssh session does not kill or lose it; this side
polls the log, shows it, and syncs outputs to a local record (--sync-local) and/or an HF bucket (--sync-bucket)
every minute, so a box that dies mid-stage loses at most that much. Learned on rented hardware:

- ssh's exit 255 (connection failure) is retried every minute indefinitely, with an alert, rather than failing
  the stage: cloud boxes drop off the network and come back.
- A line "!!! JOB FAILED:" in a stage's log raises an alert while the stage carries on (a part failed).
- Ctrl-C kills the remote stage's whole process group, not just its shell.
- --retries reruns the whole pipeline after a pause; quant.py skips what is already done, so a rerun resumes.

Alerts ring the terminal bell and, with --notify (default $EXL3_NOTIFY, e.g. ntfy.sh/<topic>), post the message
there. The record lives in work/_remote/<model>-exl3, with the driver's own log beside it as <model>-exl3.log
(--record-dir moves it): not work/<model>-exl3, which is quant.py's job directory for the same model, and syncing a
remote job into it would merge it with a local one.
"""

import argparse
import os
import sys
import subprocess
import time
import re
import shlex

RETRY_MIN = 5
HERE = os.path.dirname(os.path.abspath(__file__))

def do_log(job, msg, do_stamp=True):
    stamp = time.strftime("[%Y-%m-%d %H:%M:%S] ") if do_stamp else ""
    print(f"{stamp}{msg}")
    if "logfile" in job:
        job["logfile"].write(f"{stamp}{msg}\n")

def sync_to(job):
    if job['args'].dry_run:
        return True

    # sync stored files from local record to remote, if configured
    if job['args'].sync_local:
        do_log(job, f"*** SYNCING TO REMOTE ***")
        cmd = [ "rsync", "-aue", "ssh",
                f"{job['localpath']}/",
                f"{job['args'].host}:{job['remotework']}" ]
        subprocess.run(cmd)

    # sync stored files from bucket to remote, if configured
    if job['args'].sync_bucket:  
        do_log(job, f"*** SYNCING TO REMOTE FROM BUCKET ***")
        run_command(job, f"hf sync -q hf://buckets/{job['args'].sync_bucket}/{job['modelpath']}/ {job['remotework']}" +
                    (" --include _traces/*" if job['need_traces'] else "") +
                    " --include logs/*" +
                    (" --include _measure/*" if job["need_measure"] else "") +
                    (" --include _recipes/*" if job["need_quants"] else "") +
                    (" --include *bpw*/*" if job["need_quants"] else "") +
                    " --include main/*" +
                    (" --include _hess/*" if job["need_yaqa"] else "") +
                    (" --include _baseline/*" if job["need_baseline"] else ""))
                    
    return True

def sync_from(job, cmd_name):
    if job['args'].dry_run:
        return True

    # sync generated files from remote to local record, if configured
    if job['args'].sync_local:
        cmd = [ "rsync", "-ae", "ssh",
                "--include", "*/",
                "--include", "_traces/*",
                "--include", "logs/*",
                "--include", "_measure/*",
                "--include", "_recipes/*",
                "--include", "main/*",
                "--exclude", "*",
                f"{job['args'].host}:{job['remotework']}/",
                f"{job['localpath']}" ]
        subprocess.run(cmd)

    # sync generated files from remote to bucket, if configured
    if job['args'].sync_bucket:
        cmd = [ "ssh", job['args'].host,
                f"hf sync {job['remotework']}/ hf://buckets/{job['args'].sync_bucket}/{job['modelpath']}" +
                " --include _traces/*" +
                " --include logs/*" +
                " --include _measure/*" +
                " --include _recipes/*" +
                " --include main/*" +
                (" --include _hess/*" if job["args"].sync_yaqa else "") +
                (" --include _baseline/*" if job["args"].sync_baseline else "") +
                (" --include *bpw*/*" if job["args"].sync_quants else "") +
                " --exclude _hess/blend/*" +
                " --exclude */_work*" ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            cnt = re.search(r'Uploads: (\d+)', res.stdout)
            if cnt:
                do_log(job, f"*** SYNC REMOTE TO BUCKET {cnt.group(1)} FILES ***")
    return True

def poll_log(job, self_pid, num_lines):
    # show log, and sync generated files
    command = f'tail -n +{num_lines+1} /tmp/monitor.{self_pid}.log'
    res = run_remote(job, command, False)
    if res.returncode == 0:
        errs = re.findall(r'.*!!! JOB FAILED:.*', res.stdout.decode())
        if errs:
            get_user_attention(job, " -- ".join(errs))

        # splitlines splits on '\r' too, and tail doesn't
        lines = res.stdout.split(b'\n')[:-1]
        if lines:
            do_log(job, "\n".join(map(lambda x: f"  > {x.split(b'\r')[-1].decode(errors='replace')}", lines)), False)
        num_lines += len(lines)
    return num_lines

def run_remote(job, cmd, text=True):
    command = [ "ssh", job["args"].host, cmd ]
    failed = False
    while True:
        #print(f" running {cmd}...", end="")
        res = subprocess.run(command, capture_output=True, text=text, errors="replace" if text else None)
        #print(f" returned {res.returncode}")
        if res.returncode == 255:
            if not failed:
                get_user_attention(job, f"REMOTE COMMAND {cmd} FAILED, RETRYING")
                failed = True
            do_log(job, f"*** REMOTE COMMAND {cmd} FAILED, RETRYING ***")
            do_log(job, res.stderr)
            time.sleep(60)
        else:
            if failed:
                get_user_attention(job, f"REMOTE COMMAND {cmd} SUCCEEDED, PROCEEDING")
            return res

def monitor(job, cmd, cmd_name):
    self_pid = os.getpid()
    do_log(job, f"*** RUNNING BACKGROUND {cmd_name} ({cmd}) ***")
    bgcmd=cmd.replace('"', r'\"')
    command = f'rm -f /tmp/monitor.{self_pid}.*; nohup sh -c "{bgcmd}; echo \\$? > /tmp/monitor.{self_pid}.res" > /tmp/monitor.{self_pid}.log 2>&1 & echo $!,`ps -o pgid= -p $$`'
    if job['args'].dry_run:
        return True

    # run command, which runs in background and prints pid
    res = run_remote(job, command)
    if res.returncode != 0:
        do_log(job, f"*** COMMAND {cmd_name} FAILED TO START ***")
        return False
    (pid, pgid) = map(int, res.stdout.split(','))
    if not pid:
        do_log(job, f"*** COMMAND {cmd_name} FAILED TO START, NO PID REPORTED ***")
        return False;

    #print(f"*** RAN BACKGROUND COMMAND {' '.join(command)} PID={pid} PGID={pgid} ***")

    # monitor loop
    num_lines = 0
    try:
        time.sleep(1)

        iter = 0
        # test if process still running
        while run_remote(job, f"kill -0 {pid} > /dev/null 2>&1").returncode == 0:
            num_lines = poll_log(job, self_pid, num_lines)
            if iter % 6 == 0:
                sync_from(job, cmd_name)
            iter += 1
            time.sleep(10)
    except:
        # kill the whole process group
        do_log(job, f"*** ABORTING {cmd_name} (pgid {pgid}) ***")
        run_command(job, f"kill -- -{pgid}")
        raise

    # final sync
    poll_log(job, self_pid, num_lines)
    sync_from(job, cmd_name)

    # read result code from file where it was written
    res = run_remote(job, f"cat /tmp/monitor.{self_pid}.res")
    if res.returncode == 0:
        success = int(res.stdout) == 0
        if not success:
            do_log(job, f"*** COMMAND {cmd_name} FAILED ***")
        else:
            do_log(job, f"*** COMMAND {cmd_name} SUCCEEDED ***")
        return success
    do_log(job, f"*** COMMAND {cmd_name} ENDED WITHOUT RESULT ***")
    return False

def run_command(job, cmd):
    command = [ "ssh", job["args"].host, cmd ]
    if job['args'].dry_run:
        return True
    res = subprocess.run(command)
    return res.returncode == 0

def run_quant(job, cmd, cuda_ex_seg=1):
    cmd_name = cmd.split(' ')[0]
    return monitor(job, f"cd {job['remotepath']} && EXL3_EXPANDABLE_SEGMENTS={cuda_ex_seg} ./quant.py -m {job['args'].model} {cmd}", cmd_name)

def setup_host(job):
    if job['args'].sync_local:    
        os.makedirs(job['localpath'], exist_ok=True)

    do_log(job, f"*** WAITING FOR {job['args'].host} TO RESPOND ***")
    run_remote(job, "true")

    # include host survey
    res = run_remote(job, os.path.join(job["args"].plugin_dir, "tools", "host_survey.py"))
    if res.stdout:
        do_log(job, res.stdout)
        
    return run_remote(job, f"mkdir -p {job['remotework']}").returncode == 0 and sync_to(job)

def do_traces(job):
    args = ""
    if job['args'].traces_generator:
        args += f" --generator {job['args'].traces_generator}"
    if job['args'].traces_vllm_args:
        args += f" --vllm-args '{job['args'].traces_vllm_args}'"
    if job['args'].traces_backend:
        args += f" --backend {job['args'].traces_backend}"
        
    if not run_quant(job, f"eval-traces -d {job['devlist']} --retries 0{args}", 0):
        return False
    if not run_quant(job, f"cal-traces -d {job['devlist']} --retries 0{args}", 0):
        return False
    if job['need_yaqa'] and not run_quant(job, f"yaqa-traces -d {job['devlist']} --retries 0{args}", 0):
        return False
    return True
    
def do_recall(job):
    args = ""
    if job['args'].allow_thin:
        args += " --allow-thin"
    return run_quant(job, f"recall-slice -d {job['devlist']}{args}")

def do_yaqa(job):
    args = ""
    if job['args'].layers_per_run:
        args += f" --layers-per-run {job['args'].layers_per_run}"
    if job['args'].reserve_vram:
        args += f" --reserve-vram {job['args'].reserve_vram}"
    return run_quant(job, f"yaqa -d {job['devlist']}{args}")

def do_measure(job):
    args = ""
    if job['args'].gpu_split:
        args += f" --gpu-split {job['args'].gpu_split}"
    return run_quant(job, f"measure -d {job['devlist']}{args}")

def do_probe(job):
    args = ""
    if job['args'].yaqa:
        args += f" --yaqa"
        if job['args'].layers_per_run:
            args += f" --layers-per-run {job['args'].layers_per_run}"
        if job['args'].reserve_vram:
            args += f" --reserve-vram {job['args'].reserve_vram}"
    if job['args'].probe_draws:
        args += f" --draws {job['args'].probe_draws}"
    for arm in job['args'].probe_arm or []:
        args += f" --arm {shlex.quote(arm)}"
    if job['args'].quant_parallel:
        args += f" --quant-parallel {job['args'].quant_parallel}"

    trace_args = ""
    if job['args'].traces_generator:
        trace_args += f" --generator {job['args'].traces_generator}"
    if job['args'].traces_vllm_args:
        trace_args += f" --vllm-args '{shlex.quote(job['args'].traces_vllm_args)}'"
    if job['args'].traces_backend:
        trace_args += f" --backend {job['args'].traces_backend}"
    if trace_args:
        args += f" --trace-args='{trace_args}'"

    return monitor(job, f"cd {job['remotepath']} && EXL3_EXPANDABLE_SEGMENTS=1 ./probe.py -m {job['args'].model} -d {job['devlist']}{args}", "probe")

def do_quantize(job):
    args = ""
    if job['args'].quantize_args:
        args += f" {job['args'].quantize_args}"
    return run_quant(job, f"quantize --retries 0 -b {job['bits']} -d {job['quant_devlist']}{args}")

def do_qbench(job, reference):
    args = ""
    if job['args'].threshold:
        for r in job['args'].threshold:
            args += f" --threshold {r}"
    if job['args'].threshold_gguf:
        for r in job['args'].threshold_gguf:
            args += f" --threshold-gguf {r}"
    if reference:
        args += f" --reference-min-bits {job['args'].min_bits} --reference-max-bits {job['args'].max_bits + 0.1}"
        if job['args'].reference:
            for r in job['args'].reference:
                args += f" --reference {r}"

        if job['args'].reference_gguf:
            for r in job['args'].reference_gguf:
                args += f" --reference-gguf {r}"
            if job['args'].gguf_quants:
                args += f" --gguf-quants {job['args'].gguf_quants}"
    return run_quant(job, f"qbench -d {job['devlist']}{args}")

def do_card(job):
    return run_quant(job, f"card --gguf-kld")

def do_upload(job):
    return run_quant(job, f"upload --no-private")

def do_baseline(job):
    args=""
    if job['args'].reference:
        args += f" -b {job['bits']}"
    return run_quant(job, f"baseline --no-generator --retries 0 -d {job['quant_devlist']}{args}")

def get_user_attention(job, msg):
    do_log(job, f"*** ATTENTION {job['args'].host}: {msg} ***")
    os.system("tput bel")
    if job['args'].notify:
        # an argument list, not a shell string: messages carry remote commands, quotes included
        subprocess.run(["curl", "-sd", msg, job['args'].notify], stdout=subprocess.DEVNULL)

def attempt_pipeline(job):
    args = job['args']

    # build ladder
    ladder = []
    rate = args.min_bits
    while rate <= args.max_bits:
        if rate < 4:
            ladder.append(f'{rate}:5')
            rate += 0.5
        else:
            ladder.append(f'{rate}')
            rate += 1.0
    job['bits'] = ",".join(ladder)
    
    # dependencies
    job['need_card'] = args.card or args.upload
    job['need_qbench'] = args.qbench or job['need_card']
    job['need_quants'] = args.quantize or job['need_qbench']
    job['need_yaqa'] = job['need_quants'] and args.yaqa
    job['need_measure'] = (job['need_quants'] or args.probe) and args.measure
    job['need_baseline'] = job['need_measure'] or job['need_qbench']
    job['need_recall'] = (job['need_qbench'] or args.probe) and args.recall
    job['need_traces'] = job['need_qbench'] or job['need_quants'] or args.yaqa or args.measure or args.probe

    if not setup_host(job):
        return False

    if job['need_traces'] and not do_traces(job):
        return False

    if job['need_recall'] and not do_recall(job):
        return False
    
    if job['need_yaqa'] and not do_yaqa(job):
        return False

    if job['need_baseline'] and not do_baseline(job):
        return False

    if job['need_measure'] and not do_measure(job):
        return False

    if args.probe and not do_probe(job):
        return False

    if job['need_quants'] and not do_quantize(job):
        return False

    if job['need_qbench'] and not do_qbench(job, True):
        return False

    if job['need_card'] and not do_card(job):
        return False

    if args.upload and not do_upload(job):
        return False

    return True
    
def main():
    parser = argparse.ArgumentParser(description="EXL3 quantization pipeline driver")
    parser.add_argument('--host', default='vast', help='hostname or ssh alias to connect to')
    parser.add_argument('--plugin-dir', default="~/vllm-exl3-plugin")
    parser.add_argument('--num-devices', default=1, type=int, help='number of CUDA devices present')
    parser.add_argument('--quant-parallel', default=2, type=int, help='number of concurrent quantizations to run')

    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('--min-bits', type=float, default=3.0,
                        help='ladder low bitrate; below 3 only on evidence, such as the probe\'s low draw'),
    parser.add_argument('--max-bits', type=float, default=6, help='ladder high bitrate'),
    
    parser.add_argument('--record-dir', default=os.path.join(HERE, "work", "_remote"),
                        help='where the local record and the driver log go (default: work/_remote beside this script)')
    parser.add_argument('--dry-run', action='store_true',
                        help='log the stage commands without running them or syncing; still connects to the host and '
                             'runs its survey, which checks the ssh plumbing and changes nothing')
    parser.add_argument("--notify", default=os.environ.get("EXL3_NOTIFY"),
                        help="URL to POST alerts to, e.g. ntfy.sh/<topic> (default: $EXL3_NOTIFY; none if unset)")
    parser.add_argument("--retries", default=0, type=int)
    parser.add_argument("--notify-complete", action='store_true')

    parser.add_argument("--sync-baseline", action='store_true', help='synchronize basline quants to/from bucket')
    parser.add_argument('--sync-quants', action='store_true', help='synchronize ladder quants to/from bucket')
    parser.add_argument('--sync-yaqa', action='store_true', help='synchronize yaqa factors to/from bucket')
    parser.add_argument('--sync-bucket', help='HF bucket to synchronize to/from')
    parser.add_argument('--sync-local', action='store_true', help='synchronize logs, traces and measurements to local folder')

    # actions
    parser.add_argument('--probe', action='store_true', help='run probe on model')
    parser.add_argument('--quantize', action='store_true', help='create ladder quants')
    parser.add_argument('--qbench', action='store_true', help='evaluate ladder quants')
    parser.add_argument('--card', action='store_true', help='create model card')
    parser.add_argument('--upload', action='store_true', help='upload final results')

    # action options
    parser.add_argument('--recall', default=True, action=argparse.BooleanOptionalAction, help='create and score recall slice')
    parser.add_argument('--yaqa', default=False, action=argparse.BooleanOptionalAction, help='perform yaqa passes')
    parser.add_argument('--measure', default=False, action=argparse.BooleanOptionalAction, help='perform sensitivity measurement')

    # traces args
    parser.add_argument('--traces-generator', help='generator model to use for traces')
    parser.add_argument('--traces-backend', default='vllm', help='backend to use for traces (exllamav3 or vllm)')
    parser.add_argument('--traces-vllm-args', help='extra vllm args during trace generation')

    # measurement args
    parser.add_argument("--gpu-split", help='recipe measurement per-device memory allocation')

    # recall args
    parser.add_argument("--allow-thin", action='store_true')

    # probe args
    parser.add_argument("--probe-draws", type=int, default=3)
    parser.add_argument("--probe-arm", action="append", metavar="TAG=ARGS",
                        help="probe also a rung at quantize ARGS (probe.py --arm); repeatable")
    
    # yaqa args
    parser.add_argument("--layers-per-run", help='YAQA number of layers per iteration')
    parser.add_argument("--reserve-vram", help='YAQA amount of memory to reserve for non-weights, per device')

    # quantize args
    parser.add_argument('--quantize-args', help='extra quantize args')

    # qbench args
    parser.add_argument('--reference', action='append', help='reference repo to eval against')
    parser.add_argument('--threshold', action='append', help='threshold reference quant for eval')
    parser.add_argument('--gguf-quants', help='GGUF quants to use for reference ladder. The usual lineup for a 3-6bpw ladder is the repo equivalent of IQ2_M,IQ3_XXS,IQ3_M,IQ4_XS,Q4_K_M,Q5_K_M')
    parser.add_argument('--reference-gguf', action='append', help='GGUF reference repo to eval against')
    parser.add_argument('--threshold-gguf', action='append', help='one GGUF file as a threshold line, REPO@QUANT')

    args = parser.parse_args()
    recordpath=args.record_dir
    modelpath=f"{args.model.split('/')[1]}-exl3"
    remotepath=os.path.join(args.plugin_dir, "quantization")
    devlist=",".join(map(str, range(0, args.num_devices)))
    quant_devlist=devlist * args.quant_parallel
    job = { "args": args,
            "devlist": devlist,
            "quant_devlist": quant_devlist,
            "modelpath": modelpath,
            "localpath": os.path.join(recordpath, modelpath),
            "remotepath": remotepath,
            "remotework": os.path.join(remotepath, "work", modelpath) }

    os.makedirs(recordpath, exist_ok=True)    
    with open(f"{job['localpath']}.log", "a") as job["logfile"]:
        do_log(job, f"*** STARTING PIPELINE ({' '.join(sys.argv)}) ***")
        iter = 0
        while not attempt_pipeline(job):
            if iter < args.retries:
                get_user_attention(job, f"pipeline failed, try {iter+1}/{args.retries+1}, retrying in {RETRY_MIN}m")
                iter += 1
                time.sleep(60 * RETRY_MIN)
            else:
                get_user_attention(job, "pipeline failed, giving up")
                return 1
        if args.notify_complete:
            get_user_attention(job, "pipeline complete")
    return 0
            
if __name__ == "__main__":
    main()
