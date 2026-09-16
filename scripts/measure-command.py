#!/usr/bin/env python3
"""Linux process-group measurements; sampled RSS is not unique physical memory."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def group_rss(group):
    rss = 0
    members = 0
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / 'stat').read_text().rpartition(')')[2].split()
            if int(fields[2]) == group:
                rss += int(fields[21]) * os.sysconf('SC_PAGE_SIZE')
                members += 1
        except (OSError, ValueError, IndexError):
            pass
    return rss, members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=3600)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.timeout <= 0:
        parser.error('provide a command and a positive timeout')
    args.out.mkdir(parents=True, exist_ok=False)
    energy_path = Path('/sys/class/powercap/intel-rapl:0/energy_uj')
    try:
        energy_start = int(energy_path.read_text())
        energy_range = int(energy_path.with_name('max_energy_range_uj').read_text())
        power = {'status': 'package energy includes all host workloads', 'counter': str(energy_path)}
    except (OSError, ValueError) as error:
        energy_start = None
        power = {'status': 'unavailable', 'reason': str(error)}
    energy_last = energy_start
    energy_total = 0
    start = time.monotonic()
    report = {'command': command, 'sample_period_seconds': 0.2, 'power': power,
              'peak_group_rss_bytes': 0, 'peak_group_members': 0,
              'rss_definition': 'sum of resident pages across the process group; shared pages may be counted twice',
              'initial_load_average': os.getloadavg()}
    with (args.out / 'stdout.log').open('w') as stdout, (args.out / 'stderr.log').open('w') as stderr:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while True:
                if energy_last is not None:
                    try:
                        current = int(energy_path.read_text())
                        energy_total += (current - energy_last) % energy_range
                        energy_last = current
                    except (OSError, ValueError) as error:
                        energy_last = None
                        report['power'] = {'status': 'unavailable', 'reason': str(error)}
                pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    child.returncode = os.waitstatus_to_exitcode(status)
                    break
                rss, members = group_rss(child.pid)
                report['peak_group_rss_bytes'] = max(report['peak_group_rss_bytes'], rss)
                report['peak_group_members'] = max(report['peak_group_members'], members)
                report['wall_seconds'] = time.monotonic() - start
                (args.out / 'status.json').write_text(json.dumps(report, indent=2) + '\n')
                if report['wall_seconds'] > args.timeout:
                    raise TimeoutError(f'command exceeded {args.timeout} seconds')
                time.sleep(0.2)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            raise
    report.update(exit_code=child.returncode, wall_seconds=time.monotonic() - start,
                  user_cpu_seconds=usage.ru_utime, system_cpu_seconds=usage.ru_stime,
                  max_process_rss_kib=usage.ru_maxrss, final_load_average=os.getloadavg())
    if energy_last is not None:
        report['power'].update(energy_microjoules=energy_total,
                               scope='whole package, not application-attributed; assumes fewer than one counter wrap between samples')
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()
