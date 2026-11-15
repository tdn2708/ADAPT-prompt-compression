"""Kaggle runner: Lanes runs step lists in parallel, one lane per GPU, and stops every process before the deadline so
that Kaggle still writes the outputs (a committed notebook runs at most 12 hours).
"""

import os
import subprocess
import threading
import time
from pathlib import Path


class Lanes:
    """lanes: {name: (gpu or None, [shell command, ...])}."""

    def __init__(self, lanes, cwd, logs, deadline, on_step=None, poll=20):
        self.lanes, self.cwd, self.logs, self.deadline = lanes, cwd, Path(logs), deadline
        self.on_step, self.poll = on_step, poll
        self.status = {n: 'waiting' for n in lanes}
        self.history = []
        self.logs.mkdir(parents=True, exist_ok=True)

    def _run(self, name, gpu, steps):
        env = dict(os.environ)
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        log_path = self.logs / f'{name}.log'
        for step in steps:
            if time.time() > self.deadline:
                self.status[name] = 'stopped: deadline'
                return
            self.status[name] = step
            t0 = time.time()
            with open(log_path, 'a', encoding='utf-8') as log:
                log.write(f'\n$ {step}\n')
                log.flush()
                p = subprocess.Popen(step, shell=True, cwd=self.cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
                while p.poll() is None:
                    if time.time() > self.deadline:
                        p.terminate()
                        try:
                            p.wait(120)
                        except subprocess.TimeoutExpired:
                            p.kill()
                        break
                    time.sleep(self.poll)
            rc = p.wait()
            self.history.append({'lane': name, 'step': step, 'rc': rc, 'minutes': round((time.time() - t0) / 60, 1)})
            if self.on_step:
                self.on_step(name, step, rc)
        self.status[name] = 'done'

    def run(self, report_every=600, report=print):
        threads = [threading.Thread(target=self._run, args=(n, g, s), daemon=True) for n, (g, s) in self.lanes.items()]
        for t in threads:
            t.start()
        last = 0.0
        while any(t.is_alive() for t in threads):
            if time.time() - last >= report_every:
                report(self.snapshot())
                last = time.time()
            time.sleep(min(self.poll, 5))
        report(self.snapshot())
        return self.history

    def snapshot(self):
        lines = [time.strftime('%H:%M:%S')]
        for n in self.lanes:
            tail = ''
            p = self.logs / f'{n}.log'
            if p.is_file():
                with open(p, 'rb') as f:
                    f.seek(max(0, p.stat().st_size - 4000))
                    last = [l for l in f.read().decode('utf-8', 'replace').replace('\r', '\n').splitlines() if l.strip()]
                tail = last[-1][-160:] if last else ''
            lines.append(f'  [{n}] {self.status[n]}\n      {tail}')
        return '\n'.join(lines)
