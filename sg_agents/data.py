from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from .config import DATASET, ARTIFACTS

NUM_FILES = {'horizon': 'nextinstime.txt', 'since_repair': 'timefromlastrepair.txt',
             'chemical_count': 'Numberofchemicalcleaning.txt',
             'mechanical_count': 'Numberofmechanicalcleaning.txt'}

@dataclass
class Record:
    device: int
    index: int
    time: float
    horizon: float
    since_repair: float
    chemical_count: float
    mechanical_count: float
    text: str
    image_path: str
    target: float | None = None

    @property
    def uid(self):
        return f'{self.device}:{self.index}'

    @property
    def next_inspection(self):
        return self.time + self.horizon

    @property
    def last_repair(self):
        return self.time - self.since_repair

def read_vector(path):
    return np.atleast_1d(np.loadtxt(path)).reshape(-1)

def load_dataset(path=DATASET, with_targets=True):
    path = Path(path)
    if not path.is_dir():
        raise ValueError('找不到数据集目录')
    records = []
    dirs = sorted((d for d in path.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d:int(d.name))
    for d in dirs:
        times = read_vector(d / 'time.txt')
        nums = {k: read_vector(d / 'Numerical' / v) for k, v in NUM_FILES.items()}
        texts = (d / 'workorder.txt').read_text(encoding='utf-8-sig').splitlines()
        targets = read_vector(d / 'nextinslevel.txt') if with_targets else [None] * len(times)
        lengths = [len(times), len(texts), len(targets)] + [len(v) for v in nums.values()]
        if len(set(lengths)) != 1:
            raise ValueError(f'设备 {d.name} 的字段长度不一致：{lengths}')
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError(f'设备 {d.name} 的观察时间必须严格递增且有限')
        for k, v in nums.items():
            if not np.all(np.isfinite(v)) or np.any(v < 0):
                raise ValueError(f'设备 {d.name} 的 {k} 含非有限值或负数')
        if with_targets and not np.all(np.isfinite(targets)):
            raise ValueError(f'设备 {d.name} 标签含非有限值')
        for i, t in enumerate(times):
            image = d / 'Image' / f'{i+1}.tiff'
            records.append(Record(int(d.name), i+1, float(t),
                  **{k:float(v[i]) for k,v in nums.items()}, text=texts[i].strip().strip('"'),
                  image_path=str(image.resolve()) if image.exists() else '',
                  target=float(targets[i]) if with_targets else None))
    if not records:
        raise ValueError('没有设备记录')
    return records

def fingerprint(records):
    h = hashlib.sha256()
    for r in records:
        d = asdict(r)
        image = Path(r.image_path)
        d.pop('image_path')
        d['image_sha256'] = hashlib.sha256(image.read_bytes()).hexdigest() if image.is_file() else None
        h.update(json.dumps(d, sort_keys=True).encode())
    return h.hexdigest()

def trajectory_groups(records):
    """Conservatively group devices with identical time, numeric and text histories.

    Labels are excluded. Image file metadata cannot make a repeated trajectory independent.
    """
    histories={}
    for r in records:
        histories.setdefault(r.device,[]).append(r)
    groups={};mapping={}
    for device in sorted(histories):
        h=hashlib.sha256()
        for r in sorted(histories[device],key=lambda r:r.time):
            h.update(json.dumps([r.time,r.horizon,r.since_repair,r.chemical_count,
                                 r.mechanical_count,r.text]).encode())
        key=h.hexdigest()
        if key not in groups:groups[key]=device
        mapping[device]=groups[key]
    return mapping

def reconstruct(records):
    """Causal state reconstruction. Targets are deliberately never accessed."""
    output = {}
    by_device = {}
    for r in records:
        by_device.setdefault(r.device, []).append(r)
    for device, rows in by_device.items():
        rows = sorted(rows, key=lambda r:r.time)
        first_seen = {}
        previous = None
        run_start = None
        observed_repairs = []
        for r in rows:
            first_seen.setdefault(r.text, r.time)
            changed = previous is not None and r.text != previous.text
            if previous is None or changed:
                run_start = r.time
            new_repair = previous is not None and r.last_repair > previous.last_repair + 1e-5
            if new_repair:
                observed_repairs.append(r.last_repair)
            chemical_delta = max(0., r.chemical_count - previous.chemical_count) if previous else 0.
            mechanical_delta = max(0., r.mechanical_count - previous.mechanical_count) if previous else 0.
            output[r.uid] = {
                'text_first_seen_at': first_seen[r.text], 'text_run_start': run_start,
                'text_age': r.time - run_start, 'text_first_seen_age': r.time - first_seen[r.text],
                'text_repeated': previous is not None and not changed,
                'text_time_unknown': True, 'text_changed': changed,
                'repair_since_text': r.last_repair > run_start + 1e-5,
                'new_repair': new_repair, 'last_repair_at': r.last_repair,
                'repair_time_invalid': r.last_repair < -1e-5 or (previous is not None and r.last_repair < previous.last_repair - 1e-5),
                'chemical_delta': chemical_delta, 'mechanical_delta': mechanical_delta,
                'chemical_counter_decreased': previous is not None and r.chemical_count < previous.chemical_count,
                'mechanical_reset': previous is not None and r.mechanical_count < previous.mechanical_count,
                'observation_gap': r.time - previous.time if previous else 0.,
                'observed_repair_count': len(observed_repairs),
                'mean_repair_gap': float(np.mean(np.diff(observed_repairs))) if len(observed_repairs)>1 else 0.,
                'event_interval_start': previous.time if previous else r.time,
                'event_interval_end': r.time,
            }
            previous = r
    return output
