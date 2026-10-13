import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import HashingVectorizer
from .provider import read_cache, text_key
from .data import reconstruct

IMAGE_DIM = 40
HASH_DIM = 64
NUMERIC_NAMES = ['horizon', 'since_repair', 'chemical_count', 'mechanical_count']
QUALITY_NAMES = ['image_available', 'image_coverage', 'text_available', 'semantic_available',
                 'text_age', 'text_first_seen_age', 'text_repeated', 'repair_since_text',
                 'text_time_unknown', 'text_changed', 'new_repair', 'observation_gap']
STATE_NAMES = ['text_age', 'text_first_seen_age', 'text_repeated', 'repair_since_text',
               'text_time_unknown', 'text_changed', 'new_repair', 'chemical_delta',
               'mechanical_delta', 'mechanical_reset', 'observation_gap',
               'observed_repair_count', 'mean_repair_gap']

def curve_features(path):
    """Extract the plotted trace, excluding axes and tick labels; preserve amplitude scale."""
    missing = np.full(IMAGE_DIM, np.nan)
    if not path or not Path(path).is_file():
        return missing, {'available': False, 'reason': 'missing_file', 'coverage': 0.}
    try:
        a = np.asarray(Image.open(path).convert('L'))
    except (OSError, ValueError):
        return missing, {'available': False, 'reason': 'unreadable_image', 'coverage': 0.}
    dark = a < 150
    # Locate the two long horizontal sides of the plot rectangle.
    rows = np.flatnonzero(dark.sum(axis=1) > a.shape[1] * .5)
    if len(rows) < 2:
        return missing, {'available': False, 'reason': 'axes_not_detected', 'coverage': 0.}
    top, bottom = int(rows[0]), int(rows[-1])
    xs = np.flatnonzero(dark[top])
    if len(xs) < 2 or bottom - top < 20:
        return missing, {'available': False, 'reason': 'invalid_plot', 'coverage': 0.}
    # Use the longest run, avoiding digits in tick labels.
    groups = np.split(xs, np.where(np.diff(xs)>1)[0]+1)
    line = max(groups, key=len)
    left, right = int(line[0]), int(line[-1])
    margin = max(4, int((right-left)*.025))
    crop = dark[top+margin:bottom-margin, left+margin:right-margin]
    valid = np.flatnonzero(crop.any(axis=0))
    if len(valid) < 8:
        return missing, {'available': False, 'reason': 'blank_plot', 'coverage': 0.}
    heights = np.array([1-(np.median(np.flatnonzero(crop[:,x])) + margin)/(bottom-top) for x in valid])
    xnorm = (valid + margin)/(right-left)
    # Sampling over the whole axes retains a partial trace's extent.
    grid = np.linspace(0,1,32)
    samples = np.interp(grid,xnorm,heights,left=np.nan,right=np.nan)
    stats = [np.mean(heights),np.std(heights),np.min(heights),np.max(heights),
             np.mean(np.abs(np.diff(heights))),xnorm[np.argmin(heights)],xnorm[0],xnorm[-1]]
    coverage = len(valid)/max(1,crop.shape[1])
    return np.r_[samples,stats], {'available':True,'reason':'trace_detected','coverage':float(coverage),
                                 'plot_bounds':[left,top,right,bottom]}

def semantic_features(text, cache):
    item = cache.get(text_key(text))
    if item is None:
        return np.full(6,np.nan), False
    before, after = item.get('severity_before'), item.get('severity_after')
    return np.array([np.nan if before is None else before, np.nan if after is None else after,
          float(item['maintenance']=='mechanical'),float(item['maintenance']=='chemical'),
          float(item['urgent']),float(item['maintenance']=='unknown')]), True

def build_features(records, image_cache=None, progress=None):
    state = reconstruct(records)
    cache = read_cache()
    hashing = HashingVectorizer(n_features=HASH_DIM,alternate_sign=False,norm='l2',ngram_range=(1,2))
    hashes = hashing.transform([r.text for r in records]).toarray()
    numerical, curve, text, maintenance, quality, image_meta,semantic_evidence = [], [], [], [], [], [],[]
    for i,r in enumerate(records):
        s = state[r.uid]
        num = [r.horizon,r.since_repair,r.chemical_count,r.mechanical_count]
        if image_cache is not None and r.uid in image_cache:
            img,meta=image_cache[r.uid]
        else:
            img,meta=curve_features(r.image_path)
        sem,available = semantic_features(r.text,cache)
        semantic_evidence.append(cache.get(text_key(r.text)))
        numerical.append(num)
        curve.append(np.r_[num[:2],img])
        text.append(np.r_[num[:2],sem,hashes[i]])
        maintenance.append(num + [float(s[k]) for k in STATE_NAMES])
        quality.append([float(meta['available']),meta['coverage'],float(bool(r.text)),float(available)] +
                        [float(s[k]) for k in QUALITY_NAMES[4:]])
        image_meta.append(meta)
        if progress and i%1000==0:
            progress(i,len(records))
    return {'numeric':np.asarray(numerical),'image':np.asarray(curve),
            'text':np.asarray(text),'maintenance':np.asarray(maintenance),
            'quality':np.asarray(quality),'state':state,'image_meta':image_meta,'semantic_evidence':semantic_evidence}

def fusion_matrix(predictions, features):
    return np.c_[predictions, features['numeric'],features['quality']]

def mask_features(features, hide_image=False, hide_text=False):
    f={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in features.items()}
    if hide_image:
        f['image'][:,2:] = np.nan
        f['quality'][:,0:2] = 0
    if hide_text:
        f['text'][:,2:] = np.nan
        f['quality'][:,2:4] = 0
    return f
