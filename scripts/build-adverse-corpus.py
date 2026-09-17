#!/usr/bin/env python3
"""Deterministic synthetic corruption of hash-pinned, mono PCM16 recordings."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    result = {**source, 'corpus': {**source['corpus'], 'id': source['corpus']['id'] + '-generated-adverse-v1'}, 'clips': []}
    recipes = ['clean', 'white-noise-10db', 'white-noise-0db', 'echo-120-240ms']
    for clip in source['clips']:
        path = args.manifest.parent / clip['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != clip['sha256']:
            raise ValueError(f'Input hash mismatch: {path}')
        with wave.open(str(path)) as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError('Requires mono PCM16')
            rate = wav.getframerate()
            samples = array('h', wav.readframes(wav.getnframes()))
        if sys.byteorder != 'little':
            samples.byteswap()
        rms = math.sqrt(sum(float(x)*x for x in samples) / len(samples))
        for recipe in recipes:
            rng = random.Random(clip['id'] + '/' + recipe)
            if recipe.startswith('white-noise'):
                snr = 10 if '10db' in recipe else 0
                noise = [rng.gauss(0, 1) for _ in samples]
                noise_rms = math.sqrt(sum(x*x for x in noise) / len(noise))
                gain = rms / (10**(snr / 20)) / noise_rms
                mixed = [x + n*gain for x,n in zip(samples, noise)]
            elif recipe.startswith('echo'):
                delays = [(round(rate*0.12), 0.45), (round(rate*0.24), 0.25)]
                mixed = [float(x) for x in samples] + [0.] * delays[-1][0]
                for delay, gain in delays:
                    for i,x in enumerate(samples):
                        mixed[i+delay] += x*gain
            else:
                mixed = samples
            # Common attenuation preserves SNR and avoids clipping.
            scale = min(1., 32760 / max(1, max(abs(x) for x in mixed)))
            payload = array('h', (round(x*scale) for x in mixed))
            if sys.byteorder != 'little':
                payload.byteswap()
            filename = clip['id'] + '-' + recipe + '.wav'
            with wave.open(str(args.out / filename), 'wb') as wav:
                wav.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
                wav.writeframes(payload.tobytes())
            result['clips'].append({**clip, 'id': clip['id']+'-'+recipe, 'path': filename,
                'sha256': hashlib.sha256((args.out / filename).read_bytes()).hexdigest(),
                'split': recipe, 'tags': {**clip.get('tags', {}), 'recipe': recipe, 'source_sha256': clip['sha256'], 'kind': 'generated-corruption-not-real-room'}})
    (args.out / 'manifest.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
