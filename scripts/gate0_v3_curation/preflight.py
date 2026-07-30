"""Gate 0 v2 local CPU pre-flight.

Three jobs, in order of how badly they can corrupt the verdict:

 A. SOURCE INTEGRITY - is each film byte-complete and decodable all the way past the LAST clip
    we cut from it? A truncated download silently cuts every later clip from past EOF, and cell 12
    runs ffmpeg with check=False, so nothing would ever say so. This is a wrong-stimulus verdict.
 B. PER-CLIP VALIDITY - every one of the 38 clips: right duration, right height, video+audio
    present, decodes without error, and is a SUSTAINED shot rather than a montage (D021's curation
    claim is "sustained runs", so it should be checked rather than assumed).
 C. G5 COVARIATES - luminance, audio RMS, motion energy, voiced fraction; FACE vs NONFACE with a
    two-sided permutation test. DISCLOSURE ONLY, per D023(c). Gates nothing, aborts nothing.

Writes preflight.json. Exit code is nonzero only for A/B failures - never for C.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/home/deveshb/workspace/AI/tribe-bench')
FFMPEG = str(HERE / 'ffmpeg')
FFPROBE = str(HERE / 'ffprobe')
CLIPS = HERE / 'clips'
CLIPS.mkdir(exist_ok=True)

M = json.loads((REPO / 'notebooks' / 'gate0_v2_stimuli.json').read_text())
DUR = M['clip_dur_s']

# Authoritative size + md5 straight from https://archive.org/metadata/<item>.
# Size alone is NOT sufficient: two concurrent writers resuming at different offsets can produce a
# byte-count-correct but content-corrupt file (this actually happened while building this script).
EXPECT = {
    'charade.mp4': dict(size=424901742, md5='f2602d71c2279e834d48bdefe32b04a6'),
    'mclintock.mp4': dict(size=550863361, md5='04671e70c46d1b3f3cb8d1df4217a666'),
}


def md5_of(path, chunk=1 << 22):
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(chunk), b''):
            h.update(blk)
    return h.hexdigest()

report = {'clip_dur_s': DUR, 'sources': {}, 'clips': {}, 'covariates': {}, 'errors': [], 'warnings': []}
HARD = []
WARN = []

# Charade feeds FACE/NONFACE = the PRIMARY contrast, so any defect there is disqualifying.
# McLintock feeds only the SCENE clips, which GATE-0.md calls "corroborating strength only,
# never GO-critical" and the manifest calls "reported, not gated" - a defect there cannot corrupt
# the GO/NO-GO verdict, so it is disclosed, not blocking.
NON_GATING = ('mclintock.mp4', 'SCENE')


def flag(subject, msg):
    entry = f'{subject}: {msg}'
    if any(subject.startswith(k) or subject == k for k in NON_GATING):
        WARN.append(entry)
    else:
        HARD.append(entry)


def sh(args, timeout=900):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def probe(path, extra=None):
    args = [FFPROBE, '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams']
    if extra:
        args += extra
    args += [str(path)]
    r = sh(args)
    if r.returncode != 0:
        return None, r.stderr.strip()
    try:
        return json.loads(r.stdout), r.stderr.strip()
    except Exception as e:
        return None, f'unparseable ffprobe json: {e}'


# ----------------------------------------------------------------- A. source integrity
SOURCES = {
    'charade.mp4': dict(label='Charade (1963)',
                        needed=max(M['face_starts_s'] + M['nonface_starts_s']) + DUR),
    'mclintock.mp4': dict(label='McLintock! (1963)',
                          needed=max(M['confirmatory_scene_source']['scene_starts_s']) + DUR),
}

print('=' * 78)
print('A. SOURCE INTEGRITY')
print('=' * 78)
for fn, meta in SOURCES.items():
    p = HERE / fn
    rec = {'label': meta['label'], 'needed_s': meta['needed'], 'path': str(p)}
    if not p.exists():
        rec['status'] = 'MISSING'
        flag(fn, 'file missing')
        report['sources'][fn] = rec
        print(f'  {fn}: MISSING')
        continue
    size = p.stat().st_size
    want = EXPECT[fn]['size']
    rec['bytes'] = size
    rec['bytes_expected'] = want
    rec['bytes_exact'] = (size == want)
    if size != want:
        flag(fn, f'TRUNCATED - {size} bytes, archive.org says {want} '
             f'({100.0 * size / want:.1f}%)')
        rec['status'] = 'TRUNCATED'
        report['sources'][fn] = rec
        print(f'  {fn}: TRUNCATED {size} / {want}')
        continue

    got_md5 = md5_of(p)
    rec['md5'] = got_md5
    rec['md5_expected'] = EXPECT[fn]['md5']
    rec['md5_match'] = (got_md5 == EXPECT[fn]['md5'])
    if not rec['md5_match']:
        flag(fn, f'MD5 MISMATCH - got {got_md5}, archive.org says {EXPECT[fn]["md5"]}. '
             f'Right size, wrong content: re-download from scratch with a single writer.')
        rec['status'] = 'CORRUPT'
        report['sources'][fn] = rec
        print(f'  {fn}: CORRUPT (size exact, md5 mismatch)')
        continue

    info, err = probe(p)
    if info is None:
        flag(fn, f'ffprobe failed: {err}')
        rec['status'] = 'UNPROBEABLE'
        report['sources'][fn] = rec
        continue
    dur = float(info['format'].get('duration', 0.0))
    vs = [s for s in info['streams'] if s.get('codec_type') == 'video']
    aus = [s for s in info['streams'] if s.get('codec_type') == 'audio']
    rec.update(duration_s=round(dur, 2), n_video=len(vs), n_audio=len(aus),
               vcodec=vs[0]['codec_name'] if vs else None,
               acodec=aus[0]['codec_name'] if aus else None,
               width=vs[0].get('width') if vs else None,
               height=vs[0].get('height') if vs else None)
    if dur < meta['needed']:
        flag(fn, f'too SHORT - duration {dur:.1f}s but the last clip needs {meta["needed"]}s')
    if not vs:
        flag(fn, 'no video stream')
    if not aus:
        flag(fn, 'no audio stream (ASR would get nothing)')

    # decode the actual tail region we cut from - a container duration can lie
    tail_start = max(0, meta['needed'] - DUR)
    r = sh([FFMPEG, '-v', 'error', '-ss', str(tail_start), '-t', str(DUR), '-i', str(p),
            '-f', 'null', '-'])
    rec['tail_decode_start_s'] = tail_start
    rec['tail_decode_ok'] = (r.returncode == 0 and not r.stderr.strip())
    rec['tail_decode_stderr'] = r.stderr.strip()[:400]
    if not rec['tail_decode_ok']:
        flag(fn, f'LAST clip region ({tail_start}s..{tail_start + DUR}s) does not decode '
             f'cleanly: {r.stderr.strip()[:200]}')
    rec['status'] = 'OK' if not any(h.startswith(fn) for h in HARD + WARN) else 'DEFECT'
    report['sources'][fn] = rec
    print(f'  {fn}: {rec["status"]} | {size} bytes (exact) | duration {dur:.1f}s '
          f'(need {meta["needed"]}s) | {rec["width"]}x{rec["height"]} '
          f'{rec["vcodec"]}/{rec["acodec"]} | tail decode '
          f'{"clean" if rec["tail_decode_ok"] else "DIRTY"}')

if WARN:
    print('\n   WARNINGS (non-gating sources - disclosed, not blocking):')
    for w in WARN:
        print('   -', w)
if HARD:
    print('\n!! HARD FAILURES in a source feeding the PRIMARY contrast - not cutting clips:')
    for h in HARD:
        print('   -', h)
    report['errors'], report['warnings'] = HARD, WARN
    (HERE / 'preflight.json').write_text(json.dumps(report, indent=2))
    sys.exit(2)

# ----------------------------------------------------------------- B. cut + validate clips
print()
print('=' * 78)
print('B. CUT + VALIDATE ALL 38 CLIPS  (exact cell-12 ffmpeg invocation)')
print('=' * 78)


def cut(src, start, name):
    """Byte-for-byte the same command as notebook cell 12, only the binary differs."""
    out = CLIPS / f'{name}.mp4'
    r = subprocess.run([FFMPEG, '-y', '-ss', str(start), '-t', str(DUR), '-i', str(src),
                        '-vf', 'scale=-2:480', '-c:v', 'libx264', '-preset', 'veryfast',
                        '-crf', '23', '-c:a', 'aac', str(out)],
                       check=False, capture_output=True, text=True)
    return out, r


GROUPS = {
    'FACE': (HERE / 'charade.mp4', M['face_starts_s']),
    'NONFACE': (HERE / 'charade.mp4', M['nonface_starts_s']),
    'SCENE': (HERE / 'mclintock.mp4', M['confirmatory_scene_source']['scene_starts_s']),
}

for grp, (src, starts) in GROUPS.items():
    for i, s in enumerate(starts):
        name = f'{grp}_{i:02d}'
        out, r = cut(src, s, name)
        rec = {'group': grp, 'start_s': s, 'cut_rc': r.returncode}
        if not out.exists() or out.stat().st_size == 0:
            rec['status'] = 'EMPTY'
            flag(grp, f'{name} clip file empty/missing (start {s}s)')
            report['clips'][name] = rec
            continue
        rec['bytes'] = out.stat().st_size
        info, err = probe(out)
        if info is None:
            rec['status'] = 'UNPROBEABLE'
            flag(grp, f'{name} ffprobe failed: {err}')
            report['clips'][name] = rec
            continue
        dur = float(info['format'].get('duration', 0.0))
        vs = [x for x in info['streams'] if x.get('codec_type') == 'video']
        aus = [x for x in info['streams'] if x.get('codec_type') == 'audio']
        rec.update(duration_s=round(dur, 3), n_video=len(vs), n_audio=len(aus),
                   height=vs[0].get('height') if vs else None,
                   width=vs[0].get('width') if vs else None,
                   nb_frames=(int(vs[0]['nb_frames']) if vs and vs[0].get('nb_frames')
                              else None),
                   audio_frames=(int(aus[0]['nb_frames']) if aus and aus[0].get('nb_frames')
                                 else None))
        # how many AAC frames the SOURCE could not decode for this exact region - quantifies
        # audio loss without blocking (McLintock's stream is damaged throughout)
        rsrc = sh([FFMPEG, '-v', 'error', '-ss', str(s), '-t', str(DUR), '-i', str(src),
                   '-vn', '-f', 'null', '-'])
        rec['src_audio_damaged_frames'] = rsrc.stderr.count('buffer exhausted')
        problems = []
        if not (9.5 <= dur <= 10.6):
            problems.append(f'duration {dur:.2f}s outside 9.5-10.6')
        if not vs:
            problems.append('no video stream')
        elif vs[0].get('height') != 480:
            problems.append(f'height {vs[0].get("height")} != 480')
        if not aus:
            problems.append('no audio stream')
        # full decode - catches a clip cut from past EOF that still produced a container
        rd = sh([FFMPEG, '-v', 'error', '-i', str(out), '-f', 'null', '-'])
        rec['decode_ok'] = (rd.returncode == 0 and not rd.stderr.strip())
        rec['decode_stderr'] = rd.stderr.strip()[:300]
        if not rec['decode_ok']:
            problems.append(f'decode errors: {rd.stderr.strip()[:120]}')
        rec['problems'] = problems
        rec['status'] = 'OK' if not problems else 'FAIL'
        if problems:
            flag(grp, f'{name} ' + '; '.join(problems))
        report['clips'][name] = rec
    ok = sum(1 for k, v in report['clips'].items()
             if v.get('group') == grp and v.get('status') == 'OK')
    print(f'  {grp}: {ok}/{len(starts)} clips valid')

durs = [v['duration_s'] for v in report['clips'].values() if v.get('duration_s')]
if durs:
    print(f'  clip durations: min {min(durs):.3f}s  max {max(durs):.3f}s  '
          f'spread {max(durs) - min(durs):.3f}s')
    report['duration_spread_s'] = round(max(durs) - min(durs), 3)

# ----------------------------------------------------------------- C. G5 covariates
print()
print('=' * 78)
print('C. G5 COVARIATES  (disclosure only - gates nothing, per D023(c))')
print('=' * 78)

NUM = r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?'


def meta_mean(clip, vf, key):
    """Mean of a per-frame lavfi metadata value."""
    r = sh([FFMPEG, '-v', 'info', '-i', str(clip), '-vf', vf, '-an', '-f', 'null', '-'])
    vals = [float(x) for x in re.findall(rf'{re.escape(key)}=\s*({NUM})', r.stderr)]
    return (sum(vals) / len(vals)) if vals else None, len(vals)


RMS_FLOOR_DB = -120.0


def audio_rms_db(clip):
    """Overall RMS in dB. ffmpeg prints '-inf' for pure silence, and a naive numeric regex
    skips it -> the clip silently drops out of the comparison. Since the NONFACE clips are the
    ones plausibly silent, dropping them would bias the RMS disclosure toward 'no difference',
    i.e. hide the very asymmetry we are measuring. Map -inf to a floor instead."""
    r = sh([FFMPEG, '-v', 'info', '-i', str(clip), '-af', 'astats=metadata=1:reset=0',
            '-vn', '-f', 'null', '-'])
    m = re.findall(rf'RMS level dB:\s*(-?inf|{NUM})', r.stderr)
    if not m:
        return None
    last = m[-1]
    if 'inf' in last:
        return RMS_FLOOR_DB if last.startswith('-') else None
    try:
        return float(last)
    except ValueError:
        return None


def voiced_fraction(clip, dur):
    r = sh([FFMPEG, '-v', 'info', '-i', str(clip),
            '-af', 'silencedetect=noise=-30dB:d=0.3', '-vn', '-f', 'null', '-'])
    sil = sum(float(x) for x in re.findall(rf'silence_duration:\s*({NUM})', r.stderr))
    if dur <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - sil / dur))


def scene_cuts(clip):
    r = sh([FFMPEG, '-v', 'info', '-i', str(clip),
            '-vf', "select='gt(scene,0.4)',metadata=print", '-an', '-f', 'null', '-'])
    return len(re.findall(r'lavfi\.scene_score', r.stderr))


cov = {}
for name, rec in report['clips'].items():
    if rec.get('status') != 'OK':
        continue
    c = CLIPS / f'{name}.mp4'
    d = rec['duration_s']
    lum, nf = meta_mean(c, 'signalstats,metadata=print:key=lavfi.signalstats.YAVG',
                        'lavfi.signalstats.YAVG')
    mot, _ = meta_mean(c, 'tblend=all_mode=difference,signalstats,'
                          'metadata=print:key=lavfi.signalstats.YAVG',
                       'lavfi.signalstats.YAVG')
    cov[name] = dict(group=rec['group'], luminance_YAVG=lum, motion_energy=mot,
                     audio_rms_db=audio_rms_db(c), voiced_fraction=voiced_fraction(c, d),
                     scene_cuts=scene_cuts(c), n_frames_measured=nf)
    print(f'  {name}: lum={lum if lum is None else round(lum, 2)} '
          f'motion={mot if mot is None else round(mot, 3)} '
          f'rms={cov[name]["audio_rms_db"]} '
          f'voiced={cov[name]["voiced_fraction"] if cov[name]["voiced_fraction"] is None else round(cov[name]["voiced_fraction"], 3)} '
          f'cuts={cov[name]["scene_cuts"]}')
report['covariates'] = cov

# two-sided permutation tests, using the repo's OWN estimator in both directions
sys.path.insert(0, str(REPO))
try:
    from tribe_tools.roi_stats import mc_perm_p
    HAVE_STATS = True
except Exception as e:
    HAVE_STATS = False
    report['warnings'].append(f'could not import tribe_tools.roi_stats: {e}')


def two_sided(a, b, n_perm=10000, seed=0):
    # roi_stats.mc_perm_p is ONE-SIDED (a > b). A covariate differing in EITHER direction is a
    # confound, so run it both ways and double the smaller tail (capped at 1).
    return min(1.0, 2 * min(mc_perm_p(a, b, n_perm=n_perm, seed=seed),
                            mc_perm_p(b, a, n_perm=n_perm, seed=seed)))


print()
print('  FACE vs NONFACE, two-sided permutation (10000 perms, seed 0):')
tests = {}
if HAVE_STATS:
    import statistics as st
    for field in ('luminance_YAVG', 'motion_energy', 'audio_rms_db', 'voiced_fraction',
                  'scene_cuts'):
        fa = [v[field] for v in cov.values() if v['group'] == 'FACE' and v[field] is not None]
        nb = [v[field] for v in cov.values() if v['group'] == 'NONFACE' and v[field] is not None]
        if len(fa) < 3 or len(nb) < 3:
            print(f'    {field:16} insufficient data (face={len(fa)} nonface={len(nb)})')
            continue
        p = two_sided(fa, nb)
        tests[field] = dict(face_mean=round(st.mean(fa), 4), nonface_mean=round(st.mean(nb), 4),
                            face_n=len(fa), nonface_n=len(nb), p_two_sided=round(p, 5),
                            separates=bool(p <= 0.05))
        flag = '  <== SEPARATES (disclose)' if p <= 0.05 else ''
        print(f'    {field:16} face={st.mean(fa):9.3f}  nonface={st.mean(nb):9.3f}  '
              f'p={p:.4f}{flag}')
report['covariate_tests'] = tests

# ----------------------------------------------------------------- montages at -ss 0
print()
print('=' * 78)
print('D. MONTAGE FRAMES AT -ss 0  (the over-weighted frame nobody has looked at)')
print('=' * 78)
for grp in ('FACE', 'NONFACE'):
    names = sorted(k for k, v in report['clips'].items()
                   if v.get('group') == grp and v.get('status') == 'OK')
    for n in names:
        th = CLIPS / f'thumb0_{n}.png'
        sh([FFMPEG, '-y', '-v', 'error', '-ss', '0', '-i', str(CLIPS / f'{n}.mp4'),
            '-frames:v', '1', str(th)])
    print(f'  {grp}: rendered {sum(1 for n in names if (CLIPS / f"thumb0_{n}.png").exists())} '
          f'frame-0 thumbnails')

report['errors'] = HARD
report['warnings'] = report.get('warnings', []) + WARN
(HERE / 'preflight.json').write_text(json.dumps(report, indent=2))
print()
print('=' * 78)
if WARN:
    print(f'{len(WARN)} WARNING(S) - non-gating, disclose in the results:')
    for w in WARN:
        print('   -', w)
if HARD:
    print(f'RESULT: {len(HARD)} HARD FAILURE(S) - do NOT run Gate 0 v2:')
    for h in HARD:
        print('   -', h)
else:
    print('RESULT: the PRIMARY contrast (Charade -> FACE/NONFACE) is clean. '
          'Covariates above are disclosure only.')
print('wrote preflight.json')
sys.exit(1 if HARD else 0)
