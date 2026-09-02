# Pilot for docs/verified/dense_annotations.md (2026-09-01). Deliberately NOT the design:
# it uses f_touch = 0.05 N (runtime in_contact is 0.1 N), its own "both pads + lifted" carry
# (the runtime carry is contact streak + coupling + hand motion, see GraspTracker), a fixed
# TCP offset measured on t1, and root_velocity for object motion (use windowed displacement).
# Run from a directory of trial run dirs:  python dense_phases_pilot.py t1_BananaInBowl:0 ...

"""Pilot v2: contiguous phase labels from simulator state alone, all objects, TCP-corrected,
smoothed. Runs over every episode of every run dir given. No robolab import."""
import json, sys, glob, os, collections
import numpy as np, h5py

DT = 1 / 15.0
TCP_OFF = np.array([0.15, 0.03, 0.0])   # Robotiq base_link -> fingertip midpoint, EE frame (measured on t1)
PAD_F = 0.05        # N, a pad "touches" an object
V_EE = 0.02         # m/s, TCP considered moving
V_OBJ = 0.02        # m/s, object considered moving
NEAR = 0.08         # m, TCP within reach of an object
LIFT = 0.01         # m above its own start height

def quat_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def roles(cfg):
    targets, dests = [], []
    for k, v in cfg['terminations'].items():
        if not (isinstance(v, dict) and 'params' in v) or k == 'time_out' or 'lost' in k: continue
        p = v['params']
        o = p.get('object') or p.get('object_a') or p.get('objects')
        if o: targets += (o if isinstance(o, list) else [o])
        for dk in ('container', 'surface', 'object_b', 'target'):
            if p.get(dk) and p[dk] != 'table': dests.append(p[dk])
    return sorted(set(targets)), sorted(set(dests))

def episode(d, cfg):
    T = d['actions'].shape[0]
    objs = [o for o in d['states/rigid_object'].keys() if o != 'table']
    targets, dests = roles(cfg)
    ee = d['ee_pose/position'][:]; eq = d['ee_pose/orientation'][:]
    tcp = np.array([ee[t] + quat_R(eq[t]) @ TCP_OFF for t in range(T)])
    tcpv = np.linalg.norm(np.gradient(tcp, DT, axis=0), axis=1)
    grip_cmd = d['actions'][:, 7] > 0.5
    pos = {o: d[f'states/rigid_object/{o}/root_pose'][:, :3] for o in objs}
    vel = {o: np.linalg.norm(d[f'states/rigid_object/{o}/root_velocity'][:, :3], axis=1) for o in objs}
    pads = {o: d[f'contact/{o}'][:] for o in objs if f'contact/{o}' in d}
    tablef = d['contact/table'][:] if 'contact/table' in d else np.zeros((T, 2))
    lab = []; obj_of = []
    for t in range(T):
        touch = {o: (pads[o][t] > PAD_F) for o in pads}
        both = [o for o in touch if touch[o].all()]
        anyp = [o for o in touch if touch[o].any()]
        dist = {o: np.linalg.norm(tcp[t] - pos[o][t]) for o in objs}
        nearest = min(dist, key=dist.get)
        moving_objs = [o for o in objs if vel[o][t] > V_OBJ and o not in anyp]
        lifted = [o for o in both if pos[o][t, 2] - pos[o][0, 2] > LIFT]
        role = lambda o: 'target' if o in targets else ('dest' if o in dests else 'other')
        if t < 3: l, o = 'settle', ''
        elif lifted:
            o = lifted[0]
            vz = np.gradient(pos[o][:, 2], DT)[t]
            if vz > 0.03: l = 'lift'
            elif vz < -0.03: l = 'lower'
            else: l = 'transport' if tcpv[t] > V_EE else 'hold'
            l += f'[{role(o)}]'
        elif both and grip_cmd[t]: o = both[0]; l = f'gripped_static[{role(o)}]'
        elif anyp and grip_cmd[t]: o = anyp[0]; l = f'closing_on[{role(o)}]'
        elif anyp: o = anyp[0]; l = f'touch_open[{role(o)}]'
        elif (tablef[t] > PAD_F).any(): l, o = 'gripper_on_table', ''
        elif grip_cmd[t]: l, o = 'closed_empty', ''
        elif moving_objs: o = moving_objs[0]; l = f'obj_moving_free[{role(o)}]'
        elif tcpv[t] > V_EE:
            o = nearest
            dn = np.gradient(np.linalg.norm(tcp - pos[o], axis=1), DT)[t]
            l = (f'approach[{role(o)}]' if dist[o] < 0.25 else f'reach[{role(o)}]') if dn < -0.01 else ('retreat' if dn > 0.01 else 'move')
        else:
            o = nearest if dist[nearest] < NEAR else ''
            l = f'hover[{role(o)}]' if o else 'idle'
        lab.append(l); obj_of.append(o)
    return lab, obj_of, targets, dests

def smooth(lab, w=5):
    out = list(lab)
    for t in range(len(lab)):
        win = lab[max(0, t-w//2): t+w//2+1]
        out[t] = collections.Counter(win).most_common(1)[0][0]
    return out

def segments(lab, min_len=4):
    segs = []; s = 0
    for t in range(1, len(lab)+1):
        if t == len(lab) or lab[t] != lab[s]:
            segs.append([s, t-1, lab[s]]); s = t
    out = []
    for sg in segs:
        if out and sg[1]-sg[0]+1 < min_len: out[-1][1] = sg[1]
        else: out.append(sg)
    # merge equal neighbours produced by absorption
    m = []
    for sg in out:
        if m and m[-1][2] == sg[2]: m[-1][1] = sg[1]
        else: m.append(sg)
    return m

def main(run_dirs, show):
    stats = []
    for rd in run_dirs:
        td = glob.glob(os.path.join(rd, '*Task'))[0]
        cfg = json.load(open(os.path.join(td, 'env_cfg.json')))
        f = h5py.File(os.path.join(td, 'run_0.hdf5'), 'r')
        for k in sorted(f['data'].keys()):
            env = int(k.split('_')[1]); d = f['data'][k]
            log = json.load(open(os.path.join(td, f'log_0_env{env}.json')))
            lab, obj_of, targets, dests = episode(d, cfg)
            sl = smooth(lab); segs = segments(sl)
            T = len(lab)
            cov = collections.Counter(sl)
            stats.append((os.path.basename(rd), env, log['success'], T, len(segs), len(log['events']), cov))
            if f'{os.path.basename(rd)}:{env}' in show:
                ev = collections.defaultdict(list)
                for e in log['events']: ev[e['step']].append(e['name'])
                print(f"\n##### {os.path.basename(rd)} env{env} targets={targets} dests={dests} success={log['success']} T={T}")
                for a, b, l in segs:
                    inside = ' '.join(f"{st}:{','.join(n)}" for st, n in sorted(ev.items()) if a <= st <= b)
                    print(f"{a:4d}-{b:4d} {a*DT:5.1f}-{b*DT:5.1f}s  {l:26s}| {inside}")
                print(f"-> {len(segs)} segments vs {len(log['events'])} events")
    print("\n##### corpus summary (24 episodes)")
    print(f"{'run':28s} env ok   T  segs  ev  seg/10s  top phases (fraction of steps)")
    for run, env, ok, T, ns, ne, cov in stats:
        top = ', '.join(f"{k}:{v/T:.2f}" for k, v in cov.most_common(4))
        print(f"{run:28s} {env}  {int(ok)} {T:4d}  {ns:3d}  {ne:2d}  {ns/(T*DT)*10:5.1f}  {top}")
    allcov = collections.Counter()
    for *_, cov in stats: allcov.update(cov)
    tot = sum(allcov.values())
    print("\nphase share over all 24 episodes:")
    for k, v in allcov.most_common(): print(f"  {k:28s} {v/tot:6.3f}")

if __name__ == '__main__':
    runs = sorted(r for r in glob.glob('t*') if glob.glob(os.path.join(r, '*Task')))
    show = set(sys.argv[1:])
    main(runs, show)
