#!/usr/bin/env python3
"""Offline closed-loop oracle for the LMPC lateral loop (T-LATERAL-LMPC-SPEED-MARGIN, 2026-10-11).

Runs the PRODUCTION MPCSolver (params from config/av_stack_config.yaml) against a plant built from
measured pieces -- first-order steering actuator (tau 0.25 s, Unity steeringTimeConstant), command->angle
gain ratio 0.85, 2-frame (100 ms) pipeline delay, Smith predictor as in pid_controller, lookahead
error input e_lat = d + 6.3 sin(psi) (fit on recordings, R^2 0.999), understeer plant L_eff = L(1+K_us v^2)
-- on an R200 arc at several speeds, and reports growth of the lateral mode for controller variants.

FIDELITY STATUS: directionally right (ideal plant stable everywhere; measured plant marginal at
11.6 m/s), NOT yet validated against the recordings: production with the measured plant and no stack
limiters is stable at 9.8 (e_rms 0.09) and growth x1.6 at 11.6, but the plant's measured lateral
response per command rises 2.4x from 9.8 to 11.6 m/s where this plant rises 1.25x. Use it to rank
candidate fixes, not to certify them; a plant-identification run (open-loop steering chirps at
8-14 m/s) is the prerequisite for a certifying oracle. See tasks.md T-LATERAL-LMPC-SPEED-MARGIN.

Usage: python3 tools/analyze/lmpc_closed_loop_oracle.py
"""
import sys, math, json, numpy as np, dataclasses
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from av_stack.config import load_config
from control.mpc_controller import MPCParams, MPCSolver
cfg=load_config(None); base=MPCParams.from_config(cfg)
L=2.5; FRAME_DT=0.05; R=200.0
def max_steer(v): return float(np.interp(v,[0,12],[0.5236,0.2793]))
def run(params, v, tau=0.25, gain_ratio=0.85, delay_frames=2, smith_cap=0.08, T=60.0, e0=0.10, seed=0, Ld=6.3, understeer=True, rate_lim=0.044, jerk_lim=18.0):
    solver=MPCSolver(params); N=solver._N; kappa=1.0/R
    e, psi, delta_act, last_cmd = e0, 0.0, 0.0, 0.0
    buf=[0.0]*delay_frames; es=[]; cmds=[]; prev_cmd=0.0; prev_rate=0.0
    L_plant = 2.5*(1+0.0046*v*v) if understeer else L   # sysid: L_eff 3.65 m at 10 m/s (K_us from gain 0.684)
    rng=np.random.default_rng(seed)
    for k in range(int(T/FRAME_DT)):
        # Smith predictor as in pid_controller (legacy path)
        total_delay=min(delay_frames*FRAME_DT, smith_cap); gain=1.0 if e*psi>=0 else 0.3
        e_look = e + Ld*math.sin(psi)            # production feeds the LOOKAHEAD GT cross-track (fit: d + 6.3 sin ψ, R² 0.999)
        pred_e = e_look + gain*v*psi*total_delay
        res=solver.solve(e_lat=pred_e, e_heading=psi, v=v, last_delta_norm=last_cmd, kappa_ref_horizon=np.full(N,kappa),
                         v_target=12.0, v_max=15.0, dt=params.dt, delta_actual_norm=delta_act/max_steer(v), lookahead_distance=0.0)
        raw=float(np.clip(res['steering_normalized'],-0.7,0.7))
        # stack post-processing: per-frame rate limit, then jerk limit on the rate change, then hard clip
        rate=float(np.clip(raw-prev_cmd,-rate_lim,rate_lim)) if rate_lim else raw-prev_cmd
        if jerk_lim:
            max_drate=jerk_lim*FRAME_DT*FRAME_DT; rate=float(np.clip(rate, prev_rate-max_drate, prev_rate+max_drate))
        cmd=float(np.clip(prev_cmd+rate,-0.7,0.7)); prev_rate=cmd-prev_cmd; prev_cmd=cmd; last_cmd=cmd
        buf.append(cmd); applied=buf.pop(0)
        desired=gain_ratio*applied*max_steer(v)
        step=(FRAME_DT/tau)*(desired-delta_act); maxd=math.radians(90.0)*FRAME_DT
        delta_act+=float(np.clip(step,-maxd,maxd))
        psi+= (v*math.tan(delta_act)/L_plant - v*kappa)*FRAME_DT
        e  += v*math.sin(psi)*FRAME_DT + 0.002*rng.standard_normal()
        es.append(e); cmds.append(cmd)
        if abs(e)>3.0: break
    es=np.array(es); cm=np.array(cmds); n=len(es)
    a=es[int(0.15*n):int(0.4*n)].std(); b=es[int(0.65*n):].std() if n>100 else np.nan
    x=es-es.mean(); f=np.fft.rfftfreq(len(x),FRAME_DT); P=np.abs(np.fft.rfft(x*np.hanning(len(x))))**2; P[f<0.1]=0
    return dict(frames=n, diverged=n<int(T/FRAME_DT), e_std_early=a, e_std_late=b, growth=b/max(a,1e-6), f_dom=float(f[np.argmax(P)]), steer_p95=float(np.percentile(np.abs(cm),95)), e_rms=float(np.sqrt(np.mean(es**2))))
def variant(**kw): return dataclasses.replace(base, **kw)
V={'production': base,
   'delay compensation on (d=2)': variant(delay_compensation_enabled=True, delay_compensation_frames=2),
   'actuator model tau=0.25': variant(actuator_model_enabled=True, actuator_tau_s=0.25),
   'actuator model tau=0.70 (April default)': variant(actuator_model_enabled=True, actuator_tau_s=0.70),
   'steer_gain_ratio 0.85 (dyn only - noop?)': variant(steer_gain_ratio=0.85),
   'r_steer_rate sched onset 9 m/s': variant(r_steer_rate_scheduling_enabled=True, r_steer_rate_speed_onset=9.0),
   'r_steer_rate x2': variant(r_steer_rate=base.r_steer_rate*2),
}
plants={'measured plant + understeer + stack limiters':dict(), 'same, no stack limiters':dict(rate_lim=0.0, jerk_lim=0.0), 'same, kinematic plant (no understeer)':dict(understeer=False), 'ideal plant (no lag, gain 1, no delay)':dict(tau=1e-3, gain_ratio=1.0, delay_frames=1, smith_cap=0.0)}
print(f"base params: q_lat {base.q_lat} r_steer_rate {base.r_steer_rate} sched {base.r_steer_rate_scheduling_enabled} onset {base.r_steer_rate_speed_onset} dt {base.dt} N {base.horizon} actuator {base.actuator_model_enabled}")
for pname,pkw in plants.items():
    print(f"\n##### {pname}")
    for name,p in V.items():
        row=[]
        for v in (9.8, 11.6, 13.0, 15.0):
            r=run(p,v,**pkw); row.append(f"v{v:4.1f}: growth x{r['growth']:4.1f} f {r['f_dom']:.2f}Hz e_rms {r['e_rms']:.2f} steer p95 {r['steer_p95']:.2f}{' DIVERGED' if r['diverged'] else ''}")
        print(f"  {name:42s} | "+' | '.join(row))
