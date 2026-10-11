#!/usr/bin/env python3
"""Calibrated offline closed-loop oracle for the LMPC lateral loop (T-LATERAL-LMPC-SPEED-MARGIN, 2026-10-11).

Production MPCSolver (params from config/av_stack_config.yaml) in a straight-line loop with the SAME
multisine excitation as the plant-ID runs (tools/sysid_injection.py), against the IDENTIFIED plant:
kinematic + understeer (L_eff = L(1 + K_us v^2)), first-order actuator tau 0.25 s with the MEASURED
command->angle gain (16.5/13.1/14.0/10.4 deg per unit at 7.6/9.7/11.6/13.5 m/s), 2-frame pipeline
delay, Smith predictor as in pid_controller, lookahead error input e = d + 6.3 sin(psi).
Reports the loop sensitivity |G_ur| at the tones (controller response to the disturbance) and the
plant gain |P_d|, for the production params and candidate variants, at several stack rate limits.

Calibration (2026-10-11, tools/analyze/sysid_steer_injection.py on recordings 20261011_1121-1134):
measured |P_d| mid-band 0.60 / 0.86 / 0.87 m per unit at 9.7 / 11.6 / 13.5 -> oracle 0.66 / 0.89 / 0.80;
measured sensitivity peaks 5-9x / 4.5-16x / 6-7x / 67x at 7.6 / 9.7 / 11.6 / 13.5 -> oracle (loose rate
limit) 4.8 / 6.7 / 5.8 / 3.8. The 13.5 cliff is the stack's error-scheduled per-frame rate limiter
(binding 30-38 % of frames there) plus r_steer_rate scheduling (2.0 -> 3.7 above 12 m/s), not modelled
as a schedule here. Use for RANKING fixes; certify in Unity on the mixed_radius reproducer.

Usage: python3 tools/analyze/lmpc_closed_loop_oracle.py
"""
import sys, math, numpy as np, dataclasses
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from av_stack.config import load_config
from control.mpc_controller import MPCParams, MPCSolver
from tools.sysid_injection import InjectionConfig
cfg=load_config(None); base=MPCParams.from_config(cfg)
FRAME_DT=0.05; inj=InjectionConfig(enabled=True, amplitude=0.25, f_min_hz=0.20, f_max_hz=1.0, n_tones=8, start_s=5.0, duration_s=50.0)
def max_steer_model(v): return float(np.interp(v,[0,12],[0.5236,0.2793]))
def A_meas(v): return math.radians(float(np.interp(v,[7.6,9.7,11.6,13.5],[16.5,13.1,14.0,10.4])))   # measured cmd->angle gain (rad/unit)
def run(params, v, tau=0.25, delay_frames=2, smith_cap=0.08, Ld=6.3, kappa=0.0, T=60.0, actuator_gain=None, rate_lim=0.15, seed=0):
    solver=MPCSolver(params); N=solver._N; L_plant=2.5*(1+0.0046*v*v)
    gain = actuator_gain(v) if actuator_gain else A_meas(v)
    e=0.0; psi=0.0; delta=0.0; last=0.0; prev=0.0; buf=[0.0]*delay_frames; us=[]; rs=[]; ds=[]
    rng=np.random.default_rng(seed); t=0.0
    for k in range(int(T/FRAME_DT)):
        t=k*FRAME_DT; r=inj.value(t, v)
        e_look=e+Ld*math.sin(psi); gainsp=1.0 if e*psi>=0 else 0.3; pred=e_look+gainsp*v*psi*min(delay_frames*FRAME_DT,smith_cap)
        res=solver.solve(e_lat=pred, e_heading=psi, v=v, last_delta_norm=last, kappa_ref_horizon=np.full(N,kappa), v_target=12.0, v_max=15.0, dt=params.dt, delta_actual_norm=delta/max_steer_model(v), lookahead_distance=0.0)
        raw=float(np.clip(res['steering_normalized'],-0.7,0.7)); u=float(np.clip(raw, prev-rate_lim, prev+rate_lim)); prev=u; last=u
        buf.append(u+r); tot=buf.pop(0)
        delta+= (FRAME_DT/tau)*(gain*tot-delta)
        psi+= (v*math.tan(delta)/L_plant - v*kappa)*FRAME_DT; e+= v*math.sin(psi)*FRAME_DT + 0.003*rng.standard_normal()
        us.append(u); rs.append(r); ds.append(e)
        if abs(e)>3: break
    us=np.array(us); rs=np.array(rs); ds=np.array(ds); m=rs!=0
    if m.sum()<200: return None
    sl=slice(int(np.flatnonzero(m)[0]), int(np.flatnonzero(m)[-1])+1); tones=inj.tones_hz(); tt=np.arange(sl.stop-sl.start)*FRAME_DT; w=np.hanning(len(tt))
    def X(x): x=(x-x.mean())*w; return np.array([np.sum(x*np.exp(-2j*math.pi*f*tt)) for f in tones])
    R=X(rs[sl]); U=X(us[sl]); D=X(ds[sl]); G=np.abs(U*np.conj(R))/np.abs(R*np.conj(R)); Pd=np.abs(D*np.conj(R))/np.abs(U*np.conj(R)+R*np.conj(R))
    return dict(G=G, Pd=Pd, tones=tones, d_rms=float(np.sqrt(np.mean(ds[sl]**2))), diverged=len(ds)<int(T/FRAME_DT))
def variant(**kw): return dataclasses.replace(base, **kw)
V={'production': base,
   'actuator model tau=0.25': variant(actuator_model_enabled=True, actuator_tau_s=0.25),
   'actuator model tau=0.25 + steer gain 0.8': variant(actuator_model_enabled=True, actuator_tau_s=0.25, max_steer_low_speed_rad=0.5236*0.8, max_steer_high_speed_rad=0.2793*0.8),
   'steer gain 0.8 only (model authority = measured)': variant(max_steer_low_speed_rad=0.5236*0.8, max_steer_high_speed_rad=0.2793*0.8),
   'r_steer_rate x2': variant(r_steer_rate=base.r_steer_rate*2),
   'r_steer_rate sched onset 8': variant(r_steer_rate_scheduling_enabled=True, r_steer_rate_speed_onset=8.0),
}
import itertools
RATE=[0.15, 0.06, 0.044]
print("measured sensitivity peaks (|G_ur| max @0.4-0.5 Hz): 7.6: 5-9x  9.7: 4.5-16x  11.6: 6-7x  13.5: 67x | measured |P_d| mid-band: 9.7 0.60, 11.6 0.86, 13.5 0.87 m/unit")
for rl in RATE:
  print(f"\n### stack per-frame rate limit {rl}")
  for name,p in list(V.items())[:3]:
    row=[]
    for v in (7.6, 9.7, 11.6, 13.5):
        o=run(p,v,rate_lim=rl)
        if o is None: row.append(f"v{v:4.1f}: DIVERGED"); continue
        k=int(np.argmax(o['G'])); mb=(o['tones']>=0.33)&(o['tones']<=0.58)
        row.append(f"v{v:4.1f}: sens {o['G'][k]:5.1f}x@{o['tones'][k]:.2f} |P_d|mid {np.median(o['Pd'][mb]):.2f} d_rms {o['d_rms']:.2f}{' DIV' if o['diverged'] else ''}")
    print(f"    {name:40s} | "+' | '.join(row))
