"""Isolated QGPF-like numerical revision of the frozen historical contract.

This is not a recovery of the published filtering distribution. Probability
floors are removed; source-retention is a separate factor. Post endpoints are
read-only diagnostics. The evaluator never supplies truth to this module.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
import numpy as np

@dataclass(frozen=True)
class ModelConfig:
    horizon: int = 100
    initial_truth: tuple = (70.0,5.0,23.0,8.0)
    process_std: tuple = (1.5,1.5,.9,.03)
    observation_std: float = 1.0

@dataclass(frozen=True)
class FilterConfig:
    particle_count: int = 200
    initial_mean: tuple = (71.0,8.66,18.9,7.8)
    initial_std: tuple = (8.0,1.0,5.0,1.0)
    state_min: tuple = (-1000.,-100.,-1000.,-100.)
    state_max: tuple = (1000.,100.,1000.,100.)
    bit_length: int = 10
    generation_count: int = 4
    elite_fraction: float = .20
    rotation_angle_max_pi: float = .08
    rotation_angle_min_pi: float = .01

F=np.array([[1.,1.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,1.],[0.,0.,0.,1.]])
METHODS=('B-stable','Q-stable','Q-stable-retain')
COUNTER_NAMES=(
    'likelihood_evals','candidate_evals','proposal_chromosome_events',
    'proposal_candidate_attempts','rotation_calls','crossover_operations',
    'measurement_bit_observations','probability_entry_updates','fitness_evals',
    'accepts','rejects','candidate_zero_density_diagnostics',
    'old_zero_density_diagnostics','would_zero_tie_probability_domain',
    'stable_vs_probability_decision_disagreements','base_zero_density_diagnostics',
    'base_all_zero_density_steps',
)
COUNT_DEFINITIONS={
    'likelihood_evals':'One scalar state log-likelihood evaluation. Counts base, initial old, candidate and final updated scores.',
    'candidate_evals':'One candidate score; a subset of likelihood_evals.',
    'fitness_evals':'Same candidate score as candidate_evals, overlapping label; no added work.',
    'proposal_chromosome_events':'One nonelite chromosome selected for its complete G-generation update; historical proposal_attempts unit.',
    'proposal_candidate_attempts':'One generation candidate, same event as candidate_evals.',
    'rotation_calls':'One conditional scalar probability-rotation invocation; not a hardware gate.',
    'measurement_bit_observations':'One scalar Bernoulli bit observation; two per coordinate-bit per candidate.',
    'probability_entry_updates':'Same conditional event as rotation_calls; overlapping work.',
    'crossover_operations':'Explicit crossover absent in this QGPF-like contract; exact zero.',
    'would_zero_tie_probability_domain':'On this stable trajectory, both old/new exp(logpdf) are zero; hypothetical probability-domain tie, not a tie in stable acceptance.',
    'stable_vs_probability_decision_disagreements':'Stable ratio vs old floor-rule evaluated for the same old/new logpdf and same uniform on stable trajectory; NOT differences from historical trajectory.',
    'base_all_zero_density_steps':'All base exp(logpdf) would be zero; stable relative log weights remain defined.',
    'addition_rule':'Candidate/fitness/proposal labels overlap likelihood scoring; rotations overlap probability updates. Never add these labels into a total operation cost.',
    'unavailable':'Physical quantum gate counts/hardware measurements and fully exclusive operation-time attribution: unavailable; classical CPU implementation only.',
}

def transition(states):return np.asarray(states)@F.T
def observation(states):
    p=np.asarray(states,dtype=np.float64)
    return np.sqrt(p[:,0]**2+p[:,2]**2)

def log_likelihood(particles,y,model,counter=None):
    p=np.asarray(particles,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=4 or not np.isfinite(p).all() or not math.isfinite(float(y)) or not math.isfinite(model.observation_std) or model.observation_std<=0:
        raise FloatingPointError('finite 4D states, observation and positive standard deviation required')
    if counter is not None:counter['likelihood_evals']+=len(p)
    residual=float(y)-observation(p)
    with np.errstate(over='raise',invalid='raise'):
        result=-.5*residual**2/(model.observation_std**2)-math.log(model.observation_std*math.sqrt(2.*math.pi))
    if not np.isfinite(result).all():raise FloatingPointError('nonfinite log-likelihood')
    return result

def normalize_logweights(logw):
    a=np.asarray(logw,dtype=np.float64)
    if a.ndim!=1 or len(a)==0 or np.isnan(a).any() or np.isposinf(a).any() or not np.isfinite(a).any():
        raise FloatingPointError('invalid log weights; all unsupported states are a failure')
    w=np.exp(a-np.max(a));total=float(w.sum())
    if not math.isfinite(total) or total<=0:raise FloatingPointError('invalid weight normalizer')
    return w/total

def accept_log_candidate(old,new,uniform,mode='ratio'):
    old,new=float(old),float(new)
    if math.isnan(old) or math.isnan(new) or old==math.inf or new==math.inf or (old==-math.inf and new==-math.inf):
        raise FloatingPointError('undefined acceptance density comparison')
    if mode=='retain':return new>old
    if mode!='ratio':raise ValueError('unknown acceptance convention')
    if new>=old:return True
    if new==-math.inf:return False
    if not 0.<=uniform<1.:raise ValueError('uniform must lie in [0,1)')
    return (-math.inf if uniform==0. else math.log(uniform))<new-old

def systematic_resample(weights,rng):
    n=len(weights);points=(rng.random()+np.arange(n))/n
    cumulative=np.cumsum(weights);cumulative[-1]=1.
    return np.searchsorted(cumulative,points,side='right')

def encode_state(state,cfg):
    lo=np.asarray(cfg.state_min);hi=np.asarray(cfg.state_max);levels=(1<<cfg.bit_length)-1
    ints=np.rint((np.clip(state,lo,hi)-lo)/(hi-lo)*levels).astype(np.int64)
    bits=np.zeros((len(state),cfg.bit_length),dtype=np.int8)
    for bit in range(cfg.bit_length):bits[:,bit]=(ints>>bit)&1
    return bits

def decode_state(bits,cfg):
    lo=np.asarray(cfg.state_min);hi=np.asarray(cfg.state_max);levels=(1<<cfg.bit_length)-1
    ints=bits.astype(np.int64)@(1<<np.arange(cfg.bit_length,dtype=np.int64))
    return lo+(ints/levels)*(hi-lo)

def rotate_probability(prob_one,target_bit,angle):
    """Scalar clipping is bitwise checked against immutable historical helper."""
    p=min(1.,max(0.,float(prob_one)));c,s=math.cos(angle),math.sin(angle)
    a0,a1=math.sqrt(max(0.,1.-p)),math.sqrt(p)
    if target_bit==1:a0,a1=c*a0-s*a1,s*a0+c*a1
    else:a0,a1=c*a0+s*a1,-s*a0+c*a1
    return float(min(1.,max(0.,a1*a1)))

def qgpf_update(particles,weights,y,rng,model,cfg,counter,retention):
    n,dim=particles.shape;elite_n=max(1,min(n-1,int(round(cfg.elite_fraction*n))))
    order=np.argsort(-weights,kind='stable');elite_idx=order[:elite_n]
    elites=particles[elite_idx].copy();offspring=np.empty((n-elite_n,dim))
    elite_weights=weights[elite_idx];parent_weights=elite_weights/np.sum(elite_weights)
    for out_i in range(n-elite_n):
        parent_i=int(rng.choice(elite_n,p=parent_weights));target=elites[parent_i]
        current=particles[order[elite_n+(out_i%(n-elite_n))]]
        # Preserve the old deterministic encoding call even though unused.
        current_bits=encode_state(current,cfg)
        target_bits=encode_state(target,cfg);probs=np.full((dim,cfg.bit_length),.5)
        best=current.copy();counter['proposal_chromosome_events']+=1
        best_log=float(log_likelihood(best[None,:],y,model,counter)[0])
        for generation in range(cfg.generation_count):
            frac=generation/max(1,cfg.generation_count-1)
            angle=math.pi*(cfg.rotation_angle_max_pi*(1.-frac)+cfg.rotation_angle_min_pi*frac)
            bits=np.empty((dim,cfg.bit_length),dtype=np.int8);rotations=0
            for d in range(dim):
                for b in range(cfg.bit_length):
                    measured=int(rng.random()<probs[d,b]);target_bit=int(target_bits[d,b])
                    if measured!=target_bit:
                        probs[d,b]=rotate_probability(probs[d,b],target_bit,angle);rotations+=1
                    bits[d,b]=int(rng.random()<probs[d,b])
            counter['rotation_calls']+=rotations;counter['probability_entry_updates']+=rotations
            counter['measurement_bit_observations']+=2*dim*cfg.bit_length
            candidate=decode_state(bits,cfg)
            counter['candidate_evals']+=1;counter['fitness_evals']+=1;counter['proposal_candidate_attempts']+=1
            candidate_log=float(log_likelihood(candidate[None,:],y,model,counter)[0])
            old_prob=math.exp(best_log);new_prob=math.exp(candidate_log)
            counter['candidate_zero_density_diagnostics']+=int(new_prob==0.)
            counter['old_zero_density_diagnostics']+=int(old_prob==0.)
            counter['would_zero_tie_probability_domain']+=int(old_prob==0. and new_prob==0.)
            if retention:
                accepted=accept_log_candidate(best_log,candidate_log,.5,'retain')
            else:
                # As in the historical branch, improved candidates consume no
                # acceptance RNG draw. Lower candidates consume exactly one.
                uniform=.5 if candidate_log>=best_log else float(rng.random())
                accepted=accept_log_candidate(best_log,candidate_log,uniform,'ratio')
                old_rule=new_prob>=old_prob or uniform<min(1.,new_prob/max(old_prob,1e-300))
                counter['stable_vs_probability_decision_disagreements']+=int(accepted!=old_rule)
            if accepted:counter['accepts']+=1;best,best_log=candidate,candidate_log
            else:counter['rejects']+=1
        offspring[out_i]=best
    updated=np.vstack([elites,offspring])
    updated_weights=normalize_logweights(log_likelihood(updated,y,model,counter))
    return updated,updated_weights

def run_filter(observations,rng,method,model=ModelConfig(),cfg=FilterConfig(),record_post=True):
    if method not in METHODS:raise ValueError('unknown revision method')
    obs=np.asarray(observations,dtype=np.float64)
    if obs.shape!=(model.horizon+1,) or not np.isfinite(obs).all():raise ValueError('observation horizon/finite contract mismatch')
    start=time.perf_counter()
    particles=rng.normal(np.asarray(cfg.initial_mean),np.asarray(cfg.initial_std),size=(cfg.particle_count,4))
    initial_particles=particles.copy()
    pre=np.zeros((model.horizon+1,4));post=np.zeros_like(pre)
    weights=np.full(cfg.particle_count,1./cfg.particle_count)
    # Preserve historical zero estimate at k=0; endpoint excludes k=0.
    particles_pre=np.empty((model.horizon,cfg.particle_count,4));particles_post=np.empty_like(particles_pre)
    weights_pre=np.empty((model.horizon,cfg.particle_count));weights_post=np.empty_like(weights_pre)
    counter={name:0 for name in COUNTER_NAMES};step_counters=[];step_ess=[];step_distinct=[];step_elapsed=[]
    init_end=time.perf_counter()
    process_std=np.asarray(model.process_std)
    for k in range(1,model.horizon+1):
        step_start=time.perf_counter();before=dict(counter)
        particles=transition(particles)+rng.normal(0.,process_std,size=particles.shape)
        base_log=log_likelihood(particles,obs[k],model,counter)
        zero_count=int(np.count_nonzero(np.exp(base_log)==0.))
        counter['base_zero_density_diagnostics']+=zero_count
        counter['base_all_zero_density_steps']+=int(zero_count==len(particles))
        weights=normalize_logweights(base_log)
        pre[k]=np.sum(weights[:,None]*particles,axis=0)
        particles_pre[k-1]=particles;weights_pre[k-1]=weights
        step_ess.append(float(1./np.sum(weights**2)))
        if method=='B-stable':
            particles=particles[systematic_resample(weights,rng)];weights=np.full(cfg.particle_count,1./cfg.particle_count)
        else:
            particles,weights=qgpf_update(particles,weights,obs[k],rng,model,cfg,counter,method=='Q-stable-retain')
        if record_post:post[k]=np.sum(weights[:,None]*particles,axis=0)
        particles_post[k-1]=particles;weights_post[k-1]=weights
        step_distinct.append(int(np.unique(np.round(particles[:,[0,2]],8),axis=0).shape[0]))
        if not np.isfinite(particles).all() or not np.isfinite(weights).all() or not np.isclose(weights.sum(),1.,rtol=0,atol=1e-12):
            raise FloatingPointError('invalid filtering state/weights')
        step_counters.append([counter[name]-before[name] for name in COUNTER_NAMES])
        step_elapsed.append(time.perf_counter()-step_start)
    runtime=time.perf_counter()-start
    guardrails={'finite_estimates':bool(np.isfinite(pre).all() and np.isfinite(post).all()),
                'all_weights_normalized':bool(np.allclose(weights_pre.sum(axis=1),1.,rtol=0,atol=1e-12) and np.allclose(weights_post.sum(axis=1),1.,rtol=0,atol=1e-12)),
                'counter_sum_matches':bool(np.array_equal(np.sum(step_counters,axis=0),np.array([counter[x] for x in COUNTER_NAMES]))),
                'post_endpoint_readonly':True,'truth_unavailable_to_filter':True}
    if not all(guardrails.values()):raise AssertionError('filter guardrail failure')
    return {'estimate_pre':pre,'estimate_post':post,'initial_particles':initial_particles,
            'final_particles':particles.copy(),'final_weights':weights.copy(),
            'particles_pre':particles_pre,'particles_post':particles_post,'weights_pre':weights_pre,'weights_post':weights_post,
            'step_counters':np.asarray(step_counters,dtype=np.int64),'step_ess':np.asarray(step_ess),
            'step_distinct':np.asarray(step_distinct,dtype=np.int64),'step_elapsed_s':np.asarray(step_elapsed),
            'runtime_s':float(runtime),'initialization_runtime_s':float(init_end-start),
            'counters':counter,'guardrails':guardrails,'rng_final_state':rng.bit_generator.state}
