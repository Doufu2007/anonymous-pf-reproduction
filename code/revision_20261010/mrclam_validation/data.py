"""Causal MR.CLAM sensor intake; robot truth is an evaluator-only channel.

The Measurement.dat header says 'Subject #', but its values are barcode
numbers. Barcodes.dat must be used to map them to landmark subject identifiers.
All repeated timestamps are retained as one batch. Different timestamps are
never merged into a simultaneous observation.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np

INPUT_KEYS=('odometry_segments','event_times','dt','observations','landmark_xy',
            'known_map','state_min','state_max','start_time')

def wrap_angle(a):
    return (np.asarray(a)+np.pi)%(2*np.pi)-np.pi

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def barcode_to_subject(barcodes):
    a=np.asarray(barcodes)
    if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all():
        raise ValueError('Invalid barcode table')
    if len(np.unique(a[:,1]))!=len(a) or len(np.unique(a[:,0]))!=len(a):
        raise ValueError('Barcode/subject IDs must be unique')
    return {int(code):int(subject) for subject,code in a}

def control_segments(odometry,start,end):
    a=np.asarray(odometry,dtype=np.float64)
    if not start<=end or start<a[0,0] or end>a[-1,0]:
        raise ValueError('Control interval outside measured support')
    if end==start:
        return np.empty((0,3),dtype=np.float64)
    index=int(np.searchsorted(a[:,0],start,side='right')-1)
    breaks=a[(a[:,0]>start)&(a[:,0]<end)]
    times=np.r_[start,breaks[:,0],end]
    values=np.vstack([a[index,1:3],breaks[:,1:3]])
    raw=np.column_stack([np.diff(times),values])
    compact=[]
    for duration,v,w in raw:
        if duration<=0:
            continue
        if compact and compact[-1][1]==v and compact[-1][2]==w:
            compact[-1][0]+=duration
        else:
            compact.append([float(duration),float(v),float(w)])
    result=np.asarray(compact,dtype=np.float64).reshape(-1,3)
    if not np.isclose(result[:,0].sum(),end-start,rtol=0,atol=1e-9):
        raise AssertionError('Lost control duration')
    return result

def select_batches(measurements,start,end,min_interval):
    a=np.asarray(measurements,dtype=np.float64)
    if min_interval<=0:
        raise ValueError('Positive observation subsampling interval required')
    last=start
    selected=[]
    for t in np.unique(a[:,0]):
        if t<start+min_interval or t>end or t<last+min_interval:
            continue
        selected.append((float(t),a[a[:,0]==t].copy()))
        last=float(t)
    return selected

def interpolate_truth(groundtruth,times):
    a=np.asarray(groundtruth,dtype=np.float64)
    t=np.asarray(times,dtype=np.float64)
    valid=(t>=a[0,0])&(t<=a[-1,0])
    result=np.full((len(t),3),np.nan)
    for col in [1,2]:
        result[valid,col-1]=np.interp(t[valid],a[:,0],a[:,col])
    result[valid,2]=wrap_angle(np.interp(t[valid],a[:,0],np.unwrap(a[:,3])))
    return result,valid

def filter_inputs(sequence):
    """Whitelist: ground truth, scoring validity and IDs never enter filter."""
    return {key:sequence[key] for key in INPUT_KEYS}

def load_sequence(raw_dir,dataset_id,robot_id=1,duration_s=1200.,min_interval_s=1.):
    root=Path(raw_dir)
    candidates=[root/f'MRCLAM_Dataset{dataset_id}',root/'extracted'/f'MRCLAM_Dataset{dataset_id}']
    folder=next((p for p in candidates if p.is_dir()),None)
    if folder is None:
        raise FileNotFoundError(f'MRCLAM_Dataset{dataset_id} not found under {root}')
    files={
        'barcodes':folder/'Barcodes.dat', 'known_map':folder/'Landmark_Groundtruth.dat',
        'odometry':folder/f'Robot{robot_id}_Odometry.dat',
        'measurements':folder/f'Robot{robot_id}_Measurement.dat',
        'truth':folder/f'Robot{robot_id}_Groundtruth.dat'}
    tables={k:np.atleast_2d(np.loadtxt(path,comments='#',dtype=np.float64)) for k,path in files.items()}
    raw_timestamp_inversions={}
    for key,a in tables.items():
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite raw '+key)
        if key in ('odometry','measurements','truth'):
            inversions=int(np.count_nonzero(np.diff(a[:,0])<0))
            raw_timestamp_inversions[key]=inversions
            if inversions:
                if key != 'odometry':
                    raise ValueError('Raw timestamps not nondecreasing: '+key)
                # MR.CLAM Dataset 9 contains one leading odometry inversion.
                # Stable timestamp sorting is a deterministic data-ingest
                # repair, recorded below; no truth or outcome is consulted.
                tables[key]=a[np.argsort(a[:,0],kind='stable')]
    barcodes=barcode_to_subject(tables['barcodes'])
    map_ids=[int(row[0]) for row in tables['known_map']]
    if len(set(map_ids)) != len(map_ids):
        raise ValueError('Duplicate landmark subject IDs in known map')
    landmarks={int(row[0]):row[1:3].copy() for row in tables['known_map']}
    odom=tables['odometry']
    measured=tables['measurements'].copy()
    raw_measurement_count=len(measured)
    unknown_mask=np.array([int(code) not in barcodes for code in measured[:,1]],dtype=bool)
    unknown=int(unknown_mask.sum())
    if unknown:
        # A row with no barcode-to-subject/map entry has no defined
        # range-bearing likelihood target. Drop it under this frozen ingest
        # rule and retain the count in metadata; never silently reinterpret it.
        measured=measured[~unknown_mask]
    measured[:,1]=[barcodes[int(code)] for code in measured[:,1]]
    landmark_obs=measured[np.isin(measured[:,1],list(landmarks))]
    if not len(landmark_obs) or np.any(landmark_obs[:,2]<0):
        raise ValueError('No valid real landmark observations')
    # Ground-truth values do not determine initialization or inputs. Its
    # timestamp support is used only to ensure every retained event can be
    # scored without a silent missingness mask.
    # The scoring channel must cover every retained event. Timestamps only
    # determine support; no robot pose values are read by the filter.
    start=max(float(odom[0,0]),float(landmark_obs[0,0]),float(tables['truth'][0,0]))
    end=min(start+duration_s,float(odom[-1,0]),float(landmark_obs[-1,0]),float(tables['truth'][-1,0]))
    selected=select_batches(landmark_obs,start,end,min_interval_s)
    if not selected:
        raise ValueError('No selected observation batches')
    times=np.array([row[0] for row in selected],dtype=np.float64)
    segments=[]
    previous=start
    for t in times:
        segments.append(control_segments(odom,previous,float(t)))
        previous=float(t)
    observations=[np.column_stack([batch[:,2],wrap_angle(batch[:,3])]) for _,batch in selected]
    coords=[np.vstack([landmarks[int(code)] for code in batch[:,1]]) for _,batch in selected]
    known_map=np.vstack([landmarks[k] for k in sorted(landmarks)])
    lo=np.r_[known_map.min(axis=0)-1.,-np.pi]
    hi=np.r_[known_map.max(axis=0)+1.,np.pi]
    truth,valid=interpolate_truth(tables['truth'],times)
    within=landmark_obs[(landmark_obs[:,0]>=start)&(landmark_obs[:,0]<=end)]
    hashes={key:{'name':path.name,'sha256':sha256(path)} for key,path in files.items()}
    metadata={
        'dataset_id':int(dataset_id),'robot_id':int(robot_id),'source_files':hashes,
        'raw_measurement_rows':int(raw_measurement_count),
        'unknown_barcode_rows_dropped':unknown,
        'raw_landmark_rows':int(len(landmark_obs)),
        'landmark_rows_in_window':int(len(within)),
        'raw_batches_in_window':int(len(np.unique(within[:,0]))),
        'retained_batches':len(selected),
        'retained_landmark_observations':sum(len(z) for z in observations),
        'discarded_window_landmark_rows':int(len(within))-sum(len(z) for z in observations),
        'requested_duration_s':float(duration_s),'available_window_s':end-start,
        'evaluation_coverage_batches':int(valid.sum()),
        'start_time':start,'end_time':end,
        'max_gap_s':float(np.diff(np.r_[start,times]).max()),
        'minimum_interval_s':float(min_interval_s),
        'barcode_mapping_applied':True,
        'raw_timestamp_inversions':raw_timestamp_inversions,
        'odometry_stably_sorted_if_needed':True,
        'raw_odometry_is_logged_velocity_commands':True,
        'truth_for_scoring_only':True,'truth_timestamp_support_only':True,'initialization_uses_map_only':True,
        'third_party_raw_redistribution':'not included in public repository',
    }
    return {'event_times':times,'odometry_segments':segments,
            'dt':np.diff(np.r_[start,times]),'landmark_xy':coords,
            'observations':observations,'known_map':known_map,
            'state_min':lo,'state_max':hi,'start_time':start,
            'truth':truth,'truth_valid':valid,'metadata':metadata}
