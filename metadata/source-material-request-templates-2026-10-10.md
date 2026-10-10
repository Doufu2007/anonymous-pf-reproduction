# Source-material requests for strict reproduction

The current source-strict runners intentionally stop until the following information is confirmed. The message can be sent to the corresponding authors of Zhang et al. (2025), Wei and Guo (2021), and Wan et al. (2021), with the method-specific paragraph kept below.

## Common request

Dear authors,

I am preparing an auditable methodological review of quantum-inspired particle filtering. To reproduce the numerical results without introducing undocumented implementation choices, could you please share the source code or the following exact materials: all parameter tables, initialization rules, random seeds or trajectory files, software/version information, and the scripts used to generate the reported figures and tables?

The files would be used only for scientific verification and cited with the original paper.

## Zhang et al. (QM-HPF)

Could you clarify the particle encoding range, encoding length, continuous-state decoding rule, whether quantum amplitudes persist across generations, the low-weight-particle iteration and random-number order, whether all weights are recomputed after acceptance, and whether final systematic resampling is applied? For the battery experiment, could you provide the OCV curve, RC parameters, initial state, current/voltage/SOC data, and process/measurement-noise generation code?

## Wei and Guo (QGPF)

Could you clarify what the reported ratio 0.7 denotes, whether the entries of the 4-by-2 noise matrix are variances or standard deviations, the observation-noise variance, the exact replacement/retention rule, whether weights are recomputed after quantum genetic optimization, and whether the 99 simulations reuse one trajectory or regenerate trajectories? The MATLAB code or raw arrays for Table 1 would resolve these points.

## Wan et al. (EC-QPF)

Could you provide the complete Table II parameters, the velocity/heading/step-size and IMU-noise generators, the initial particle distribution, the relation between confidence parameter beta and ellipse scale s, the scalar definition of V in Eq. (22), the empty-subset and tie-breaking rules, the duplication allocation, random seeds or trajectories, and the CMCR-PF/SR-PF implementations? Eq. (22) appears dimensionally open if Dis is Euclidean distance and V is variance; could you confirm the intended convention?
