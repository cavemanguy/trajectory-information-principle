# E1 Persistent Calibration Final Result

**Primary classification:** P2 — compact persistent adapter, no specific retention edge

- Gate A calibration: **True**
- Gate N neutral noninferiority: **True**
- Gate Rfull: **False**
- Gate Rmatch(F2): **False**
- Efficiency gate: **True**

## Primary statistics

- F1 calibration gain: mean +0.1127; median +0.1144; 95% CI [+0.1055,+0.1194]
- Neutral F1-FULL at +120: mean +0.0063; median +0.0063; 95% CI [+0.0039,+0.0086]
- Retained-gain F1-FULL: mean +0.0016; median +0.0023; 95% CI [-0.0023,+0.0055]
- Retained-gain F1-F2: mean -0.0002; median +0.0002; 95% CI [-0.0021,+0.0015]

## Trainable parameter counts

- FULL: 1412
- F1: 544
- F2: 528
- F1/FULL: 0.3853

## Online update cost

- FULL: optimizer steps 2400; trainable-parameter element updates 3388800
- F1: optimizer steps 2400; trainable-parameter element updates 1305600
- F2: optimizer steps 2400; trainable-parameter element updates 1267200
- HEAD: optimizer steps 2400; trainable-parameter element updates 163200
- NOADAPT: optimizer steps 0; trainable-parameter element updates 0

## Condition summaries

- FULL: post-cal acc mean 0.9234; cal gain mean +0.1200; median +0.1217; 95% CI [+0.1126,+0.1269]; retained gain mean -0.0013; median -0.0002; 95% CI [-0.0047,+0.0017]; retention fraction mean -0.0168; neutral acc +120 mean 0.9257
- F1: post-cal acc mean 0.9162; cal gain mean +0.1127; median +0.1144; 95% CI [+0.1056,+0.1194]; retained gain mean +0.0002; median +0.0011; 95% CI [-0.0018,+0.0020]; retention fraction mean -0.0051; neutral acc +120 mean 0.9320
- F2: post-cal acc mean 0.9082; cal gain mean +0.1047; median +0.1065; 95% CI [+0.0986,+0.1108]; retained gain mean +0.0005; median +0.0004; 95% CI [-0.0023,+0.0032]; retention fraction mean -0.0031; neutral acc +120 mean 0.9323
- HEAD: post-cal acc mean 0.8418; cal gain mean +0.0384; median +0.0395; 95% CI [+0.0318,+0.0437]; retained gain mean -0.0009; median -0.0000; 95% CI [-0.0023,+0.0003]; retention fraction mean -0.0262; neutral acc +120 mean 0.9333
- NOADAPT: post-cal acc mean 0.8035; cal gain mean +0.0000; median +0.0000; 95% CI [+0.0000,+0.0000]; retained gain mean +0.0000; median +0.0000; 95% CI [+0.0000,+0.0000]; retention fraction mean n/a; neutral acc +120 mean 0.9322

## Switch challenge mean accuracy

- FULL: +1 0.8433; +3 0.8971; +5 0.9088; +10 0.9177
- F1: +1 0.8424; +3 0.8843; +5 0.8985; +10 0.9097
- F2: +1 0.8371; +3 0.8787; +5 0.8918; +10 0.9033
- HEAD: +1 0.8101; +3 0.8289; +5 0.8399; +10 0.8487
- NOADAPT: +1 0.8035; +3 0.8035; +5 0.8035; +10 0.8035

## Explicit checkpoint oracle

Restoring the exact post-calibration snapshot yields the post-calibration accuracy reported above; this is an explicit-storage ceiling, not a no-mode baseline.

## Per-family primary contrasts

- seed 1109: F1 cal=+0.0999, D_neutral=+0.0094, D_full=-0.0041, D_F2=-0.0012
- seed 1127: F1 cal=+0.1261, D_neutral=+0.0039, D_full=+0.0039, D_F2=-0.0027
- seed 1144: F1 cal=+0.1105, D_neutral=+0.0112, D_full=+0.0020, D_F2=+0.0015
- seed 1162: F1 cal=+0.1200, D_neutral=+0.0126, D_full=+0.0172, D_F2=+0.0020
- seed 1181: F1 cal=+0.1125, D_neutral=+0.0112, D_full=-0.0052, D_F2=+0.0024
- seed 1199: F1 cal=+0.1163, D_neutral=+0.0022, D_full=-0.0103, D_F2=-0.0015
- seed 1218: F1 cal=+0.0859, D_neutral=+0.0064, D_full=+0.0098, D_F2=-0.0016
- seed 1237: F1 cal=+0.1200, D_neutral=+0.0061, D_full=+0.0031, D_F2=-0.0046
- seed 1255: F1 cal=+0.1242, D_neutral=+0.0065, D_full=+0.0039, D_F2=+0.0027
- seed 1274: F1 cal=+0.0982, D_neutral=+0.0047, D_full=-0.0012, D_F2=+0.0025
- seed 1292: F1 cal=+0.1295, D_neutral=+0.0032, D_full=+0.0026, D_F2=+0.0043
- seed 1311: F1 cal=+0.1091, D_neutral=-0.0019, D_full=-0.0029, D_F2=-0.0067

## Interpretation boundary

E1 is an engineering benchmark. A positive result supports only a retention/efficiency tradeoff on this hidden synthetic sensor-calibration task. It does not establish real-sensor generalization, runtime hidden-state memory, or a stronger trajectory-information claim.
