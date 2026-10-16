# Post-hoc failure tables for experiment 0001

AI-assisted analysis of saved predictions only. No new experiment, new method, threshold refit, or efficacy claim.

Coverage uses every frame, including missing predictions; selected failure rate is undefined when nothing is accepted.
Appearance frames 8–9 and 10–31 are post-hoc memory stages. Other boundaries follow protocol interventions.
The JSON includes descriptive paired sequence-bootstrap intervals; they are unadjusted for multiple strata and conditional on frozen calibration.

| Condition / phase | Method | Frames | Missing | Accepted | Accepted failures | Coverage | Selected failure | Zero residual / valid | Equal scores / valid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean / all (1–31) | static | 1240 | 0 | 1080 | 0 | 0.8710 | 0.0000 | 52/1240 | 52/1240 |
| clean / all (1–31) | temporal | 1240 | 0 | 1099 | 0 | 0.8863 | 0.0000 | 52/1240 | 52/1240 |
| clean / all (1–31) | random | 1240 | 0 | 984 | 0 | 0.7935 | 0.0000 | 52/1240 | 52/1240 |
| clean / all (1–31) | accept_all | 1240 | 0 | 1240 | 0 | 1.0000 | 0.0000 | 52/1240 | 52/1240 |
| clean / nominal (1–31) | static | 1240 | 0 | 1080 | 0 | 0.8710 | 0.0000 | 52/1240 | 52/1240 |
| clean / nominal (1–31) | temporal | 1240 | 0 | 1099 | 0 | 0.8863 | 0.0000 | 52/1240 | 52/1240 |
| clean / nominal (1–31) | random | 1240 | 0 | 984 | 0 | 0.7935 | 0.0000 | 52/1240 | 52/1240 |
| clean / nominal (1–31) | accept_all | 1240 | 0 | 1240 | 0 | 1.0000 | 0.0000 | 52/1240 | 52/1240 |
| occlusion / all (1–31) | static | 1240 | 160 | 522 | 0 | 0.4210 | 0.0000 | 57/1080 | 57/1080 |
| occlusion / all (1–31) | temporal | 1240 | 160 | 459 | 0 | 0.3702 | 0.0000 | 57/1080 | 57/1080 |
| occlusion / all (1–31) | random | 1240 | 160 | 858 | 315 | 0.6919 | 0.3671 | 57/1080 | 57/1080 |
| occlusion / all (1–31) | accept_all | 1240 | 160 | 1080 | 412 | 0.8710 | 0.3815 | 57/1080 | 57/1080 |
| occlusion / before (1–7) | static | 280 | 0 | 241 | 0 | 0.8607 | 0.0000 | 43/280 | 43/280 |
| occlusion / before (1–7) | temporal | 280 | 0 | 254 | 0 | 0.9071 | 0.0000 | 43/280 | 43/280 |
| occlusion / before (1–7) | random | 280 | 0 | 225 | 0 | 0.8036 | 0.0000 | 43/280 | 43/280 |
| occlusion / before (1–7) | accept_all | 280 | 0 | 280 | 0 | 1.0000 | 0.0000 | 43/280 | 43/280 |
| occlusion / partial_first (8–13) | static | 240 | 0 | 0 | 0 | 0.0000 | undefined | 5/240 | 5/240 |
| occlusion / partial_first (8–13) | temporal | 240 | 0 | 0 | 0 | 0.0000 | undefined | 5/240 | 5/240 |
| occlusion / partial_first (8–13) | random | 240 | 0 | 185 | 155 | 0.7708 | 0.8378 | 5/240 | 5/240 |
| occlusion / partial_first (8–13) | accept_all | 240 | 0 | 240 | 207 | 1.0000 | 0.8625 | 5/240 | 5/240 |
| occlusion / full (14–17) | static | 160 | 160 | 0 | 0 | 0.0000 | undefined | 0/0 | 0/0 |
| occlusion / full (14–17) | temporal | 160 | 160 | 0 | 0 | 0.0000 | undefined | 0/0 | 0/0 |
| occlusion / full (14–17) | random | 160 | 160 | 0 | 0 | 0.0000 | undefined | 0/0 | 0/0 |
| occlusion / full (14–17) | accept_all | 160 | 160 | 0 | 0 | 0.0000 | undefined | 0/0 | 0/0 |
| occlusion / partial_second (18–23) | static | 240 | 0 | 0 | 0 | 0.0000 | undefined | 6/240 | 6/240 |
| occlusion / partial_second (18–23) | temporal | 240 | 0 | 0 | 0 | 0.0000 | undefined | 6/240 | 6/240 |
| occlusion / partial_second (18–23) | random | 240 | 0 | 186 | 160 | 0.7750 | 0.8602 | 6/240 | 6/240 |
| occlusion / partial_second (18–23) | accept_all | 240 | 0 | 240 | 205 | 1.0000 | 0.8542 | 6/240 | 6/240 |
| occlusion / recovery (24–31) | static | 320 | 0 | 281 | 0 | 0.8781 | 0.0000 | 3/320 | 3/320 |
| occlusion / recovery (24–31) | temporal | 320 | 0 | 205 | 0 | 0.6406 | 0.0000 | 3/320 | 3/320 |
| occlusion / recovery (24–31) | random | 320 | 0 | 262 | 0 | 0.8187 | 0.0000 | 3/320 | 3/320 |
| occlusion / recovery (24–31) | accept_all | 320 | 0 | 320 | 0 | 1.0000 | 0.0000 | 3/320 | 3/320 |
| appearance / all (1–31) | static | 1240 | 0 | 985 | 744 | 0.7944 | 0.7553 | 923/1240 | 923/1240 |
| appearance / all (1–31) | temporal | 1240 | 0 | 1134 | 880 | 0.9145 | 0.7760 | 923/1240 | 923/1240 |
| appearance / all (1–31) | random | 1240 | 0 | 984 | 759 | 0.7935 | 0.7713 | 923/1240 | 923/1240 |
| appearance / all (1–31) | accept_all | 1240 | 0 | 1240 | 960 | 1.0000 | 0.7742 | 923/1240 | 923/1240 |
| appearance / before (1–7) | static | 280 | 0 | 241 | 0 | 0.8607 | 0.0000 | 43/280 | 43/280 |
| appearance / before (1–7) | temporal | 280 | 0 | 254 | 0 | 0.9071 | 0.0000 | 43/280 | 43/280 |
| appearance / before (1–7) | random | 280 | 0 | 225 | 0 | 0.8036 | 0.0000 | 43/280 | 43/280 |
| appearance / before (1–7) | accept_all | 280 | 0 | 280 | 0 | 1.0000 | 0.0000 | 43/280 | 43/280 |
| appearance / switch_transient (8–9) | static | 80 | 0 | 62 | 62 | 0.7750 | 1.0000 | 0/80 | 0/80 |
| appearance / switch_transient (8–9) | temporal | 80 | 0 | 0 | 0 | 0.0000 | undefined | 0/80 | 0/80 |
| appearance / switch_transient (8–9) | random | 80 | 0 | 60 | 60 | 0.7500 | 1.0000 | 0/80 | 0/80 |
| appearance / switch_transient (8–9) | accept_all | 80 | 0 | 80 | 80 | 1.0000 | 1.0000 | 0/80 | 0/80 |
| appearance / stable_wrong_identity (10–31) | static | 880 | 0 | 682 | 682 | 0.7750 | 1.0000 | 880/880 | 880/880 |
| appearance / stable_wrong_identity (10–31) | temporal | 880 | 0 | 880 | 880 | 1.0000 | 1.0000 | 880/880 | 880/880 |
| appearance / stable_wrong_identity (10–31) | random | 880 | 0 | 699 | 699 | 0.7943 | 1.0000 | 880/880 | 880/880 |
| appearance / stable_wrong_identity (10–31) | accept_all | 880 | 0 | 880 | 880 | 1.0000 | 1.0000 | 880/880 | 880/880 |
| abrupt_motion / all (1–31) | static | 1240 | 0 | 1074 | 0 | 0.8661 | 0.0000 | 50/1240 | 50/1240 |
| abrupt_motion / all (1–31) | temporal | 1240 | 0 | 1060 | 0 | 0.8548 | 0.0000 | 50/1240 | 50/1240 |
| abrupt_motion / all (1–31) | random | 1240 | 0 | 984 | 0 | 0.7935 | 0.0000 | 50/1240 | 50/1240 |
| abrupt_motion / all (1–31) | accept_all | 1240 | 0 | 1240 | 0 | 1.0000 | 0.0000 | 50/1240 | 50/1240 |
| abrupt_motion / before (1–15) | static | 600 | 0 | 522 | 0 | 0.8700 | 0.0000 | 47/600 | 47/600 |
| abrupt_motion / before (1–15) | temporal | 600 | 0 | 548 | 0 | 0.9133 | 0.0000 | 47/600 | 47/600 |
| abrupt_motion / before (1–15) | random | 600 | 0 | 474 | 0 | 0.7900 | 0.0000 | 47/600 | 47/600 |
| abrupt_motion / before (1–15) | accept_all | 600 | 0 | 600 | 0 | 1.0000 | 0.0000 | 47/600 | 47/600 |
| abrupt_motion / turn (16–16) | static | 40 | 0 | 35 | 0 | 0.8750 | 0.0000 | 0/40 | 0/40 |
| abrupt_motion / turn (16–16) | temporal | 40 | 0 | 0 | 0 | 0.0000 | undefined | 0/40 | 0/40 |
| abrupt_motion / turn (16–16) | random | 40 | 0 | 32 | 0 | 0.8000 | 0.0000 | 0/40 | 0/40 |
| abrupt_motion / turn (16–16) | accept_all | 40 | 0 | 40 | 0 | 1.0000 | 0.0000 | 0/40 | 0/40 |
| abrupt_motion / after (17–31) | static | 600 | 0 | 517 | 0 | 0.8617 | 0.0000 | 3/600 | 3/600 |
| abrupt_motion / after (17–31) | temporal | 600 | 0 | 512 | 0 | 0.8533 | 0.0000 | 3/600 | 3/600 |
| abrupt_motion / after (17–31) | random | 600 | 0 | 478 | 0 | 0.7967 | 0.0000 | 3/600 | 3/600 |
| abrupt_motion / after (17–31) | accept_all | 600 | 0 | 600 | 0 | 1.0000 | 0.0000 | 3/600 | 3/600 |
