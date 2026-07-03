# ctrl_p0.03 — defense breakdown

## ctrl_p0.03   (Ctrl (clean-label))


### ASR after defense (%)  (ctrl_p0.03)


sel\def      strip    scan      ac     abl      fp     fst      nc     rnp   i-bau
----------------------------------------------------------------------------------
random           -       -    84.2    66.5    94.9    93.6     1.0    26.3    39.5
forget           -       -    93.6    57.4    97.4    88.7    95.7    42.2    98.1
res_log          -       -    91.0    85.9    97.8    97.4     1.3    84.9    65.7
res_linear       -       -    89.6    84.7    99.2    98.7     1.1    35.9    45.9
res_square       -       -    89.9    73.0    98.7    97.2    94.8    39.2    40.3

### Clean ACC after defense (%)  (ctrl_p0.03)


sel\def      strip    scan      ac     abl      fp     fst      nc     rnp   i-bau
----------------------------------------------------------------------------------
random           -       -    89.6    92.4    93.1    92.4    91.4    31.7    91.2
forget           -       -    89.7    91.6    92.3    92.3    93.0    56.8    87.8
res_log          -       -    89.3    91.8    92.1    92.1    90.1    54.0    88.3
res_linear       -       -    89.5    91.3    92.3    91.9    90.5    60.2    88.0
res_square       -       -    89.0    91.1    92.1    91.8    93.4    24.6    88.5

### Detection TPR%/FPR%  (ctrl_p0.03)


sel\def          strip        scan
----------------------------------
random            52/8        95/0
forget            52/7        98/0
res_log          80/10        97/0
res_linear       76/11        97/0
res_square        44/9        96/0


## Coverage (parsed / total logs)

- **strip**: 5 / 5 parsed
- **scan**: 5 / 5 parsed
- **ac**: 5 / 5 parsed
- **abl**: 5 / 5 parsed
- **fp**: 5 / 5 parsed
- **fst**: 5 / 5 parsed
- **nc**: 5 / 5 parsed
- **rnp**: 5 / 5 parsed
- **i-bau**: 5 / 5 parsed
