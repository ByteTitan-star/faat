# badnet_C_p0.03 — defense breakdown

## badnet_C_p0.03   (BadNets + Component C)


### ASR after defense (%)  (badnet_C_p0.03)


sel\def      strip    scan      ac     abl      fp     fst      nc     rnp   i-bau
----------------------------------------------------------------------------------
random           -       -    28.4     5.5    28.9    50.4     1.1    24.1     7.4
forget           -       -    65.7     2.6    37.8    70.3     1.0    88.2    26.1
res_log          -       -    67.9     1.6    51.2    86.4     1.1    30.2     3.5
res_linear       -       -    65.8     0.0    36.4    80.3     1.3    47.6     7.4
res_square       -       -    61.6     0.6    50.1    85.2    81.2     0.0    14.8
stealth          -       -    51.8     0.2    25.9    37.2    54.0     0.0     5.9

### Clean ACC after defense (%)  (badnet_C_p0.03)


sel\def      strip    scan      ac     abl      fp     fst      nc     rnp   i-bau
----------------------------------------------------------------------------------
random           -       -    90.0    43.8    92.3    92.8    91.6    43.1    89.8
forget           -       -    89.3    49.3    92.0    92.3    91.3    62.0    88.0
res_log          -       -    89.2    45.7    92.3    91.5    90.9    22.1    85.3
res_linear       -       -    89.4    53.5    92.1    91.8    90.8    60.3    88.6
res_square       -       -    89.2    46.1    92.3    91.7    93.0    18.0    85.7
stealth          -       -    89.6    57.8    92.8    92.4    93.4    53.2    90.1

### Detection TPR%/FPR%  (badnet_C_p0.03)


sel\def          strip        scan
----------------------------------
random            12/8        99/0
forget            44/8        99/0
res_log           48/7       100/0
res_linear        15/8       100/0
res_square        56/8       100/0
stealth           14/6       100/0


## Coverage (parsed / total logs)

- **strip**: 6 / 6 parsed
- **scan**: 6 / 6 parsed
- **ac**: 6 / 6 parsed
- **abl**: 6 / 6 parsed
- **fp**: 6 / 6 parsed
- **fst**: 6 / 6 parsed
- **nc**: 6 / 6 parsed
- **rnp**: 6 / 6 parsed
- **i-bau**: 6 / 6 parsed
