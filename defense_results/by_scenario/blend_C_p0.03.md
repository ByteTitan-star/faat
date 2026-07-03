# blend_C_p0.03 — defense breakdown

## blend_C_p0.03   (Blended + Component C)


### ASR after defense (%)  (blend_C_p0.03)


sel\def      strip    scan      ac     abl      fp     fst      nc     rnp   i-bau
----------------------------------------------------------------------------------
random           -       -    62.9     6.1    74.5    64.2    74.8     0.0    48.2
forget           -       -    84.8     1.9    95.6    80.0    91.6     0.1    76.5
res_log          -       -    89.5     1.8    98.5    89.7    96.0    55.5    35.8
res_linear       -       -    93.9     0.7    94.7    88.0    93.9     0.4    29.6
res_square       -       -    91.8     1.6    97.6    90.9    73.1    97.3    46.1
stealth          -       -    85.5     8.9    92.8    84.3    91.0    71.7    22.0

### Clean ACC after defense (%)  (blend_C_p0.03)


sel\def      strip    scan      ac     abl      fp     fst      nc     rnp   i-bau
----------------------------------------------------------------------------------
random           -       -    89.7    57.8    92.3    92.6    93.9    33.4    88.0
forget           -       -    89.9    53.6    92.7    91.9    93.2    27.0    84.8
res_log          -       -    89.0    43.3    92.6    92.1    93.3    51.4    85.7
res_linear       -       -    89.2    39.1    92.5    91.9    93.2     7.7    87.2
res_square       -       -    89.2    54.4    92.5    92.0    91.0    66.8    87.7
stealth          -       -    89.8    58.8    92.3    92.5    93.7    60.6    86.9

### Detection TPR%/FPR%  (blend_C_p0.03)


sel\def          strip        scan
----------------------------------
random             1/8        96/0
forget             5/8        99/0
res_log          14/10        98/0
res_linear        10/8        98/0
res_square        27/7        98/0
stealth            2/9        98/0


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
