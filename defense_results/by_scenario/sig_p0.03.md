# sig_p0.03 — defense breakdown

## sig_p0.03   (Signal (Sig))


### ASR after defense (%)  (sig_p0.03)


sel\def      strip    scan      ac     abl      fp     fst      nc     rnp   i-bau
----------------------------------------------------------------------------------
random           -       -    93.5     0.4    61.6    51.2    94.2     0.0     8.9
forget           -       -    96.5     0.4    75.8    86.9    96.5    90.5    17.7
res_log          -       -    96.9     0.0    80.5    89.0    97.1     0.0     7.8
res_linear       -       -    97.5     0.0    86.9    81.5    97.2     0.0    42.4
res_square       -       -    96.0     0.0    88.4    70.8    98.0     0.0    12.3

### Clean ACC after defense (%)  (sig_p0.03)


sel\def      strip    scan      ac     abl      fp     fst      nc     rnp   i-bau
----------------------------------------------------------------------------------
random           -       -    89.4    40.6    91.6    92.4    93.6    63.4    88.3
forget           -       -    89.2    40.9    91.9    91.9    93.2    56.3    86.8
res_log          -       -    88.9    46.5    92.3    91.9    93.0    37.9    87.1
res_linear       -       -    89.2    40.3    92.5    92.3    93.3    33.0    87.9
res_square       -       -    89.3    49.3    92.1    91.9    93.0    48.4    88.1

### Detection TPR%/FPR%  (sig_p0.03)


sel\def          strip        scan
----------------------------------
random            82/7        98/0
forget            94/9        98/0
res_log           96/5        97/0
res_linear        96/6        99/0
res_square        98/9        99/0


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
