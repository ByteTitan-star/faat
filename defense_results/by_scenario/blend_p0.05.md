# blend_p0.05 — defense breakdown

## blend_p0.05   (Blended (higher poison rate))


### ASR after defense (%)  (blend_p0.05)


sel\def      strip    scan      ac     abl      fp     fst      nc     rnp   i-bau
----------------------------------------------------------------------------------
random           -       .    47.2    40.4    40.2       .    52.7     0.0     9.5
forget           -       .    65.4    50.8    40.7       .    71.9     0.0     9.7
res_log          -       .    68.9    63.6    50.8       .    44.6     0.0     3.4
res_linear       -       .    67.0    51.4    47.5       .    54.0     0.0    10.2
res_square       -       .    63.9    61.0    41.4       .    49.2     0.0    11.5

### Clean ACC after defense (%)  (blend_p0.05)


sel\def      strip    scan      ac     abl      fp     fst      nc     rnp   i-bau
----------------------------------------------------------------------------------
random           -       .    89.6    91.7    92.4       .    93.3    47.4    86.8
forget           -       .    89.0    90.9    92.1       .    92.5    19.4    86.3
res_log          -       .    88.6    90.5    92.3       .    89.5    61.3    87.2
res_linear       -       .    88.1    90.2    92.4       .    91.1    40.8    87.8
res_square       -       .    88.8    90.7    91.8       .    90.5    35.9    88.0

### Detection TPR%/FPR%  (blend_p0.05)


sel\def          strip        scan
----------------------------------
random            3/10           -
forget             1/6           -
res_log            2/7           -
res_linear        5/11           -
res_square         3/7           -


## Coverage (parsed / total logs)

- **strip**: 5 / 5 parsed
- **scan**: 0 / 0 parsed
- **ac**: 5 / 5 parsed
- **abl**: 5 / 5 parsed
- **fp**: 5 / 5 parsed
- **fst**: 0 / 0 parsed
- **nc**: 5 / 5 parsed
- **rnp**: 5 / 5 parsed
- **i-bau**: 5 / 5 parsed
