# blend_p0.03 — defense breakdown

## blend_p0.03   (Blended)


### ASR after defense (%)  (blend_p0.03)


sel\def      strip    scan      ac     abl      fp     fst      nc     rnp   i-bau
----------------------------------------------------------------------------------
random           -       -    52.4    15.1    39.8    42.8    57.6    27.1    28.3
forget           -       -    73.2     9.2    63.5    64.1    76.1     0.0     7.0
res_log          -       -    76.9     3.9    64.3    69.5    74.9    36.1    20.0
res_linear       -       -    71.7     2.2    58.7    69.2    19.5     0.2    16.6
res_square       -       -    76.6     2.1    57.5    67.0    77.9     1.1    14.6
stealth          -       -    67.7     3.4    60.1    58.6    71.7    19.7    14.3

### Clean ACC after defense (%)  (blend_p0.03)


sel\def      strip    scan      ac     abl      fp     fst      nc     rnp   i-bau
----------------------------------------------------------------------------------
random           -       -    89.4    61.3    92.6    92.5    93.7    60.1    88.3
forget           -       -    89.5    59.8    92.7    92.3    93.3    27.0    85.6
res_log          -       -    89.3    53.9    92.4    91.9    93.1    47.8    87.2
res_linear       -       -    89.1    56.7    92.4    91.8    89.6    33.1    88.1
res_square       -       -    89.5    47.3    92.3    91.6    93.0    58.4    86.7
stealth          -       -    89.4    58.5    92.4    92.6    93.5    44.6    88.8

### Detection TPR%/FPR%  (blend_p0.03)


sel\def          strip        scan
----------------------------------
random             1/6        91/0
forget            2/10        96/0
res_log            4/8        95/0
res_linear         5/8        96/0
res_square         3/9        98/0
stealth            2/8        95/0


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
