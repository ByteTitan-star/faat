# badnet_p0.03 — defense breakdown

## badnet_p0.03   (BadNets)


### ASR after defense (%)  (badnet_p0.03)


sel\def      strip    scan      ac     abl      fp     fst      nc     rnp   i-bau
----------------------------------------------------------------------------------
random           -       -    18.0     0.1    14.4    22.0    18.8    10.5     8.0
forget           -       -    36.3     8.4    31.0    54.7    52.9     8.8    17.3
res_log          -       -    47.7     5.4    36.5    62.4    56.2    30.7    27.9
res_linear       -       -    43.8     4.4    35.3    58.3    54.3    11.5    14.8
res_square       -       -    46.2    14.0    31.0    48.2    54.1    36.3    22.1
stealth          -       -    25.6     3.5    20.2    40.5    37.9    34.9    32.5

### Clean ACC after defense (%)  (badnet_p0.03)


sel\def      strip    scan      ac     abl      fp     fst      nc     rnp   i-bau
----------------------------------------------------------------------------------
random           -       -    90.0    40.4    92.2    92.5    93.7    79.9    88.0
forget           -       -    89.5    48.0    92.0    91.8    93.1    46.0    87.8
res_log          -       -    89.3    55.7    91.6    91.9    92.8    56.3    88.2
res_linear       -       -    89.0    54.9    92.4    92.5    93.5    40.4    89.5
res_square       -       -    89.3    55.8    92.5    91.9    93.4    56.8    90.2
stealth          -       -    89.6    39.1    92.2    92.1    93.5    69.3    88.3

### Detection TPR%/FPR%  (badnet_p0.03)


sel\def          strip        scan
----------------------------------
random            7/11        82/0
forget           70/14        89/0
res_log           67/8        88/0
res_linear       64/11        87/0
res_square       75/12        88/0
stealth          35/11        82/0


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
