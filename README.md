# tempformer-xl

## Test Run to Reproduce

====================================================================================================
    - data : ../data/wikitext-103/
    - dataset : wt103
    - n_layer : 16
    - n_head : 10
    - d_head : 41
    - d_embed : 410
    - d_model : 410
    - d_inner : 2100
    - dropout : 0.1
    - dropatt : 0.0
    - init : normal
    - emb_init : normal
    - init_range : 0.1
    - emb_init_range : 0.01
    - init_std : 0.02
    - proj_init_std : 0.01
    - optim : adam
    - lr : 0.00025
    - mom : 0.0
    - scheduler : cosine
    - warmup_step : 0
    - decay_rate : 0.5
    - lr_min : 0.0
    - clip : 0.25
    - clip_nonemb : False
    - max_step : 200000
    - batch_size : 16
    - batch_chunk : 4
    - tgt_len : 150
    - eval_tgt_len : 150
    - ext_len : 0
    - mem_len : 150
    - not_tied : False
    - seed : 1111
    - cuda : True
    - adaptive : True
    - div_val : 1
    - pre_lnorm : False
    - varlen : False
    - multi_gpu : False
    - log_interval : 200
    - eval_interval : 4000
    - work_dir : LM-TFM-wt103/20210921-182940
    - restart : False
    - restart_dir : 
    - debug : False
    - same_length : False
    - attn_type : 0
    - clamp_len : -1
    - eta_min : 0.0
    - gpu0_bsz : 4
    - max_eval_steps : -1
    - sample_softmax : -1
    - patience : 0
    - finetune_v2 : False
    - finetune_v3 : False
    - fp16 : False
    - static_loss_scale : 1
    - dynamic_loss_scale : False
    - tied : True
    - n_token : 267735
    - n_all_param : 151107538
    - n_nonemb_param : 41066400
====================================================================================================
#params = 151107538
#non emb params = 41066400
| epoch   1 step      200 |    200 batches | lr 0.00025 | ms/batch 493.63 | loss  7.13 | ppl  1254.910
| epoch   1 step      400 |    400 batches | lr 0.00025 | ms/batch 493.89 | loss  6.40 | ppl   603.377
| epoch   1 step      600 |    600 batches | lr 0.00025 | ms/batch 493.86 | loss  6.09 | ppl   443.252
| epoch   1 step      800 |    800 batches | lr 0.00025 | ms/batch 494.39 | loss  5.95 | ppl   383.134
| epoch   1 step     1000 |   1000 batches | lr 0.00025 | ms/batch 495.03 | loss  5.79 | ppl   325.725
| epoch   1 step     1200 |   1200 batches | lr 0.00025 | ms/batch 496.30 | loss  5.67 | ppl   289.864
| epoch   1 step     1400 |   1400 batches | lr 0.00025 | ms/batch 494.21 | loss  5.56 | ppl   261.054
| epoch   1 step     1600 |   1600 batches | lr 0.00025 | ms/batch 494.64 | loss  5.48 | ppl   239.977
| epoch   1 step     1800 |   1800 batches | lr 0.00025 | ms/batch 494.97 | loss  5.38 | ppl   217.782
| epoch   1 step     2000 |   2000 batches | lr 0.00025 | ms/batch 494.88 | loss  5.31 | ppl   201.882
| epoch   1 step     2200 |   2200 batches | lr 0.00025 | ms/batch 495.40 | loss  5.33 | ppl   206.980
| epoch   1 step     2400 |   2400 batches | lr 0.00025 | ms/batch 494.62 | loss  5.21 | ppl   183.674
| epoch   1 step     2600 |   2600 batches | lr 0.00025 | ms/batch 496.11 | loss  5.19 | ppl   179.883
| epoch   1 step     2800 |   2800 batches | lr 0.00025 | ms/batch 496.94 | loss  5.19 | ppl   179.249
| epoch   1 step     3000 |   3000 batches | lr 0.00025 | ms/batch 497.89 | loss  5.15 | ppl   172.448
| epoch   1 step     3200 |   3200 batches | lr 0.00025 | ms/batch 496.77 | loss  5.07 | ppl   159.928
| epoch   1 step     3400 |   3400 batches | lr 0.00025 | ms/batch 497.92 | loss  5.04 | ppl   154.906
| epoch   1 step     3600 |   3600 batches | lr 0.00025 | ms/batch 495.40 | loss  4.96 | ppl   142.222
| epoch   1 step     3800 |   3800 batches | lr 0.00025 | ms/batch 495.15 | loss  5.04 | ppl   155.004
| epoch   1 step     4000 |   4000 batches | lr 0.00025 | ms/batch 493.45 | loss  4.91 | ppl   136.168
----------------------------------------------------------------------------------------------------
| Eval   1 at step     4000 | time: 1991.50s | valid loss  4.92 | valid ppl   137.152
----------------------------------------------------------------------------------------------------
| epoch   1 step     4200 |   4200 batches | lr 0.00025 | ms/batch 553.14 | loss  4.93 | ppl   138.226
| epoch   1 step     4400 |   4400 batches | lr 0.00025 | ms/batch 494.08 | loss  4.86 | ppl   128.969
| epoch   1 step     4600 |   4600 batches | lr 0.00025 | ms/batch 495.18 | loss  4.88 | ppl   131.913
| epoch   1 step     4800 |   4800 batches | lr 0.00025 | ms/batch 495.28 | loss  4.90 | ppl   133.687
| epoch   1 step     5000 |   5000 batches | lr 0.00025 | ms/batch 494.85 | loss  4.85 | ppl   127.346
| epoch   1 step     5200 |   5200 batches | lr 0.00025 | ms/batch 495.66 | loss  4.82 | ppl   123.937
| epoch   1 step     5400 |   5400 batches | lr 0.00025 | ms/batch 495.57 | loss  4.82 | ppl   124.102
| epoch   1 step     5600 |   5600 batches | lr 0.00025 | ms/batch 496.71 | loss  4.82 | ppl   123.789
| epoch   1 step     5800 |   5800 batches | lr 0.000249 | ms/batch 496.55 | loss  4.82 | ppl   123.759
| epoch   1 step     6000 |   6000 batches | lr 0.000249 | ms/batch 494.19 | loss  4.71 | ppl   110.718
| epoch   1 step     6200 |   6200 batches | lr 0.000249 | ms/batch 496.46 | loss  4.78 | ppl   118.613
| epoch   1 step     6400 |   6400 batches | lr 0.000249 | ms/batch 496.31 | loss  4.73 | ppl   113.399
| epoch   1 step     6600 |   6600 batches | lr 0.000249 | ms/batch 495.11 | loss  4.70 | ppl   110.007
| epoch   1 step     6800 |   6800 batches | lr 0.000249 | ms/batch 494.73 | loss  4.70 | ppl   109.674
| epoch   1 step     7000 |   7000 batches | lr 0.000249 | ms/batch 496.12 | loss  4.67 | ppl   106.686
| epoch   1 step     7200 |   7200 batches | lr 0.000249 | ms/batch 495.62 | loss  4.61 | ppl   100.260
| epoch   1 step     7400 |   7400 batches | lr 0.000249 | ms/batch 495.47 | loss  4.70 | ppl   109.966
| epoch   1 step     7600 |   7600 batches | lr 0.000249 | ms/batch 495.16 | loss  4.63 | ppl   102.847
| epoch   1 step     7800 |   7800 batches | lr 0.000249 | ms/batch 496.97 | loss  4.68 | ppl   107.495
| epoch   1 step     8000 |   8000 batches | lr 0.000249 | ms/batch 496.60 | loss  4.60 | ppl    99.703
----------------------------------------------------------------------------------------------------
| Eval   2 at step     8000 | time: 1992.63s | valid loss  4.52 | valid ppl    92.106
----------------------------------------------------------------------------------------------------
| epoch   1 step     8200 |   8200 batches | lr 0.000249 | ms/batch 557.80 | loss  4.58 | ppl    97.425
| epoch   1 step     8400 |   8400 batches | lr 0.000249 | ms/batch 494.54 | loss  4.57 | ppl    96.720
| epoch   1 step     8600 |   8600 batches | lr 0.000249 | ms/batch 495.85 | loss  4.58 | ppl    97.233
| epoch   1 step     8800 |   8800 batches | lr 0.000249 | ms/batch 495.76 | loss  4.57 | ppl    96.731
| epoch   1 step     9000 |   9000 batches | lr 0.000249 | ms/batch 496.09 | loss  4.57 | ppl    96.297
| epoch   1 step     9200 |   9200 batches | lr 0.000249 | ms/batch 495.72 | loss  4.52 | ppl    92.185
| epoch   1 step     9400 |   9400 batches | lr 0.000249 | ms/batch 495.37 | loss  4.53 | ppl    92.500
| epoch   1 step     9600 |   9600 batches | lr 0.000249 | ms/batch 496.28 | loss  4.54 | ppl    93.689
| epoch   1 step     9800 |   9800 batches | lr 0.000249 | ms/batch 497.26 | loss  4.54 | ppl    93.596
| epoch   1 step    10000 |  10000 batches | lr 0.000248 | ms/batch 495.58 | loss  4.51 | ppl    90.923
| epoch   1 step    10200 |  10200 batches | lr 0.000248 | ms/batch 493.49 | loss  4.45 | ppl    86.041
| epoch   1 step    10400 |  10400 batches | lr 0.000248 | ms/batch 495.47 | loss  4.52 | ppl    92.075
| epoch   1 step    10600 |  10600 batches | lr 0.000248 | ms/batch 496.26 | loss  4.56 | ppl    95.282
| epoch   1 step    10800 |  10800 batches | lr 0.000248 | ms/batch 496.28 | loss  4.56 | ppl    95.310
| epoch   1 step    11000 |  11000 batches | lr 0.000248 | ms/batch 495.41 | loss  4.44 | ppl    84.739
| epoch   1 step    11200 |  11200 batches | lr 0.000248 | ms/batch 494.72 | loss  4.45 | ppl    85.724
| epoch   1 step    11400 |  11400 batches | lr 0.000248 | ms/batch 495.96 | loss  4.49 | ppl    88.765
| epoch   1 step    11600 |  11600 batches | lr 0.000248 | ms/batch 495.57 | loss  4.48 | ppl    88.031
| epoch   1 step    11800 |  11800 batches | lr 0.000248 | ms/batch 493.73 | loss  4.40 | ppl    81.858
| epoch   1 step    12000 |  12000 batches | lr 0.000248 | ms/batch 495.55 | loss  4.43 | ppl    83.758
----------------------------------------------------------------------------------------------------
| Eval   3 at step    12000 | time: 1992.34s | valid loss  4.36 | valid ppl    78.589
----------------------------------------------------------------------------------------------------
| epoch   1 step    12200 |  12200 batches | lr 0.000248 | ms/batch 559.15 | loss  4.34 | ppl    76.575
| epoch   1 step    12400 |  12400 batches | lr 0.000248 | ms/batch 495.18 | loss  4.49 | ppl    88.876
| epoch   1 step    12600 |  12600 batches | lr 0.000248 | ms/batch 494.69 | loss  4.40 | ppl    81.145
| epoch   1 step    12800 |  12800 batches | lr 0.000247 | ms/batch 494.14 | loss  4.37 | ppl    79.132
| epoch   1 step    13000 |  13000 batches | lr 0.000247 | ms/batch 495.72 | loss  4.36 | ppl    77.931
| epoch   1 step    13200 |  13200 batches | lr 0.000247 | ms/batch 495.80 | loss  4.39 | ppl    80.577
| epoch   1 step    13400 |  13400 batches | lr 0.000247 | ms/batch 495.00 | loss  4.33 | ppl    75.747
| epoch   1 step    13600 |  13600 batches | lr 0.000247 | ms/batch 496.33 | loss  4.44 | ppl    84.548
| epoch   1 step    13800 |  13800 batches | lr 0.000247 | ms/batch 493.89 | loss  4.34 | ppl    76.400
| epoch   1 step    14000 |  14000 batches | lr 0.000247 | ms/batch 494.09 | loss  4.33 | ppl    75.718
| epoch   1 step    14200 |  14200 batches | lr 0.000247 | ms/batch 494.65 | loss  4.32 | ppl    75.012
| epoch   1 step    14400 |  14400 batches | lr 0.000247 | ms/batch 495.00 | loss  4.39 | ppl    80.497
| epoch   1 step    14600 |  14600 batches | lr 0.000247 | ms/batch 495.23 | loss  4.32 | ppl    75.051
| epoch   1 step    14800 |  14800 batches | lr 0.000247 | ms/batch 496.61 | loss  4.41 | ppl    82.513
| epoch   1 step    15000 |  15000 batches | lr 0.000247 | ms/batch 496.71 | loss  4.43 | ppl    83.742
| epoch   1 step    15200 |  15200 batches | lr 0.000246 | ms/batch 495.23 | loss  4.32 | ppl    74.858
| epoch   1 step    15400 |  15400 batches | lr 0.000246 | ms/batch 494.87 | loss  4.31 | ppl    74.685
| epoch   1 step    15600 |  15600 batches | lr 0.000246 | ms/batch 495.58 | loss  4.31 | ppl    74.456
| epoch   1 step    15800 |  15800 batches | lr 0.000246 | ms/batch 495.30 | loss  4.28 | ppl    72.591
| epoch   1 step    16000 |  16000 batches | lr 0.000246 | ms/batch 495.05 | loss  4.34 | ppl    76.342
----------------------------------------------------------------------------------------------------
| Eval   4 at step    16000 | time: 1991.43s | valid loss  4.22 | valid ppl    67.748
----------------------------------------------------------------------------------------------------
| epoch   1 step    16200 |  16200 batches | lr 0.000246 | ms/batch 557.94 | loss  4.23 | ppl    68.568
| epoch   1 step    16400 |  16400 batches | lr 0.000246 | ms/batch 495.36 | loss  4.30 | ppl    73.653
| epoch   1 step    16600 |  16600 batches | lr 0.000246 | ms/batch 496.68 | loss  4.33 | ppl    76.010
| epoch   1 step    16800 |  16800 batches | lr 0.000246 | ms/batch 494.43 | loss  4.27 | ppl    71.756
| epoch   1 step    17000 |  17000 batches | lr 0.000246 | ms/batch 495.25 | loss  4.31 | ppl    74.142
| epoch   1 step    17200 |  17200 batches | lr 0.000245 | ms/batch 493.69 | loss  4.23 | ppl    68.605
| epoch   1 step    17400 |  17400 batches | lr 0.000245 | ms/batch 496.04 | loss  4.28 | ppl    71.968
| epoch   1 step    17600 |  17600 batches | lr 0.000245 | ms/batch 494.80 | loss  4.28 | ppl    71.954
| epoch   1 step    17800 |  17800 batches | lr 0.000245 | ms/batch 496.22 | loss  4.29 | ppl    73.030
| epoch   1 step    18000 |  18000 batches | lr 0.000245 | ms/batch 494.77 | loss  4.28 | ppl    72.048
| epoch   1 step    18200 |  18200 batches | lr 0.000245 | ms/batch 496.33 | loss  4.29 | ppl    72.879
| epoch   1 step    18400 |  18400 batches | lr 0.000245 | ms/batch 494.48 | loss  4.27 | ppl    71.578
| epoch   1 step    18600 |  18600 batches | lr 0.000245 | ms/batch 495.50 | loss  4.31 | ppl    74.375
| epoch   1 step    18800 |  18800 batches | lr 0.000245 | ms/batch 494.74 | loss  4.23 | ppl    68.390
| epoch   1 step    19000 |  19000 batches | lr 0.000244 | ms/batch 494.98 | loss  4.25 | ppl    70.149
| epoch   1 step    19200 |  19200 batches | lr 0.000244 | ms/batch 494.33 | loss  4.29 | ppl    72.882
| epoch   1 step    19400 |  19400 batches | lr 0.000244 | ms/batch 495.57 | loss  4.19 | ppl    66.004
| epoch   1 step    19600 |  19600 batches | lr 0.000244 | ms/batch 495.22 | loss  4.26 | ppl    70.812
| epoch   1 step    19800 |  19800 batches | lr 0.000244 | ms/batch 493.93 | loss  4.26 | ppl    70.826
| epoch   1 step    20000 |  20000 batches | lr 0.000244 | ms/batch 495.25 | loss  4.23 | ppl    68.646
----------------------------------------------------------------------------------------------------
| Eval   5 at step    20000 | time: 1990.91s | valid loss  4.13 | valid ppl    61.969
----------------------------------------------------------------------------------------------------
| epoch   1 step    20200 |  20200 batches | lr 0.000244 | ms/batch 559.08 | loss  4.23 | ppl    68.973
| epoch   1 step    20400 |  20400 batches | lr 0.000244 | ms/batch 494.76 | loss  4.27 | ppl    71.687
| epoch   1 step    20600 |  20600 batches | lr 0.000244 | ms/batch 495.68 | loss  4.21 | ppl    67.664
| epoch   1 step    20800 |  20800 batches | lr 0.000243 | ms/batch 494.72 | loss  4.26 | ppl    70.515
| epoch   1 step    21000 |  21000 batches | lr 0.000243 | ms/batch 494.79 | loss  4.21 | ppl    67.453
| epoch   1 step    21200 |  21200 batches | lr 0.000243 | ms/batch 497.26 | loss  4.28 | ppl    72.279
| epoch   1 step    21400 |  21400 batches | lr 0.000243 | ms/batch 495.60 | loss  4.23 | ppl    68.897
| epoch   1 step    21600 |  21600 batches | lr 0.000243 | ms/batch 495.42 | loss  4.22 | ppl    68.008
| epoch   1 step    21800 |  21800 batches | lr 0.000243 | ms/batch 494.56 | loss  4.22 | ppl    67.926
| epoch   1 step    22000 |  22000 batches | lr 0.000243 | ms/batch 495.34 | loss  4.22 | ppl    67.941
| epoch   1 step    22200 |  22200 batches | lr 0.000242 | ms/batch 494.26 | loss  4.18 | ppl    65.468
| epoch   1 step    22400 |  22400 batches | lr 0.000242 | ms/batch 497.13 | loss  4.30 | ppl    74.059
| epoch   1 step    22600 |  22600 batches | lr 0.000242 | ms/batch 495.71 | loss  4.25 | ppl    70.022
| epoch   1 step    22800 |  22800 batches | lr 0.000242 | ms/batch 496.27 | loss  4.21 | ppl    67.276
| epoch   1 step    23000 |  23000 batches | lr 0.000242 | ms/batch 495.30 | loss  4.18 | ppl    65.076
| epoch   1 step    23200 |  23200 batches | lr 0.000242 | ms/batch 495.81 | loss  4.22 | ppl    67.702
| epoch   1 step    23400 |  23400 batches | lr 0.000242 | ms/batch 496.40 | loss  4.28 | ppl    72.323
| epoch   1 step    23600 |  23600 batches | lr 0.000242 | ms/batch 496.13 | loss  4.20 | ppl    66.390
| epoch   1 step    23800 |  23800 batches | lr 0.000241 | ms/batch 496.34 | loss  4.18 | ppl    65.382
| epoch   1 step    24000 |  24000 batches | lr 0.000241 | ms/batch 494.92 | loss  4.20 | ppl    66.459
----------------------------------------------------------------------------------------------------
| Eval   6 at step    24000 | time: 1992.92s | valid loss  4.04 | valid ppl    57.046
----------------------------------------------------------------------------------------------------
| epoch   1 step    24200 |  24200 batches | lr 0.000241 | ms/batch 559.25 | loss  4.21 | ppl    67.595
| epoch   1 step    24400 |  24400 batches | lr 0.000241 | ms/batch 496.58 | loss  4.23 | ppl    68.785
| epoch   1 step    24600 |  24600 batches | lr 0.000241 | ms/batch 495.04 | loss  4.18 | ppl    65.233
| epoch   1 step    24800 |  24800 batches | lr 0.000241 | ms/batch 494.46 | loss  4.09 | ppl    59.573
| epoch   1 step    25000 |  25000 batches | lr 0.00024 | ms/batch 495.32 | loss  4.15 | ppl    63.328
| epoch   1 step    25200 |  25200 batches | lr 0.00024 | ms/batch 495.26 | loss  4.20 | ppl    66.452
| epoch   1 step    25400 |  25400 batches | lr 0.00024 | ms/batch 494.97 | loss  4.18 | ppl    65.285
| epoch   1 step    25600 |  25600 batches | lr 0.00024 | ms/batch 495.18 | loss  4.19 | ppl    66.063
| epoch   1 step    25800 |  25800 batches | lr 0.00024 | ms/batch 495.22 | loss  4.12 | ppl    61.621
| epoch   1 step    26000 |  26000 batches | lr 0.00024 | ms/batch 495.01 | loss  4.13 | ppl    62.086
| epoch   1 step    26200 |  26200 batches | lr 0.00024 | ms/batch 494.93 | loss  4.17 | ppl    64.889
| epoch   1 step    26400 |  26400 batches | lr 0.000239 | ms/batch 495.86 | loss  4.15 | ppl    63.361
| epoch   1 step    26600 |  26600 batches | lr 0.000239 | ms/batch 496.34 | loss  4.18 | ppl    65.286
| epoch   1 step    26800 |  26800 batches | lr 0.000239 | ms/batch 494.61 | loss  4.12 | ppl    61.841
| epoch   1 step    27000 |  27000 batches | lr 0.000239 | ms/batch 494.51 | loss  4.14 | ppl    62.551
| epoch   1 step    27200 |  27200 batches | lr 0.000239 | ms/batch 494.92 | loss  4.06 | ppl    58.149
| epoch   1 step    27400 |  27400 batches | lr 0.000239 | ms/batch 495.30 | loss  4.09 | ppl    60.002
| epoch   1 step    27600 |  27600 batches | lr 0.000238 | ms/batch 494.81 | loss  4.14 | ppl    62.604
| epoch   1 step    27800 |  27800 batches | lr 0.000238 | ms/batch 495.34 | loss  4.14 | ppl    62.556
| epoch   1 step    28000 |  28000 batches | lr 0.000238 | ms/batch 494.90 | loss  4.14 | ppl    62.946
----------------------------------------------------------------------------------------------------
| Eval   7 at step    28000 | time: 1991.22s | valid loss  3.97 | valid ppl    53.117
----------------------------------------------------------------------------------------------------
| epoch   1 step    28200 |  28200 batches | lr 0.000238 | ms/batch 558.32 | loss  4.21 | ppl    67.249
| epoch   1 step    28400 |  28400 batches | lr 0.000238 | ms/batch 495.47 | loss  4.16 | ppl    63.903
| epoch   1 step    28600 |  28600 batches | lr 0.000238 | ms/batch 496.60 | loss  4.16 | ppl    64.322
| epoch   1 step    28800 |  28800 batches | lr 0.000237 | ms/batch 496.06 | loss  4.10 | ppl    60.291
| epoch   1 step    29000 |  29000 batches | lr 0.000237 | ms/batch 495.81 | loss  4.09 | ppl    59.745
| epoch   1 step    29200 |  29200 batches | lr 0.000237 | ms/batch 494.19 | loss  4.10 | ppl    60.189
| epoch   1 step    29400 |  29400 batches | lr 0.000237 | ms/batch 495.97 | loss  4.18 | ppl    65.042
| epoch   1 step    29600 |  29600 batches | lr 0.000237 | ms/batch 494.26 | loss  4.09 | ppl    59.554
| epoch   1 step    29800 |  29800 batches | lr 0.000237 | ms/batch 496.95 | loss  4.15 | ppl    63.361
| epoch   1 step    30000 |  30000 batches | lr 0.000236 | ms/batch 495.55 | loss  4.19 | ppl    66.002
| epoch   1 step    30200 |  30200 batches | lr 0.000236 | ms/batch 496.41 | loss  4.16 | ppl    64.291
| epoch   1 step    30400 |  30400 batches | lr 0.000236 | ms/batch 494.15 | loss  4.10 | ppl    60.164
| epoch   1 step    30600 |  30600 batches | lr 0.000236 | ms/batch 496.83 | loss  4.18 | ppl    65.406
| epoch   1 step    30800 |  30800 batches | lr 0.000236 | ms/batch 496.48 | loss  4.12 | ppl    61.410
| epoch   1 step    31000 |  31000 batches | lr 0.000235 | ms/batch 495.90 | loss  4.13 | ppl    62.201
| epoch   1 step    31200 |  31200 batches | lr 0.000235 | ms/batch 496.49 | loss  4.05 | ppl    57.348
| epoch   1 step    31400 |  31400 batches | lr 0.000235 | ms/batch 496.77 | loss  4.17 | ppl    64.479
| epoch   1 step    31600 |  31600 batches | lr 0.000235 | ms/batch 494.30 | loss  4.10 | ppl    60.633
| epoch   1 step    31800 |  31800 batches | lr 0.000235 | ms/batch 496.40 | loss  4.13 | ppl    62.229
| epoch   1 step    32000 |  32000 batches | lr 0.000235 | ms/batch 495.74 | loss  4.10 | ppl    60.596
----------------------------------------------------------------------------------------------------
| Eval   8 at step    32000 | time: 1993.61s | valid loss  3.93 | valid ppl    51.052
----------------------------------------------------------------------------------------------------
| epoch   1 step    32200 |  32200 batches | lr 0.000234 | ms/batch 557.24 | loss  4.10 | ppl    60.198
| epoch   1 step    32400 |  32400 batches | lr 0.000234 | ms/batch 495.39 | loss  4.05 | ppl    57.633
| epoch   1 step    32600 |  32600 batches | lr 0.000234 | ms/batch 493.81 | loss  4.05 | ppl    57.556
| epoch   1 step    32800 |  32800 batches | lr 0.000234 | ms/batch 493.47 | loss  3.98 | ppl    53.459
| epoch   1 step    33000 |  33000 batches | lr 0.000234 | ms/batch 495.80 | loss  4.07 | ppl    58.451
| epoch   1 step    33200 |  33200 batches | lr 0.000233 | ms/batch 494.68 | loss  4.12 | ppl    61.372
| epoch   1 step    33400 |  33400 batches | lr 0.000233 | ms/batch 493.90 | loss  4.02 | ppl    55.470
| epoch   1 step    33600 |  33600 batches | lr 0.000233 | ms/batch 494.79 | loss  4.07 | ppl    58.691
| epoch   1 step    33800 |  33800 batches | lr 0.000233 | ms/batch 494.25 | loss  4.10 | ppl    60.156
| epoch   1 step    34000 |  34000 batches | lr 0.000233 | ms/batch 495.29 | loss  4.09 | ppl    59.498
| epoch   1 step    34200 |  34200 batches | lr 0.000232 | ms/batch 495.41 | loss  4.12 | ppl    61.711
| epoch   1 step    34400 |  34400 batches | lr 0.000232 | ms/batch 495.29 | loss  4.10 | ppl    60.081
| epoch   1 step    34600 |  34600 batches | lr 0.000232 | ms/batch 495.62 | loss  4.07 | ppl    58.368
| epoch   1 step    34800 |  34800 batches | lr 0.000232 | ms/batch 494.25 | loss  3.97 | ppl    53.031
| epoch   1 step    35000 |  35000 batches | lr 0.000232 | ms/batch 494.85 | loss  4.03 | ppl    56.116
| epoch   1 step    35200 |  35200 batches | lr 0.000231 | ms/batch 496.65 | loss  4.05 | ppl    57.620
| epoch   1 step    35400 |  35400 batches | lr 0.000231 | ms/batch 497.25 | loss  4.12 | ppl    61.276
| epoch   1 step    35600 |  35600 batches | lr 0.000231 | ms/batch 495.56 | loss  4.07 | ppl    58.342
| epoch   1 step    35800 |  35800 batches | lr 0.000231 | ms/batch 494.66 | loss  4.06 | ppl    57.992
| epoch   1 step    36000 |  36000 batches | lr 0.000231 | ms/batch 494.61 | loss  4.02 | ppl    55.884
----------------------------------------------------------------------------------------------------
| Eval   9 at step    36000 | time: 1990.56s | valid loss  3.88 | valid ppl    48.337
----------------------------------------------------------------------------------------------------
| epoch   1 step    36200 |  36200 batches | lr 0.00023 | ms/batch 557.43 | loss  4.09 | ppl    59.777
| epoch   1 step    36400 |  36400 batches | lr 0.00023 | ms/batch 494.63 | loss  4.09 | ppl    59.574
| epoch   1 step    36600 |  36600 batches | lr 0.00023 | ms/batch 495.90 | loss  4.07 | ppl    58.375
| epoch   1 step    36800 |  36800 batches | lr 0.00023 | ms/batch 496.20 | loss  4.10 | ppl    60.599
| epoch   1 step    37000 |  37000 batches | lr 0.000229 | ms/batch 494.16 | loss  4.01 | ppl    55.261
| epoch   1 step    37200 |  37200 batches | lr 0.000229 | ms/batch 494.10 | loss  4.01 | ppl    55.018
| epoch   1 step    37400 |  37400 batches | lr 0.000229 | ms/batch 495.78 | loss  4.07 | ppl    58.684
| epoch   1 step    37600 |  37600 batches | lr 0.000229 | ms/batch 496.04 | loss  4.07 | ppl    58.281
| epoch   1 step    37800 |  37800 batches | lr 0.000229 | ms/batch 494.88 | loss  4.06 | ppl    58.206
| epoch   1 step    38000 |  38000 batches | lr 0.000228 | ms/batch 494.31 | loss  4.01 | ppl    55.040
| epoch   1 step    38200 |  38200 batches | lr 0.000228 | ms/batch 495.86 | loss  4.05 | ppl    57.358
| epoch   1 step    38400 |  38400 batches | lr 0.000228 | ms/batch 495.35 | loss  4.06 | ppl    57.934
| epoch   1 step    38600 |  38600 batches | lr 0.000228 | ms/batch 495.88 | loss  4.07 | ppl    58.393
| epoch   1 step    38800 |  38800 batches | lr 0.000227 | ms/batch 496.58 | loss  4.05 | ppl    57.265
| epoch   1 step    39000 |  39000 batches | lr 0.000227 | ms/batch 495.65 | loss  4.04 | ppl    56.597
| epoch   1 step    39200 |  39200 batches | lr 0.000227 | ms/batch 495.46 | loss  4.07 | ppl    58.739
| epoch   1 step    39400 |  39400 batches | lr 0.000227 | ms/batch 495.33 | loss  4.03 | ppl    56.401
| epoch   1 step    39600 |  39600 batches | lr 0.000227 | ms/batch 495.20 | loss  3.94 | ppl    51.659
| epoch   1 step    39800 |  39800 batches | lr 0.000226 | ms/batch 495.55 | loss  4.00 | ppl    54.441
| epoch   1 step    40000 |  40000 batches | lr 0.000226 | ms/batch 495.02 | loss  3.95 | ppl    51.827
----------------------------------------------------------------------------------------------------
| Eval  10 at step    40000 | time: 1991.55s | valid loss  3.84 | valid ppl    46.715
----------------------------------------------------------------------------------------------------
| epoch   1 step    40200 |  40200 batches | lr 0.000226 | ms/batch 558.01 | loss  4.08 | ppl    59.363
| epoch   1 step    40400 |  40400 batches | lr 0.000226 | ms/batch 496.85 | loss  4.04 | ppl    56.787
| epoch   1 step    40600 |  40600 batches | lr 0.000225 | ms/batch 495.07 | loss  4.01 | ppl    55.161
| epoch   1 step    40800 |  40800 batches | lr 0.000225 | ms/batch 494.20 | loss  4.03 | ppl    56.332
| epoch   1 step    41000 |  41000 batches | lr 0.000225 | ms/batch 495.88 | loss  4.06 | ppl    57.847
| epoch   1 step    41200 |  41200 batches | lr 0.000225 | ms/batch 495.97 | loss  4.05 | ppl    57.524
| epoch   1 step    41400 |  41400 batches | lr 0.000224 | ms/batch 496.80 | loss  4.08 | ppl    59.077
| epoch   1 step    41600 |  41600 batches | lr 0.000224 | ms/batch 495.89 | loss  4.06 | ppl    58.196
| epoch   1 step    41800 |  41800 batches | lr 0.000224 | ms/batch 496.28 | loss  4.03 | ppl    56.345
| epoch   1 step    42000 |  42000 batches | lr 0.000224 | ms/batch 495.94 | loss  4.02 | ppl    55.501
| epoch   1 step    42200 |  42200 batches | lr 0.000224 | ms/batch 496.29 | loss  4.06 | ppl    58.231
| epoch   1 step    42400 |  42400 batches | lr 0.000223 | ms/batch 495.29 | loss  3.97 | ppl    52.860
| epoch   1 step    42600 |  42600 batches | lr 0.000223 | ms/batch 496.21 | loss  4.06 | ppl    57.864
| epoch   1 step    42800 |  42800 batches | lr 0.000223 | ms/batch 496.03 | loss  4.04 | ppl    56.727
| epoch   1 step    43000 |  43000 batches | lr 0.000223 | ms/batch 495.90 | loss  3.98 | ppl    53.271
| epoch   2 step    43200 |    188 batches | lr 0.000222 | ms/batch 493.50 | loss  3.92 | ppl    50.609
| epoch   2 step    43400 |    388 batches | lr 0.000222 | ms/batch 495.47 | loss  4.01 | ppl    55.227
| epoch   2 step    43600 |    588 batches | lr 0.000222 | ms/batch 495.32 | loss  3.94 | ppl    51.605
| epoch   2 step    43800 |    788 batches | lr 0.000222 | ms/batch 495.97 | loss  3.97 | ppl    53.222
| epoch   2 step    44000 |    988 batches | lr 0.000221 | ms/batch 495.38 | loss  3.97 | ppl    52.989
----------------------------------------------------------------------------------------------------
| Eval  11 at step    44000 | time: 1993.15s | valid loss  3.83 | valid ppl    46.109
----------------------------------------------------------------------------------------------------
| epoch   2 step    44200 |   1188 batches | lr 0.000221 | ms/batch 559.07 | loss  3.97 | ppl    53.248
| epoch   2 step    44400 |   1388 batches | lr 0.000221 | ms/batch 494.68 | loss  3.96 | ppl    52.667
| epoch   2 step    44600 |   1588 batches | lr 0.000221 | ms/batch 494.35 | loss  3.96 | ppl    52.444
| epoch   2 step    44800 |   1788 batches | lr 0.00022 | ms/batch 494.91 | loss  3.93 | ppl    51.038
| epoch   2 step    45000 |   1988 batches | lr 0.00022 | ms/batch 494.94 | loss  3.87 | ppl    47.784
| epoch   2 step    45200 |   2188 batches | lr 0.00022 | ms/batch 495.42 | loss  4.00 | ppl    54.509
| epoch   2 step    45400 |   2388 batches | lr 0.00022 | ms/batch 494.41 | loss  3.93 | ppl    51.073
| epoch   2 step    45600 |   2588 batches | lr 0.000219 | ms/batch 495.58 | loss  3.96 | ppl    52.590
| epoch   2 step    45800 |   2788 batches | lr 0.000219 | ms/batch 496.16 | loss  4.02 | ppl    55.726
| epoch   2 step    46000 |   2988 batches | lr 0.000219 | ms/batch 496.33 | loss  4.00 | ppl    54.416
| epoch   2 step    46200 |   3188 batches | lr 0.000219 | ms/batch 495.82 | loss  3.98 | ppl    53.610
| epoch   2 step    46400 |   3388 batches | lr 0.000218 | ms/batch 496.96 | loss  3.93 | ppl    51.086
| epoch   2 step    46600 |   3588 batches | lr 0.000218 | ms/batch 495.24 | loss  3.90 | ppl    49.277
| epoch   2 step    46800 |   3788 batches | lr 0.000218 | ms/batch 495.73 | loss  3.98 | ppl    53.441
| epoch   2 step    47000 |   3988 batches | lr 0.000217 | ms/batch 493.89 | loss  3.91 | ppl    50.059
| epoch   2 step    47200 |   4188 batches | lr 0.000217 | ms/batch 494.70 | loss  3.96 | ppl    52.327
| epoch   2 step    47400 |   4388 batches | lr 0.000217 | ms/batch 494.54 | loss  3.90 | ppl    49.602
| epoch   2 step    47600 |   4588 batches | lr 0.000217 | ms/batch 495.68 | loss  3.96 | ppl    52.449
| epoch   2 step    47800 |   4788 batches | lr 0.000216 | ms/batch 496.07 | loss  3.97 | ppl    53.221
| epoch   2 step    48000 |   4988 batches | lr 0.000216 | ms/batch 495.15 | loss  3.95 | ppl    52.156
----------------------------------------------------------------------------------------------------
| Eval  12 at step    48000 | time: 1991.81s | valid loss  3.78 | valid ppl    43.968
----------------------------------------------------------------------------------------------------
| epoch   2 step    48200 |   5188 batches | lr 0.000216 | ms/batch 558.46 | loss  3.92 | ppl    50.446
| epoch   2 step    48400 |   5388 batches | lr 0.000216 | ms/batch 494.79 | loss  3.99 | ppl    53.853
| epoch   2 step    48600 |   5588 batches | lr 0.000215 | ms/batch 496.17 | loss  3.96 | ppl    52.632
| epoch   2 step    48800 |   5788 batches | lr 0.000215 | ms/batch 495.78 | loss  3.99 | ppl    54.164
| epoch   2 step    49000 |   5988 batches | lr 0.000215 | ms/batch 493.52 | loss  3.91 | ppl    49.760
| epoch   2 step    49200 |   6188 batches | lr 0.000214 | ms/batch 495.37 | loss  4.00 | ppl    54.738
| epoch   2 step    49400 |   6388 batches | lr 0.000214 | ms/batch 495.41 | loss  3.97 | ppl    52.745
| epoch   2 step    49600 |   6588 batches | lr 0.000214 | ms/batch 495.17 | loss  3.93 | ppl    50.900
| epoch   2 step    49800 |   6788 batches | lr 0.000214 | ms/batch 494.18 | loss  3.95 | ppl    52.117
| epoch   2 step    50000 |   6988 batches | lr 0.000213 | ms/batch 495.74 | loss  3.94 | ppl    51.518
| epoch   2 step    50200 |   7188 batches | lr 0.000213 | ms/batch 495.40 | loss  3.88 | ppl    48.458
| epoch   2 step    50400 |   7388 batches | lr 0.000213 | ms/batch 495.16 | loss  3.99 | ppl    53.891
| epoch   2 step    50600 |   7588 batches | lr 0.000213 | ms/batch 494.60 | loss  3.92 | ppl    50.577
| epoch   2 step    50800 |   7788 batches | lr 0.000212 | ms/batch 496.40 | loss  4.00 | ppl    54.519
| epoch   2 step    51000 |   7988 batches | lr 0.000212 | ms/batch 496.47 | loss  3.93 | ppl    51.114
| epoch   2 step    51200 |   8188 batches | lr 0.000212 | ms/batch 494.86 | loss  3.89 | ppl    49.058
| epoch   2 step    51400 |   8388 batches | lr 0.000211 | ms/batch 494.24 | loss  3.91 | ppl    50.115
| epoch   2 step    51600 |   8588 batches | lr 0.000211 | ms/batch 495.53 | loss  3.92 | ppl    50.483
| epoch   2 step    51800 |   8788 batches | lr 0.000211 | ms/batch 495.57 | loss  3.93 | ppl    50.802
| epoch   2 step    52000 |   8988 batches | lr 0.000211 | ms/batch 496.46 | loss  3.94 | ppl    51.316
----------------------------------------------------------------------------------------------------
| Eval  13 at step    52000 | time: 1991.74s | valid loss  3.76 | valid ppl    42.893
----------------------------------------------------------------------------------------------------
| epoch   2 step    52200 |   9188 batches | lr 0.00021 | ms/batch 559.02 | loss  3.91 | ppl    49.795
| epoch   2 step    52400 |   9388 batches | lr 0.00021 | ms/batch 495.80 | loss  3.90 | ppl    49.541
| epoch   2 step    52600 |   9588 batches | lr 0.00021 | ms/batch 496.53 | loss  3.94 | ppl    51.324
| epoch   2 step    52800 |   9788 batches | lr 0.000209 | ms/batch 497.30 | loss  3.94 | ppl    51.604
| epoch   2 step    53000 |   9988 batches | lr 0.000209 | ms/batch 495.88 | loss  3.91 | ppl    49.936
| epoch   2 step    53200 |  10188 batches | lr 0.000209 | ms/batch 493.70 | loss  3.90 | ppl    49.410
| epoch   2 step    53400 |  10388 batches | lr 0.000209 | ms/batch 495.61 | loss  3.94 | ppl    51.630
| epoch   2 step    53600 |  10588 batches | lr 0.000208 | ms/batch 496.78 | loss  3.97 | ppl    53.035
| epoch   2 step    53800 |  10788 batches | lr 0.000208 | ms/batch 496.68 | loss  4.02 | ppl    55.639
| epoch   2 step    54000 |  10988 batches | lr 0.000208 | ms/batch 495.55 | loss  3.88 | ppl    48.318
| epoch   2 step    54200 |  11188 batches | lr 0.000207 | ms/batch 495.30 | loss  3.93 | ppl    51.040
| epoch   2 step    54400 |  11388 batches | lr 0.000207 | ms/batch 496.20 | loss  3.95 | ppl    52.192
| epoch   2 step    54600 |  11588 batches | lr 0.000207 | ms/batch 495.87 | loss  3.91 | ppl    50.140
| epoch   2 step    54800 |  11788 batches | lr 0.000206 | ms/batch 494.11 | loss  3.88 | ppl    48.454
| epoch   2 step    55000 |  11988 batches | lr 0.000206 | ms/batch 495.70 | loss  3.91 | ppl    49.908
| epoch   2 step    55200 |  12188 batches | lr 0.000206 | ms/batch 496.02 | loss  3.82 | ppl    45.716
| epoch   2 step    55400 |  12388 batches | lr 0.000206 | ms/batch 495.38 | loss  3.97 | ppl    53.052
| epoch   2 step    55600 |  12588 batches | lr 0.000205 | ms/batch 495.23 | loss  3.89 | ppl    48.827
| epoch   2 step    55800 |  12788 batches | lr 0.000205 | ms/batch 494.48 | loss  3.87 | ppl    47.978
| epoch   2 step    56000 |  12988 batches | lr 0.000205 | ms/batch 496.10 | loss  3.86 | ppl    47.343
----------------------------------------------------------------------------------------------------
| Eval  14 at step    56000 | time: 1993.32s | valid loss  3.74 | valid ppl    42.019
----------------------------------------------------------------------------------------------------
| epoch   2 step    56200 |  13188 batches | lr 0.000204 | ms/batch 558.95 | loss  3.88 | ppl    48.464
| epoch   2 step    56400 |  13388 batches | lr 0.000204 | ms/batch 495.17 | loss  3.84 | ppl    46.385
| epoch   2 step    56600 |  13588 batches | lr 0.000204 | ms/batch 496.78 | loss  3.96 | ppl    52.693
| epoch   2 step    56800 |  13788 batches | lr 0.000203 | ms/batch 494.41 | loss  3.86 | ppl    47.471
| epoch   2 step    57000 |  13988 batches | lr 0.000203 | ms/batch 494.72 | loss  3.84 | ppl    46.647
| epoch   2 step    57200 |  14188 batches | lr 0.000203 | ms/batch 495.21 | loss  3.86 | ppl    47.382
| epoch   2 step    57400 |  14388 batches | lr 0.000203 | ms/batch 495.05 | loss  3.92 | ppl    50.358
| epoch   2 step    57600 |  14588 batches | lr 0.000202 | ms/batch 495.93 | loss  3.86 | ppl    47.396
| epoch   2 step    57800 |  14788 batches | lr 0.000202 | ms/batch 497.39 | loss  3.96 | ppl    52.268
| epoch   2 step    58000 |  14988 batches | lr 0.000202 | ms/batch 497.55 | loss  3.99 | ppl    54.254
| epoch   2 step    58200 |  15188 batches | lr 0.000201 | ms/batch 495.94 | loss  3.86 | ppl    47.381
| epoch   2 step    58400 |  15388 batches | lr 0.000201 | ms/batch 495.54 | loss  3.89 | ppl    48.709
| epoch   2 step    58600 |  15588 batches | lr 0.000201 | ms/batch 495.59 | loss  3.86 | ppl    47.269
| epoch   2 step    58800 |  15788 batches | lr 0.0002 | ms/batch 495.87 | loss  3.86 | ppl    47.621
| epoch   2 step    59000 |  15988 batches | lr 0.0002 | ms/batch 495.48 | loss  3.89 | ppl    49.104
| epoch   2 step    59200 |  16188 batches | lr 0.0002 | ms/batch 495.65 | loss  3.82 | ppl    45.749
| epoch   2 step    59400 |  16388 batches | lr 0.000199 | ms/batch 495.87 | loss  3.86 | ppl    47.636
| epoch   2 step    59600 |  16588 batches | lr 0.000199 | ms/batch 497.22 | loss  3.90 | ppl    49.258
| epoch   2 step    59800 |  16788 batches | lr 0.000199 | ms/batch 495.60 | loss  3.85 | ppl    46.765
| epoch   2 step    60000 |  16988 batches | lr 0.000198 | ms/batch 496.19 | loss  3.89 | ppl    48.787
----------------------------------------------------------------------------------------------------
| Eval  15 at step    60000 | time: 1993.97s | valid loss  3.71 | valid ppl    41.041
----------------------------------------------------------------------------------------------------
| epoch   2 step    60200 |  17188 batches | lr 0.000198 | ms/batch 557.08 | loss  3.83 | ppl    46.022
| epoch   2 step    60400 |  17388 batches | lr 0.000198 | ms/batch 496.48 | loss  3.88 | ppl    48.226
| epoch   2 step    60600 |  17588 batches | lr 0.000198 | ms/batch 495.33 | loss  3.88 | ppl    48.220
| epoch   2 step    60800 |  17788 batches | lr 0.000197 | ms/batch 496.65 | loss  3.88 | ppl    48.574
| epoch   2 step    61000 |  17988 batches | lr 0.000197 | ms/batch 495.47 | loss  3.88 | ppl    48.547
| epoch   2 step    61200 |  18188 batches | lr 0.000197 | ms/batch 496.48 | loss  3.90 | ppl    49.269
| epoch   2 step    61400 |  18388 batches | lr 0.000196 | ms/batch 494.84 | loss  3.88 | ppl    48.486
| epoch   2 step    61600 |  18588 batches | lr 0.000196 | ms/batch 496.53 | loss  3.93 | ppl    50.705
| epoch   2 step    61800 |  18788 batches | lr 0.000196 | ms/batch 495.09 | loss  3.83 | ppl    46.292
| epoch   2 step    62000 |  18988 batches | lr 0.000195 | ms/batch 495.88 | loss  3.87 | ppl    47.904
| epoch   2 step    62200 |  19188 batches | lr 0.000195 | ms/batch 494.84 | loss  3.93 | ppl    51.085
| epoch   2 step    62400 |  19388 batches | lr 0.000195 | ms/batch 496.22 | loss  3.81 | ppl    45.232
| epoch   2 step    62600 |  19588 batches | lr 0.000194 | ms/batch 496.07 | loss  3.89 | ppl    48.883
| epoch   2 step    62800 |  19788 batches | lr 0.000194 | ms/batch 494.66 | loss  3.90 | ppl    49.350
| epoch   2 step    63000 |  19988 batches | lr 0.000194 | ms/batch 495.95 | loss  3.87 | ppl    47.748
| epoch   2 step    63200 |  20188 batches | lr 0.000193 | ms/batch 496.49 | loss  3.85 | ppl    47.091
| epoch   2 step    63400 |  20388 batches | lr 0.000193 | ms/batch 495.47 | loss  3.91 | ppl    49.997
| epoch   2 step    63600 |  20588 batches | lr 0.000193 | ms/batch 496.90 | loss  3.86 | ppl    47.537
| epoch   2 step    63800 |  20788 batches | lr 0.000192 | ms/batch 495.20 | loss  3.90 | ppl    49.301
| epoch   2 step    64000 |  20988 batches | lr 0.000192 | ms/batch 496.10 | loss  3.86 | ppl    47.668
----------------------------------------------------------------------------------------------------
| Eval  16 at step    64000 | time: 1993.39s | valid loss  3.71 | valid ppl    40.796
----------------------------------------------------------------------------------------------------
| epoch   2 step    64200 |  21188 batches | lr 0.000192 | ms/batch 560.14 | loss  3.92 | ppl    50.365
| epoch   2 step    64400 |  21388 batches | lr 0.000191 | ms/batch 496.35 | loss  3.89 | ppl    49.015
| epoch   2 step    64600 |  21588 batches | lr 0.000191 | ms/batch 496.19 | loss  3.87 | ppl    48.032
| epoch   2 step    64800 |  21788 batches | lr 0.000191 | ms/batch 495.09 | loss  3.86 | ppl    47.602
| epoch   2 step    65000 |  21988 batches | lr 0.00019 | ms/batch 496.32 | loss  3.88 | ppl    48.326
| epoch   2 step    65200 |  22188 batches | lr 0.00019 | ms/batch 495.11 | loss  3.83 | ppl    46.153
| epoch   2 step    65400 |  22388 batches | lr 0.00019 | ms/batch 497.54 | loss  3.98 | ppl    53.359
| epoch   2 step    65600 |  22588 batches | lr 0.000189 | ms/batch 496.12 | loss  3.90 | ppl    49.293
| epoch   2 step    65800 |  22788 batches | lr 0.000189 | ms/batch 496.86 | loss  3.88 | ppl    48.452
| epoch   2 step    66000 |  22988 batches | lr 0.000189 | ms/batch 495.92 | loss  3.85 | ppl    47.180
| epoch   2 step    66200 |  23188 batches | lr 0.000188 | ms/batch 495.95 | loss  3.86 | ppl    47.666
| epoch   2 step    66400 |  23388 batches | lr 0.000188 | ms/batch 496.74 | loss  3.94 | ppl    51.185
| epoch   2 step    66600 |  23588 batches | lr 0.000188 | ms/batch 496.43 | loss  3.89 | ppl    48.950
| epoch   2 step    66800 |  23788 batches | lr 0.000187 | ms/batch 496.86 | loss  3.85 | ppl    46.882
| epoch   2 step    67000 |  23988 batches | lr 0.000187 | ms/batch 495.38 | loss  3.87 | ppl    48.173
| epoch   2 step    67200 |  24188 batches | lr 0.000187 | ms/batch 495.30 | loss  3.89 | ppl    48.739
| epoch   2 step    67400 |  24388 batches | lr 0.000186 | ms/batch 497.35 | loss  3.91 | ppl    50.042
| epoch   2 step    67600 |  24588 batches | lr 0.000186 | ms/batch 495.12 | loss  3.86 | ppl    47.671
| epoch   2 step    67800 |  24788 batches | lr 0.000186 | ms/batch 495.23 | loss  3.79 | ppl    44.108
| epoch   2 step    68000 |  24988 batches | lr 0.000185 | ms/batch 495.67 | loss  3.83 | ppl    46.003
----------------------------------------------------------------------------------------------------
| Eval  17 at step    68000 | time: 1994.89s | valid loss  3.67 | valid ppl    39.391
----------------------------------------------------------------------------------------------------
| epoch   2 step    68200 |  25188 batches | lr 0.000185 | ms/batch 558.31 | loss  3.90 | ppl    49.376
| epoch   2 step    68400 |  25388 batches | lr 0.000185 | ms/batch 494.18 | loss  3.86 | ppl    47.690
| epoch   2 step    68600 |  25588 batches | lr 0.000184 | ms/batch 495.95 | loss  3.90 | ppl    49.344
| epoch   2 step    68800 |  25788 batches | lr 0.000184 | ms/batch 495.95 | loss  3.81 | ppl    45.296
| epoch   2 step    69000 |  25988 batches | lr 0.000183 | ms/batch 495.79 | loss  3.83 | ppl    45.854
| epoch   2 step    69200 |  26188 batches | lr 0.000183 | ms/batch 495.66 | loss  3.87 | ppl    47.957
| epoch   2 step    69400 |  26388 batches | lr 0.000183 | ms/batch 496.65 | loss  3.84 | ppl    46.513
| epoch   2 step    69600 |  26588 batches | lr 0.000182 | ms/batch 497.01 | loss  3.87 | ppl    48.017
| epoch   2 step    69800 |  26788 batches | lr 0.000182 | ms/batch 495.70 | loss  3.83 | ppl    46.210
| epoch   2 step    70000 |  26988 batches | lr 0.000182 | ms/batch 495.46 | loss  3.84 | ppl    46.512
| epoch   2 step    70200 |  27188 batches | lr 0.000181 | ms/batch 495.48 | loss  3.75 | ppl    42.498
| epoch   2 step    70400 |  27388 batches | lr 0.000181 | ms/batch 496.41 | loss  3.83 | ppl    45.953
| epoch   2 step    70600 |  27588 batches | lr 0.000181 | ms/batch 495.71 | loss  3.83 | ppl    45.834
| epoch   2 step    70800 |  27788 batches | lr 0.00018 | ms/batch 496.09 | loss  3.85 | ppl    46.892
| epoch   2 step    71000 |  27988 batches | lr 0.00018 | ms/batch 495.28 | loss  3.85 | ppl    46.962
| epoch   2 step    71200 |  28188 batches | lr 0.00018 | ms/batch 496.89 | loss  3.93 | ppl    51.107
| epoch   2 step    71400 |  28388 batches | lr 0.000179 | ms/batch 496.23 | loss  3.88 | ppl    48.250
| epoch   2 step    71600 |  28588 batches | lr 0.000179 | ms/batch 497.05 | loss  3.87 | ppl    47.972
| epoch   2 step    71800 |  28788 batches | lr 0.000179 | ms/batch 496.56 | loss  3.83 | ppl    46.151
| epoch   2 step    72000 |  28988 batches | lr 0.000178 | ms/batch 496.29 | loss  3.81 | ppl    45.268
----------------------------------------------------------------------------------------------------
| Eval  18 at step    72000 | time: 1994.51s | valid loss  3.65 | valid ppl    38.435
----------------------------------------------------------------------------------------------------
| epoch   2 step    72200 |  29188 batches | lr 0.000178 | ms/batch 558.09 | loss  3.81 | ppl    44.979
| epoch   2 step    72400 |  29388 batches | lr 0.000178 | ms/batch 496.79 | loss  3.88 | ppl    48.647
| epoch   2 step    72600 |  29588 batches | lr 0.000177 | ms/batch 495.04 | loss  3.82 | ppl    45.625
| epoch   2 step    72800 |  29788 batches | lr 0.000177 | ms/batch 497.27 | loss  3.86 | ppl    47.306
| epoch   2 step    73000 |  29988 batches | lr 0.000176 | ms/batch 496.16 | loss  3.93 | ppl    50.787
| epoch   2 step    73200 |  30188 batches | lr 0.000176 | ms/batch 496.80 | loss  3.90 | ppl    49.601
| epoch   2 step    73400 |  30388 batches | lr 0.000176 | ms/batch 494.70 | loss  3.80 | ppl    44.753
| epoch   2 step    73600 |  30588 batches | lr 0.000175 | ms/batch 497.47 | loss  3.90 | ppl    49.536
| epoch   2 step    73800 |  30788 batches | lr 0.000175 | ms/batch 497.58 | loss  3.86 | ppl    47.410
| epoch   2 step    74000 |  30988 batches | lr 0.000175 | ms/batch 496.34 | loss  3.85 | ppl    47.135
| epoch   2 step    74200 |  31188 batches | lr 0.000174 | ms/batch 497.49 | loss  3.78 | ppl    43.905
| epoch   2 step    74400 |  31388 batches | lr 0.000174 | ms/batch 497.79 | loss  3.91 | ppl    49.900
| epoch   2 step    74600 |  31588 batches | lr 0.000174 | ms/batch 495.27 | loss  3.84 | ppl    46.505
| epoch   2 step    74800 |  31788 batches | lr 0.000173 | ms/batch 497.41 | loss  3.86 | ppl    47.500
| epoch   2 step    75000 |  31988 batches | lr 0.000173 | ms/batch 496.86 | loss  3.83 | ppl    46.289
| epoch   2 step    75200 |  32188 batches | lr 0.000172 | ms/batch 496.06 | loss  3.83 | ppl    46.287
| epoch   2 step    75400 |  32388 batches | lr 0.000172 | ms/batch 496.56 | loss  3.82 | ppl    45.578
| epoch   2 step    75600 |  32588 batches | lr 0.000172 | ms/batch 495.09 | loss  3.78 | ppl    43.878
| epoch   2 step    75800 |  32788 batches | lr 0.000171 | ms/batch 493.97 | loss  3.72 | ppl    41.282
| epoch   2 step    76000 |  32988 batches | lr 0.000171 | ms/batch 497.01 | loss  3.81 | ppl    45.076
----------------------------------------------------------------------------------------------------
| Eval  19 at step    76000 | time: 1995.86s | valid loss  3.64 | valid ppl    37.967
----------------------------------------------------------------------------------------------------
| epoch   2 step    76200 |  33188 batches | lr 0.000171 | ms/batch 559.47 | loss  3.87 | ppl    47.730
| epoch   2 step    76400 |  33388 batches | lr 0.00017 | ms/batch 494.65 | loss  3.76 | ppl    43.043
| epoch   2 step    76600 |  33588 batches | lr 0.00017 | ms/batch 495.93 | loss  3.82 | ppl    45.561
| epoch   2 step    76800 |  33788 batches | lr 0.00017 | ms/batch 495.31 | loss  3.83 | ppl    46.026
| epoch   2 step    77000 |  33988 batches | lr 0.000169 | ms/batch 496.22 | loss  3.83 | ppl    46.269
| epoch   2 step    77200 |  34188 batches | lr 0.000169 | ms/batch 496.25 | loss  3.88 | ppl    48.310
| epoch   2 step    77400 |  34388 batches | lr 0.000168 | ms/batch 496.43 | loss  3.85 | ppl    46.997
| epoch   2 step    77600 |  34588 batches | lr 0.000168 | ms/batch 496.44 | loss  3.82 | ppl    45.519
| epoch   2 step    77800 |  34788 batches | lr 0.000168 | ms/batch 494.99 | loss  3.71 | ppl    40.734
| epoch   2 step    78000 |  34988 batches | lr 0.000167 | ms/batch 495.23 | loss  3.77 | ppl    43.317
| epoch   2 step    78200 |  35188 batches | lr 0.000167 | ms/batch 497.83 | loss  3.82 | ppl    45.826
| epoch   2 step    78400 |  35388 batches | lr 0.000167 | ms/batch 498.12 | loss  3.86 | ppl    47.646
| epoch   2 step    78600 |  35588 batches | lr 0.000166 | ms/batch 496.14 | loss  3.81 | ppl    45.088
| epoch   2 step    78800 |  35788 batches | lr 0.000166 | ms/batch 495.79 | loss  3.84 | ppl    46.738
| epoch   2 step    79000 |  35988 batches | lr 0.000165 | ms/batch 495.83 | loss  3.77 | ppl    43.227
| epoch   2 step    79200 |  36188 batches | lr 0.000165 | ms/batch 495.80 | loss  3.85 | ppl    47.068
| epoch   2 step    79400 |  36388 batches | lr 0.000165 | ms/batch 495.60 | loss  3.84 | ppl    46.464
| epoch   2 step    79600 |  36588 batches | lr 0.000164 | ms/batch 497.03 | loss  3.82 | ppl    45.647
| epoch   2 step    79800 |  36788 batches | lr 0.000164 | ms/batch 496.96 | loss  3.85 | ppl    47.228
| epoch   2 step    80000 |  36988 batches | lr 0.000164 | ms/batch 495.58 | loss  3.78 | ppl    43.972
----------------------------------------------------------------------------------------------------
| Eval  20 at step    80000 | time: 1994.91s | valid loss  3.62 | valid ppl    37.500
----------------------------------------------------------------------------------------------------
| epoch   2 step    80200 |  37188 batches | lr 0.000163 | ms/batch 557.73 | loss  3.76 | ppl    42.794
| epoch   2 step    80400 |  37388 batches | lr 0.000163 | ms/batch 496.61 | loss  3.83 | ppl    45.896
| epoch   2 step    80600 |  37588 batches | lr 0.000163 | ms/batch 497.21 | loss  3.84 | ppl    46.540
| epoch   2 step    80800 |  37788 batches | lr 0.000162 | ms/batch 495.64 | loss  3.84 | ppl    46.398
| epoch   2 step    81000 |  37988 batches | lr 0.000162 | ms/batch 495.25 | loss  3.76 | ppl    42.921
| epoch   2 step    81200 |  38188 batches | lr 0.000161 | ms/batch 495.68 | loss  3.82 | ppl    45.510
| epoch   2 step    81400 |  38388 batches | lr 0.000161 | ms/batch 495.53 | loss  3.83 | ppl    46.083
| epoch   2 step    81600 |  38588 batches | lr 0.000161 | ms/batch 496.25 | loss  3.83 | ppl    46.291
| epoch   2 step    81800 |  38788 batches | lr 0.00016 | ms/batch 496.61 | loss  3.82 | ppl    45.570
| epoch   2 step    82000 |  38988 batches | lr 0.00016 | ms/batch 496.02 | loss  3.80 | ppl    44.920
| epoch   2 step    82200 |  39188 batches | lr 0.000159 | ms/batch 495.76 | loss  3.84 | ppl    46.452
| epoch   2 step    82400 |  39388 batches | lr 0.000159 | ms/batch 496.01 | loss  3.80 | ppl    44.545
| epoch   2 step    82600 |  39588 batches | lr 0.000159 | ms/batch 496.40 | loss  3.72 | ppl    41.278
| epoch   2 step    82800 |  39788 batches | lr 0.000158 | ms/batch 495.66 | loss  3.76 | ppl    43.083
| epoch   2 step    83000 |  39988 batches | lr 0.000158 | ms/batch 496.01 | loss  3.72 | ppl    41.378
| epoch   2 step    83200 |  40188 batches | lr 0.000158 | ms/batch 496.24 | loss  3.84 | ppl    46.708
| epoch   2 step    83400 |  40388 batches | lr 0.000157 | ms/batch 497.08 | loss  3.82 | ppl    45.394
| epoch   2 step    83600 |  40588 batches | lr 0.000157 | ms/batch 496.06 | loss  3.78 | ppl    44.016
| epoch   2 step    83800 |  40788 batches | lr 0.000156 | ms/batch 494.97 | loss  3.80 | ppl    44.576
| epoch   2 step    84000 |  40988 batches | lr 0.000156 | ms/batch 496.15 | loss  3.83 | ppl    46.051
----------------------------------------------------------------------------------------------------
| Eval  21 at step    84000 | time: 1994.58s | valid loss  3.60 | valid ppl    36.595
----------------------------------------------------------------------------------------------------
| epoch   2 step    84200 |  41188 batches | lr 0.000156 | ms/batch 559.50 | loss  3.82 | ppl    45.687
| epoch   2 step    84400 |  41388 batches | lr 0.000155 | ms/batch 497.53 | loss  3.85 | ppl    46.897
| epoch   2 step    84600 |  41588 batches | lr 0.000155 | ms/batch 496.57 | loss  3.84 | ppl    46.431
| epoch   2 step    84800 |  41788 batches | lr 0.000155 | ms/batch 497.04 | loss  3.81 | ppl    45.316
| epoch   2 step    85000 |  41988 batches | lr 0.000154 | ms/batch 496.63 | loss  3.78 | ppl    43.897
| epoch   2 step    85200 |  42188 batches | lr 0.000154 | ms/batch 496.84 | loss  3.83 | ppl    46.270
| epoch   2 step    85400 |  42388 batches | lr 0.000153 | ms/batch 496.37 | loss  3.76 | ppl    42.737
| epoch   2 step    85600 |  42588 batches | lr 0.000153 | ms/batch 496.77 | loss  3.83 | ppl    46.015
| epoch   2 step    85800 |  42788 batches | lr 0.000153 | ms/batch 496.07 | loss  3.82 | ppl    45.791
| epoch   2 step    86000 |  42988 batches | lr 0.000152 | ms/batch 496.34 | loss  3.75 | ppl    42.588
| epoch   3 step    86200 |    176 batches | lr 0.000152 | ms/batch 493.99 | loss  3.73 | ppl    41.515
| epoch   3 step    86400 |    376 batches | lr 0.000152 | ms/batch 495.98 | loss  3.81 | ppl    45.185
| epoch   3 step    86600 |    576 batches | lr 0.000151 | ms/batch 495.77 | loss  3.74 | ppl    42.030
| epoch   3 step    86800 |    776 batches | lr 0.000151 | ms/batch 496.16 | loss  3.77 | ppl    43.477
| epoch   3 step    87000 |    976 batches | lr 0.00015 | ms/batch 496.08 | loss  3.77 | ppl    43.371
| epoch   3 step    87200 |   1176 batches | lr 0.00015 | ms/batch 496.94 | loss  3.77 | ppl    43.194
| epoch   3 step    87400 |   1376 batches | lr 0.00015 | ms/batch 494.88 | loss  3.75 | ppl    42.443
| epoch   3 step    87600 |   1576 batches | lr 0.000149 | ms/batch 494.32 | loss  3.75 | ppl    42.433
| epoch   3 step    87800 |   1776 batches | lr 0.000149 | ms/batch 495.20 | loss  3.74 | ppl    42.272
| epoch   3 step    88000 |   1976 batches | lr 0.000148 | ms/batch 495.05 | loss  3.63 | ppl    37.799
----------------------------------------------------------------------------------------------------
| Eval  22 at step    88000 | time: 1994.60s | valid loss  3.58 | valid ppl    35.821
----------------------------------------------------------------------------------------------------
| epoch   3 step    88200 |   2176 batches | lr 0.000148 | ms/batch 559.31 | loss  3.80 | ppl    44.774
| epoch   3 step    88400 |   2376 batches | lr 0.000148 | ms/batch 495.26 | loss  3.73 | ppl    41.521
| epoch   3 step    88600 |   2576 batches | lr 0.000147 | ms/batch 496.26 | loss  3.75 | ppl    42.511
| epoch   3 step    88800 |   2776 batches | lr 0.000147 | ms/batch 496.64 | loss  3.81 | ppl    45.369
| epoch   3 step    89000 |   2976 batches | lr 0.000146 | ms/batch 496.30 | loss  3.79 | ppl    44.406
| epoch   3 step    89200 |   3176 batches | lr 0.000146 | ms/batch 496.92 | loss  3.80 | ppl    44.569
| epoch   3 step    89400 |   3376 batches | lr 0.000146 | ms/batch 497.01 | loss  3.72 | ppl    41.090
| epoch   3 step    89600 |   3576 batches | lr 0.000145 | ms/batch 495.41 | loss  3.69 | ppl    39.899
| epoch   3 step    89800 |   3776 batches | lr 0.000145 | ms/batch 496.24 | loss  3.77 | ppl    43.355
| epoch   3 step    90000 |   3976 batches | lr 0.000145 | ms/batch 494.38 | loss  3.72 | ppl    41.354
| epoch   3 step    90200 |   4176 batches | lr 0.000144 | ms/batch 494.91 | loss  3.75 | ppl    42.478
| epoch   3 step    90400 |   4376 batches | lr 0.000144 | ms/batch 495.14 | loss  3.69 | ppl    40.209
| epoch   3 step    90600 |   4576 batches | lr 0.000143 | ms/batch 495.99 | loss  3.77 | ppl    43.557
| epoch   3 step    90800 |   4776 batches | lr 0.000143 | ms/batch 496.55 | loss  3.77 | ppl    43.368
| epoch   3 step    91000 |   4976 batches | lr 0.000143 | ms/batch 495.50 | loss  3.77 | ppl    43.277
| epoch   3 step    91200 |   5176 batches | lr 0.000142 | ms/batch 496.48 | loss  3.70 | ppl    40.385
| epoch   3 step    91400 |   5376 batches | lr 0.000142 | ms/batch 496.20 | loss  3.79 | ppl    44.400
| epoch   3 step    91600 |   5576 batches | lr 0.000141 | ms/batch 497.60 | loss  3.78 | ppl    43.714
| epoch   3 step    91800 |   5776 batches | lr 0.000141 | ms/batch 497.35 | loss  3.78 | ppl    43.996
| epoch   3 step    92000 |   5976 batches | lr 0.000141 | ms/batch 495.02 | loss  3.71 | ppl    40.849
----------------------------------------------------------------------------------------------------
| Eval  23 at step    92000 | time: 1994.78s | valid loss  3.56 | valid ppl    35.327
----------------------------------------------------------------------------------------------------
| epoch   3 step    92200 |   6176 batches | lr 0.00014 | ms/batch 559.91 | loss  3.81 | ppl    45.102
| epoch   3 step    92400 |   6376 batches | lr 0.00014 | ms/batch 496.58 | loss  3.77 | ppl    43.505
| epoch   3 step    92600 |   6576 batches | lr 0.000139 | ms/batch 496.09 | loss  3.73 | ppl    41.623
| epoch   3 step    92800 |   6776 batches | lr 0.000139 | ms/batch 495.74 | loss  3.75 | ppl    42.649
| epoch   3 step    93000 |   6976 batches | lr 0.000139 | ms/batch 496.82 | loss  3.76 | ppl    42.786
| epoch   3 step    93200 |   7176 batches | lr 0.000138 | ms/batch 496.71 | loss  3.69 | ppl    40.153
| epoch   3 step    93400 |   7376 batches | lr 0.000138 | ms/batch 496.65 | loss  3.78 | ppl    43.863
| epoch   3 step    93600 |   7576 batches | lr 0.000138 | ms/batch 496.07 | loss  3.73 | ppl    41.884
| epoch   3 step    93800 |   7776 batches | lr 0.000137 | ms/batch 497.40 | loss  3.81 | ppl    44.955
| epoch   3 step    94000 |   7976 batches | lr 0.000137 | ms/batch 497.29 | loss  3.74 | ppl    42.176
----------------------------------------------------------------------------------------------------
Exiting from training early
====================================================================================================
| End of training | test loss  3.62 | test ppl    37.262
====================================================================================================

