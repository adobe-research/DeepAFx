#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
import random
import numpy as np

def lineartodB(x):
    return 20*np.log10(x) 
def dBtoLinear(x):
    return np.power(10,x/20)

kFlat = {
    15:0, # HP
    16:30,
    18:0, # LP 
    19:10000.0,  
    21:0, # LS
    22:dBtoLinear(0.0),
    23:100.0,
    24:0, # HS
    25:dBtoLinear(0.0),
    26:5000.0,
    27:0, # F1
    28:dBtoLinear(0.0),
    29:100.0,
    30:1,
    31:0, # F2
    32:dBtoLinear(0.0),
    33:500.0,
    34:1,
    35:0, # F3
    36:dBtoLinear(0.0),
    37:2000.0,
    38:1,
    39:0, # F4
    40:dBtoLinear(0.0),
    41:5000.0,
    42:1,
}

kOldTimeRadio = {
    15:0, # HP
    16:30,
    18:0, # LP 
    19:10000.0,  
    21:1, # LS
    22:dBtoLinear(-5.0),
    23:42.0,
    24:1, # HS
    25:dBtoLinear(-22.0),
    26:7000.0,
    27:1, # F1
    28:dBtoLinear(-30.0),
    29:80.0,
    30:4,
    31:1, # F2
    32:dBtoLinear(-13.0),
    33:200.0,
    34:4,
    35:1, # F3
    36:dBtoLinear(11.0),
    37:672.0,
    38:4,
    39:1, # F4
    40:dBtoLinear(-12),
    41:2032.0,
    42:4,
}

kVocalEnhancer = {
    15:1, # HP
    16:80,
    18:0, # LP 
    19:10000.0,  
    21:1, # LS
    22:dBtoLinear(6.0),
    23:110.0,
    24:1, # HS
    25:dBtoLinear(13.9),
    26:10000.0,
    27:0, # F1
    28:dBtoLinear(0.0),
    29:80.0,
    30:4,
    31:1, # F2
    32:dBtoLinear(-3.0),
    33:291.0,
    34:1.5,
    35:1, # F3
    36:dBtoLinear(0.0),
    37:3200.0,
    38:2,
    39:1, # F4
    40:dBtoLinear(0),
    41:8800.0,
    42:2,
}

kRapVocals = {
    15:1, # HP
    16:80,
    18:0, # LP 
    19:10000.0,  
    21:1, # LS
    22:dBtoLinear(6.0),
    23:110.0,
    24:1, # HS
    25:dBtoLinear(33.9),
    26:10000.0,
    27:0, # F1
    28:dBtoLinear(0.0),
    29:80.0,
    30:4,
    31:1, # F2
    32:dBtoLinear(-3.0),
    33:291.0,
    34:1.5,
    35:1, # F3
    36:dBtoLinear(5.1),
    37:1440.0,
    38:2,
    39:1, # F4
    40:dBtoLinear(0),
    41:8800.0,
    42:2,
}

kLoudnessMaximizer = {
    15:0, # HP
    16:80,
    18:0, # LP 
    19:10000.0,  
    21:1, # LS
    22:dBtoLinear(6.0),
    23:120.0,
    24:1, # HS
    25:dBtoLinear(12.0),
    26:6000.0,
    27:0, # F1
    28:dBtoLinear(0.0),
    29:80.0,
    30:4,
    31:0, # F2
    32:dBtoLinear(-3.0),
    33:291.0,
    34:1.5,
    35:0, # F3
    36:dBtoLinear(5.1),
    37:1440.0,
    38:2,
    39:0, # F4
    40:dBtoLinear(0),
    41:8800.0,
    42:2,
}

kKick = {
    15:0, # HP
    16:80,
    18:0, # LP 
    19:10000.0,  
    21:1, # LS
    22:dBtoLinear(-9.0),
    23:20.0,
    24:1, # HS
    25:dBtoLinear(-20.0),
    26:10000.0,
    27:1, # F1
    28:dBtoLinear(3.0),
    29:80.0,
    30:2,
    31:1, # F2
    32:dBtoLinear(-3.0),
    33:120.0,
    34:4,
    35:0, # F3
    36:dBtoLinear(5.1),
    37:1440.0,
    38:2,
    39:1, # F4
    40:dBtoLinear(4),
    41:2474,
    42:2,
}

kHeavyGuitar = {
    15:0, # HP
    16:80,
    18:0, # LP 
    19:10000.0,  
    21:1, # LS
    22:dBtoLinear(-15.0),
    23:20.0,
    24:1, # HS
    25:dBtoLinear(-15.0),
    26:10000.0,
    27:1, # F1
    28:dBtoLinear(-3.0),
    29:120.0,
    30:1,
    31:1, # F2
    32:dBtoLinear(4.0),
    33:454.0,
    34:2,
    35:1, # F3
    36:dBtoLinear(-3),
    37:2938.0,
    38:1,
    39:0, # F4
    40:dBtoLinear(4),
    41:2474,
    42:2,
}

kLowpass = {
    15:0, # HP
    16:30,
    18:0, # LP 
    19:10000.0,  
    21:0, # LS
    22:dBtoLinear(0.0),
    23:100.0,
    24:1, # HS
    25:dBtoLinear(-30.0),
    26:8000.0,
    27:0, # F1
    28:dBtoLinear(0.0),
    29:100.0,
    30:1,
    31:0, # F2
    32:dBtoLinear(0.0),
    33:500.0,
    34:1,
    35:0, # F3
    36:dBtoLinear(0.0),
    37:2000.0,
    38:1,
    39:0, # F4
    40:dBtoLinear(0.0),
    41:5000.0,
    42:1,
}

kHighpass = {
    15:0, # HP
    16:30,
    18:0, # LP 
    19:10000.0,  
    21:1, # LS
    22:dBtoLinear(-38.5),
    23:20.0,
    24:0, # HS
    25:dBtoLinear(-30.0),
    26:8000.0,
    27:0, # F1
    28:dBtoLinear(2.0),
    29:110.0,
    30:1,
    31:0, # F2
    32:dBtoLinear(0.0),
    33:500.0,
    34:1,
    35:0, # F3
    36:dBtoLinear(0.0),
    37:2000.0,
    38:1,
    39:0, # F4
    40:dBtoLinear(0.0),
    41:5000.0,
    42:1,
}

kBeefySnare = {
    15:0, # HP
    16:30,
    18:0, # LP 
    19:10000.0,  
    21:1, # LS
    22:dBtoLinear(-20.0),
    23:20.0,
    24:1, # HS
    25:dBtoLinear(-10.0),
    26:10000.0,
    27:1, # F1
    28:dBtoLinear(8.0),
    29:271.0,
    30:4,
    31:1, # F2
    32:dBtoLinear(-10.0),
    33:5705.0,
    34:1,
    35:0, # F3
    36:dBtoLinear(0.0),
    37:2000.0,
    38:1,
    39:0, # F4
    40:dBtoLinear(0.0),
    41:5000.0,
    42:1,
}

kAcousticGuitar = {
    15:0, # HP
    16:30,
    18:0, # LP 
    19:10000.0,  
    21:1, # LS
    22:dBtoLinear(-6.0),
    23:40.0,
    24:1, # HS
    25:dBtoLinear(2.0),
    26:9000.0,
    27:1, # F1
    28:dBtoLinear(-3.0),
    29:200.0,
    30:1,
    31:1, # F2
    32:dBtoLinear(-2.0),
    33:2000.0,
    34:1,
    35:1, # F3
    36:dBtoLinear(2.0),
    37:5000.0,
    38:1,
    39:0, # F4
    40:dBtoLinear(0.0),
    41:5000.0,
    42:1,
}

kPresets = [kFlat,
            kOldTimeRadio,
            kVocalEnhancer,
            kRapVocals,
            kLoudnessMaximizer,
            kKick,
            kHeavyGuitar,
            kLowpass,
            kHighpass,
            kBeefySnare,
            kAcousticGuitar
           ]

np.save('/home/code-base/runtime/deepafx/data/EQ_PRESETS.pkl', kPresets)