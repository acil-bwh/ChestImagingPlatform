import numpy as np
from cip_python.utils import compute_dice_coefficient

def test_compute_dice_coefficient():
    ref = np.zeros([3, 3, 3])
    test = np.zeros([3, 3, 3])    
    
    dice = compute_dice_coefficient(ref, test, 1)
    assert dice == 1.0, "Unexpected Dice coefficient"

    test += 1
    dice = compute_dice_coefficient(ref, test, 1)
    assert dice == 0.0, "Unexpected Dice coefficient"    

    ref += 1
    dice = compute_dice_coefficient(ref, test, 1)
    assert dice == 1.0, "Unexpected Dice coefficient"        

    ref[1, 1, 1] = 0
    dice = compute_dice_coefficient(ref, test, 1)
    assert dice == 26*2/(27+26.), "Unexpected Dice coefficient"    

    ref[1, 1, 1] = 2
    dice = compute_dice_coefficient(ref, test, 1)
    assert dice == 26*2/(27+27.), "Unexpected Dice coefficient"    
