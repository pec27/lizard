"""
Test the Friends-of-Friends
"""

from lizard import fof
import numpy as np

def test_fof_5pt():

    from lizard.log import VerboseTimingLog
    log = VerboseTimingLog()

    # test with 1 isolated point, 4 next to each other (using periodicity)
    pos = np.array(((0.5, 0.2, 0.2), # isolated point
                    (0.95, 0.5, 0.4),
                    (0.95, 0.4, 0.5),
                    (0.95, 0.5, 0.6),
                    (0.05, 0.5, 0.5))) # connected by periodicity
                    
    

    labels = fof.fof_groups(pos, b=0.2, log=log)
    id_isol = labels[0]
    id_other = labels[1]
    assert(id_isol != id_other)

    
    assert((labels==(id_isol, id_other, id_other, id_other, id_other)).all())
