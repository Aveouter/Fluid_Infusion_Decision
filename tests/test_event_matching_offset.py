"""Frozen #81 rule: maximum cardinality, then minimum total offset."""
import itertools
import numpy as np
from utils.changepoints import match_events


def test_nearest_exact_match_wins_after_cardinality_tie():
    true=np.zeros((8,1),bool);pred=true.copy()
    true[5,0]=True;pred[[4,5],0]=True
    out=match_events(true,pred,tolerance=1)
    assert out['n_matched']==1
    assert out['matches']==[(0,5,5,0)]


def test_both_matching_objectives_against_exhaustive_oracle():
    rng=np.random.default_rng(20261015)
    for _ in range(80):
        true_ts=sorted(rng.choice(8,size=int(rng.integers(1,5)),replace=False))
        pred_ts=sorted(rng.choice(8,size=int(rng.integers(1,5)),replace=False))
        tolerance=int(rng.integers(0,3));best=(0,0)
        for size in range(1,min(len(true_ts),len(pred_ts))+1):
            for left in itertools.combinations(true_ts,size):
                for right in itertools.permutations(pred_ts,size):
                    offsets=[abs(a-b) for a,b in zip(left,right)]
                    if max(offsets)<=tolerance:best=max(best,(size,-sum(offsets)))
        true=np.zeros((8,1),bool);pred=true.copy();true[true_ts,0]=True;pred[pred_ts,0]=True
        out=match_events(true,pred,tolerance=tolerance)
        actual=(out['n_matched'],-sum(abs(m[3]) for m in out['matches']))
        assert actual==best
