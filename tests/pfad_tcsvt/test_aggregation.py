"""Guard independent-unit aggregation and F1 direction in scientific reporting."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location("tcsvt_summary",ROOT / "scripts/audits/pfad_tcsvt/summarize.py")
SUMMARY=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def test_replicates_and_anatomies_have_equal_weight_within_specimens():
    cases={}
    for specimen in SUMMARY.SPECIMENS:
        for anatomy,values in zip(SUMMARY.ANATOMIES,[[1,3,5],[9,9,9],[6,6,6]],strict=True):
            cases[(specimen,anatomy)]={"conditions":[
                {"condition":{"family":"pose","level":1,"replicate":r},
                 "metrics":{"pfad":{"chamfer_l1_mm":value,"points":1000000 if anatomy=="foot" else 1}}}
                for r,value in enumerate(values)]}
    values=SUMMARY.specimen_vector(cases,"pose",1,"pfad","chamfer_l1_mm")
    np.testing.assert_array_equal(values,np.full(14,6.0))


def test_fscore_is_higher_better_despite_mm_in_its_name():
    assert SUMMARY.M.POINTSET_METRIC_DIRECTIONS["fscore_1mm"]=="higher"
    stats=SUMMARY.M.paired_endpoint_statistics(np.full(14,.7),np.full(14,.8),
        direction=SUMMARY.M.POINTSET_METRIC_DIRECTIONS["fscore_1mm"],bootstrap_seed=7,bootstrap_resamples=100)
    assert stats["wins"]==14
    assert stats["improvement"]["mean"]>0
