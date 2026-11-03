import numpy as np
import pytest

from physiosight.perception.base import BLAZEPOSE_33, COCO_17
from physiosight.perception.cache import LandmarkSequence
from physiosight.testing import synth


def test_cache_roundtrip(tmp_path):
    seq = synth.render(synth.curl_script(1), fps=15)
    seq.save(tmp_path / "x.npz")
    back = LandmarkSequence.load(tmp_path / "x.npz")
    assert len(back) == len(seq) and back.meta["synthetic"]
    np.testing.assert_array_equal(back.xy, seq.xy)
    np.testing.assert_array_equal(back.t, seq.t)


def test_timestamps_must_increase():
    seq = synth.render(synth.curl_script(1), fps=15)
    with pytest.raises(ValueError):
        LandmarkSequence(seq.skeleton, seq.width, seq.height, seq.t[::-1].copy(), seq.present, seq.xy,
                         seq.visibility, seq.presence)


def test_convert_to_coco17_by_name():
    seq = synth.render(synth.squat_script(1), fps=10)
    coco = seq.to_skeleton("coco17")
    assert coco.xy.shape[1] == len(COCO_17)
    k = BLAZEPOSE_33.index("left_knee")
    np.testing.assert_array_equal(coco.xy[:, COCO_17.index("left_knee")], seq.xy[:, k])
