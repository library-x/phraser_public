import os

import numpy as np

from phraser.eval.datasets import map_label, load_harmonix, load_salami


def test_map_label_core_vocab():
    assert map_label("chorus") == "chorus"
    assert map_label("Verse") == "verse"
    assert map_label("silence") == "silence"


def test_map_label_strips_variant_digits():
    assert map_label("chorus2") == "chorus"
    assert map_label("verse3") == "verse"


def test_map_label_function_mapping():
    assert map_label("prechorus") == "verse"
    assert map_label("postchorus") == "chorus"
    assert map_label("solo") == "bridge"
    assert map_label("instrumental") == "bridge"


def test_map_label_unknown_falls_back_to_verse():
    assert map_label("gibberish") == "verse"


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def test_load_harmonix_fixture(tmp_path):
    root = str(tmp_path)
    _write(f"{root}/dataset/segments/0001_test.txt",
           "0.0 silence\n1.5 intro\n20.0 chorus2\n60.0 end\n")
    _write(f"{root}/dataset/beats_and_downbeats/0001_test.txt",
           "0.5\t1\t1\n1.0\t2\t1\n1.5\t3\t1\n")
    tracks = load_harmonix(root)
    assert len(tracks) == 1
    t = tracks[0]
    assert t["id"] == "0001_test"
    assert t["ref_intervals"].shape == (3, 2)
    assert t["ref_labels"] == ["silence", "intro", "chorus"]
    assert np.allclose(t["ref_beats"], [0.5, 1.0, 1.5])
    assert t["audio"] is None  # no audio dir given


def test_load_salami_fixture(tmp_path):
    root = str(tmp_path)
    _write(f"{root}/annotations/42/parsed/textfile1_functions.txt",
           "0.0\tSilence\n0.5\tIntro\n30.0\tVerse\n90.0\tEnd\n")
    tracks = load_salami(root)
    assert len(tracks) == 1
    t = tracks[0]
    assert t["id"] == "42"
    assert t["ref_intervals"].shape == (3, 2)
    assert t["ref_labels"] == ["silence", "intro", "verse"]
