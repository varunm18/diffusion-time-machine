"""Date conditioning for MVGenMaster (the model we moved to on 2026-11-19; see docs/mvgenmaster_time_plan.md).

MVGenMaster is used from its own checkout (``external/MVGenMaster``, never edited) and its own
environment (``external/MVGenMaster/.venv``, with this package installed in editable mode). Code
here therefore imports ``my_diffusers`` and MVGenMaster's ``src.modules`` and only runs there;
run scripts through ``slurm/mvgen_gpu.sbatch`` and tests from inside the checkout::

    cd external/MVGenMaster && .venv/bin/python -m pytest <repo>/tests/test_mvgen_*.py -q -p no:cacheprovider

Modules:

* :mod:`~timemachine.mvgen.date`     the per-view date embedding and how it enters every ResBlock
* :mod:`~timemachine.mvgen.model`    loading the released model, adapters (date, LoRA), checkpoints
* :mod:`~timemachine.mvgen.step`     one training step: conditioning tensors and the v-prediction loss
* :mod:`~timemachine.mvgen.data`     dated view sets of a scene, with depth and co-visibility sampling
* :mod:`~timemachine.mvgen.trainer`  the training loop
* :mod:`~timemachine.mvgen.probes`   fixed evaluations: loss with true / wrong / unknown dates, renders
"""
