# v10 (archived)

This folder keeps the **v10** pipeline speculation implementation (`shallow_hidden_layer_indices`, `(n+1)·S` training layout, `g_0`-only query).

Run scripts from the parent `speculative_pipeline_decoding/` directory so `pipeline_linear_cache` resolves correctly, e.g.:

```bash
cd speculative_pipeline_decoding
python old_version_v10/train.py ...
python old_version_v10/pipeline_inference.py ...
```

v11 checkpoints use the top-level `train.py` / `pipeline_model.py`. v10 checkpoints (`config['version'] == 10`) can still be loaded by the main `pipeline_inference.py` and `eval.py` via dynamic import of this module.
