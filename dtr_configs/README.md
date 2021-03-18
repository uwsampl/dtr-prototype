## Pareto Curve Config

Example config for Pareto curve with non-JSON standards-compliant comments for explaining the parameters.

```
{
    "active": true,                             // whether the experiment is active
    "save_logs": false,                         // whether to save a DTR log
    "log_dest": "~/dtr_nightly_logs",           // if save_logs is true, gives the location for saving logs
    "sync_gpu": true,                           // whether to put PyTorch into "synchronous mode"
    "dtr_torch_cmd": "~/dtr_venv/bin/python3",  // location of Python executable with DTR PyTorch installed
    "dry_run": 10,                              // number of non-timed "warm up" runs
    "n_inputs": 1,                              // how many distinct inputs to take measurements for
    "n_reps": 50,                               // number of timed runs (after warm-up)
    "report_errors": true,                      // if false, stop the eval after first error; else, continue the eval and *report* the error as a result
    "set_seed": true,                           // whether to set the random seed in Python and PyTorch
    "seed": 1,                                  // seed to use
    "notify": [],                               // Slack ids to ping in case of an error
    "models": ["model1", "model2", ...],        // list of models to execute
    "dtr_settings": {                           // (slight misnomer) settings for timed trials
    
    "default": {                      // default for all models/settings, can be overwritten
        "batch_size": 32,             // batch size
        "timeout": 1200               // kill the trial if this many seconds passes
        "ignore_small_tensors": true, // have DTR ignore tensors <1% of the average size (on by default)
        "use_sampling": true          // have DTR take a sqrt(N)-size random sample of the eviction pool instead of searching over all of it
    },
    
    "model1": [                        // commands to execute in trials
        {
            "type": "baseline",       // unmodified PT (no "kind" parameter used)
            "batch_size": 56,         // overwrites default
            "extra_params": {         // model-specific parameters
            ...
            }
        },
        {
            "type": "dtr",
            "kind": "fixed",          // "fixed" command: run on exactly the below budgets
            "use_profiling": true,    // activate DTR profiling
            "memory_budget": [        // run on all of these budgets (in bytes)
                9e9, 8e9, ..., 4e9
            ],
            "extra_params": {
                "retry_on_error": true, // if true, do not halt the testing loop if an entry errors out; retry and record the number of retries
                "max_retries": 15,      // max number of retries if retry_on_error is selected
                "no_sampling_below_budget": 5e9, // if use_sampling is true, do _not_ use sampling below the given budget (inclusive)
                ...
            }
        },
        {
            "type": "dtr",
            "kind": "ratio",          // "ratio" command: run DTR using the given ratio of the baseline memory usage
                                      // (do not use simultaneously with no_sampling_below_ratio)
            "ratio": [                // ratios to use
                0.9, 0.8, ..., 0.1
            ],
            "extra_params": {
                "no_sampling_below_ratio": 5e9, // if use_sampling is true, do _not_ use sampling below the given ratio (inclusive)
                                                // (do not use simultaneously with no_sampling_below_budget)
                ....
            }
        }
    ],
    "model2": [...],
    ...
    }
}
```
