Footprint Replay
================

The **Footprint Replay** plugin adds a trial timeline to the Configuration
Footprint. It uses one fixed MDS projection for the complete run, so changing
the trial slider cannot move already displayed points.

The performance surface, border configurations, and random support
configurations are calculated once after pressing **Process**. The slider then
only changes which evaluated configurations have appeared by that trial. The
red line shows the incumbent configuration sequence up to the selected trial.
Configurations that have not appeared yet can be shown in light gray.

The replay order is based on successful trials at the selected budget, sorted
by their ``end_time``. A configuration is considered visited at its first
successful trial. Runs that are still being written can be refreshed by
processing the plugin again; this intentionally avoids continuously rebuilding
the expensive MDS projection while dragging the slider.

The evaluator is also available in API mode:

.. code-block:: python

    from deepcave.evaluators.footprint_replay import build_cloud, subset

    cloud = build_cloud(run, objective, budget)
    frame = subset(cloud, trial=10)
