"""Provisioning-pipeline packaging for dao-ai.

Ships the Lakeflow ``deploy_job`` assets (the ``databricks.yaml`` template and
the step notebooks) as package data so ``dao-ai generate-workflow`` can stage and submit
the job from an installed wheel, with no source checkout. See
:func:`dao_ai.pipeline.bundle.write_pipeline_bundle`.
"""
