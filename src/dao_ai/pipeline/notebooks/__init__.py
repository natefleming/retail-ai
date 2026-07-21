"""Packaged Databricks notebooks for the dao-ai provisioning pipeline.

These ``NN_*.py`` files are Databricks *notebook source* files, not importable
modules — they are materialized into a staging bundle by
:func:`dao_ai.pipeline.bundle.write_pipeline_bundle` and executed as
``notebook_task`` steps of the Lakeflow ``deploy_job``. This ``__init__`` exists
only so the directory is discoverable via ``importlib.resources.files``.
"""
