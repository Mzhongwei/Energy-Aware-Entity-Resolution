"""Reusable pipeline stages.

Import stages from their defining modules so embedding-only processes do not load
the unrelated BERT, Kafka, or evaluation dependency stacks.
"""
