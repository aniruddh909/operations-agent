"""Evaluation harness for the triage agent.

Scores the agent's *decisions* against a labeled golden set, so prompt/model
changes can be measured for regressions rather than eyeballed. The standout
metric is confidence-gate accuracy: did the agent correctly decide to ask a
human vs. proceed?
"""
