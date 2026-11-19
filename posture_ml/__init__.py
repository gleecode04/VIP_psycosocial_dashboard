"""Posture forecasting ML pipeline.

Turns raw Y_Pitch angle readings (posture_cleaned CSV) into 2-second windows
and trains a classifier that predicts whether the *next* 2-second window will
be bad posture, following "The Pathway - Posture Machine Learning" doc.
"""
