import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats



class GroomAim2:
    def __init__(self):
        self.position_dict = {
            'delta_face': 'Face',
            'corrected_delta_face': 'Face (Corrected)',
            'delta_nose': 'Nose',
            'corrected_delta_nose': 'Nose (Corrected)',
            'nose-face': 'Nose - Face',
            'corrected_nose-face': 'Corrected Nose - Face'
        }
        pass