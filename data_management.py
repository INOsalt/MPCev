# data_management.py
import pandas as pd

class DataManager:
    def __init__(self, weather_file, ev_file, pv_file):
        self.weather_file = weather_file
        self.ev_file = ev_file
        self.pv_file = pv_file

    def load_data(self):
        # Load data from CSV
        weather_data = pd.read_csv(self.weather_file)
        ev_data = pd.read_csv(self.ev_file)
        pv_data = pd.read_csv(self.pv_file)
        return weather_data, ev_data, pv_data