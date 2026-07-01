import torch
import datetime

class DigitalTwinStateManager:
    def __init__(self, initial_thermo_tensor, initial_hydro_tensor, current_date):
        self.thermo_state = initial_thermo_tensor.clone().detach()
        self.hydro_state = initial_hydro_tensor.clone().detach()
        self.current_date = datetime.datetime.strptime(current_date, "%Y-%m-%d")
        
        # NEW: Store the last prediction to compute the error tomorrow
        # Initialize with the last day of the starting memory
        self.last_hydro_prediction = self.hydro_state[:, -1:, ...].clone() 

    def get_current_state(self):
        return self.thermo_state, self.hydro_state

    def set_last_prediction(self, prediction_tensor):
        """Saves today's forecast so it can be evaluated tomorrow."""
        self.last_hydro_prediction = prediction_tensor.clone().detach()

    def assimilate_and_correct(self, new_thermo_day, actual_hydro_day, kalman_gain=0.3):
        """
        The True Digital Twin Feedback Loop (Data Assimilation).
        Blends the model's prediction with actual reality to create a Corrected State.
        """
        # 1. Calculate the Discrepancy (Error Map)
        # Error = Reality - What we predicted
        error_map = actual_hydro_day - self.last_hydro_prediction
        
        # 2. Compute the Corrected State (The "Analysis")
        # Nudges the twin's memory using the error residual
        corrected_hydro_day = self.last_hydro_prediction + (kalman_gain * error_map)
        
        # Prevent negative rainfall due to extreme negative errors
        corrected_hydro_day = torch.clamp(corrected_hydro_day, min=0.0)

        # 3. Update the Rolling Memory with the CORRECTED state, not just raw data
        self.thermo_state = torch.cat([self.thermo_state[:, 1:, ...], new_thermo_day], dim=1)
        self.hydro_state = torch.cat([self.hydro_state[:, 1:, ...], corrected_hydro_day], dim=1)
        
        self.current_date += datetime.timedelta(days=1)
        
        return self.current_date, error_map, corrected_hydro_day

    # ... (Keep the generate_what_if_scenario function exactly as it was) ...