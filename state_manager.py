import torch
import datetime

class DigitalTwinStateManager:
    def __init__(self, initial_thermo_tensor, initial_hydro_tensor, current_date):
        """
        Initializes the "Living Memory" of the Digital Twin.
        Expected tensor shapes: [Batch(1), Time(T), Channels, Lat, Lon]
        """
        # Ground Truth State (The real world)
        self.thermo_state = initial_thermo_tensor.clone().detach() # Tmax, Tmin
        self.hydro_state = initial_hydro_tensor.clone().detach()   # Rainfall
        
        # Track the current timeline
        self.current_date = datetime.datetime.strptime(current_date, "%Y-%m-%d")
        self.lookback_window = self.thermo_state.shape[1] # e.g., 10 or 14 days

    def get_current_state(self):
        """Returns the real-world state for baseline forecasting."""
        return self.thermo_state, self.hydro_state

    def assimilate_new_observation(self, new_thermo_day, new_hydro_day):
        """
        The continuous update loop. Drops the oldest day, appends the newest day.
        new_day shapes should be: [Batch(1), Time(1), Channels, Lat, Lon]
        """
        # Slide the window: keep from index 1 to end, concatenate new day at the end
        self.thermo_state = torch.cat([self.thermo_state[:, 1:, ...], new_thermo_day], dim=1)
        self.hydro_state = torch.cat([self.hydro_state[:, 1:, ...], new_hydro_day], dim=1)
        
        # Advance the twin's internal clock
        self.current_date += datetime.timedelta(days=1)
        
        return self.current_date

    def generate_what_if_scenario(self, delta_t=0.0, delta_h=0.0):
        """
        Creates a temporary, perturbed clone of reality for the scenario engine.
        Does NOT alter the ground truth state.
        """
        # 1. Clone the real world
        scenario_thermo = self.thermo_state.clone()
        scenario_hydro = self.hydro_state.clone()
        
        # 2. Apply Perturbations (e.g., +2.0°C to Tmax/Tmin)
        if delta_t != 0.0:
            scenario_thermo += delta_t
            
        # (Optional) Apply moisture changes to Rainfall if your slider supports it
        if delta_h != 0.0:
            # Assuming channel 0 is rainfall. Prevent negative rainfall.
            scenario_hydro = torch.clamp(scenario_hydro + (scenario_hydro * delta_h), min=0.0)

        return scenario_thermo, scenario_hydro