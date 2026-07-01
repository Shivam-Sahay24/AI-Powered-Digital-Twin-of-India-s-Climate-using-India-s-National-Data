import time
import torch

def simulate_daily_assimilation(twin_manager, historical_data_loader, days_to_simulate=1):
    """
    Simulates the passage of time. Fetches 'new' daily IMD observations
    and forces the Digital Twin to update its state memory.
    """
    for _ in range(days_to_simulate):
        # 1. Fetch the physical observation for the "new" day
        # In production, this reads today's IMD NetCDF file.
        # For the hackathon, we pull the next day from our unseen test dataset.
        new_thermo_day, new_hydro_day = historical_data_loader.get_next_day()
        
        # 2. Assimilate into the Digital Twin (Drops oldest, appends newest)
        updated_date = twin_manager.assimilate_new_observation(new_thermo_day, new_hydro_day)
        
        # 3. Log the sync for the judges to see the "Cyber-Physical" connection
        print(f"[SYNC] IMD Observation assimilated. Twin state updated to: {updated_date.strftime('%Y-%m-%d')}")
        
        # Optional: Add a slight delay if you want to animate this in the terminal/UI
        time.sleep(0.5)
        
    return twin_manager.get_current_state()