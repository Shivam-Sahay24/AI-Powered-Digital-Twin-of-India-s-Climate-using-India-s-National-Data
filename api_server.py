from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import datetime
import numpy as np

# Import the classes we built earlier (assuming they are in these files)
from state_manager import DigitalTwinStateManager
# from your_model_file import DualStream_AttnUConvLSTM

app = FastAPI(title="CLIMATE-TWIN Backend API", version="1.0")

# --- 1. GLOBAL STATE INITIALIZATION ---
# In a real scenario, load your best .pth weights here
# model = DualStream_AttnUConvLSTM()
# model.load_state_dict(torch.load("best_model.pth"))
# model.eval() 

# Mocking initial historical data (14 days lookback) for the Hackathon MVP
# Shapes: [Batch, Time, Channels, Lat, Lon]
init_thermo = torch.rand(1, 14, 2, 64, 64) 
init_hydro = torch.rand(1, 14, 1, 64, 64)
twin = DigitalTwinStateManager(init_thermo, init_hydro, current_date="2026-07-01")

# --- 2. DATA MODELS (For the Designer's JSON Payloads) ---
class ScenarioRequest(BaseModel):
    delta_t: float = 0.0      # Temperature perturbation (e.g., +2.0)
    delta_h: float = 0.0      # Moisture perturbation percentage (e.g., -0.10 for -10%)
    target_date: str = None   # Optional specific date

# --- 3. HELPER: CONFIDENCE SCORING ---
def calculate_twin_health(predicted_tensor, actual_tensor):
    """
    Calculates the 'Confidence/Error' metric required by the judges.
    Compares what the Twin predicted yesterday vs what actually happened today.
    """
    mse = torch.nn.functional.mse_loss(predicted_tensor, actual_tensor).item()
    # Convert MSE to a human-readable 0-100% "Health Score"
    health_score = max(0.0, 100.0 - (mse * 10)) 
    return round(health_score, 2)

# --- 4. API ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "Active", "twin_date": twin.current_date.strftime("%Y-%m-%d")}

@app.post("/trigger_daily_assimilation")
def assimilate_next_day():
    """
    ENDPOINT 1: The Data Assimilation Loop.
    Executes the error-correction feedback loop.
    """
    # 1. Fetch the actual IMD physical observations (Mocked for hackathon)
    new_thermo_day = torch.rand(1, 1, 2, 64, 64)
    actual_hydro_day = torch.rand(1, 1, 1, 64, 64) # The "Truth"
    
    # 2. Execute Data Assimilation (The Feedback Loop)
    new_date, error_map, corrected_state = twin.assimilate_and_correct(
        new_thermo_day, 
        actual_hydro_day, 
        kalman_gain=0.3
    )
    
    # 3. Generate TOMORROW'S prediction and save it for the next loop
    # Mocking the model inference: next_pred = model(*twin.get_current_state())
    next_prediction = torch.rand(1, 1, 1, 64, 64) 
    twin.set_last_prediction(next_prediction) 
    
    # Calculate overall health score based on the error map's Mean Squared Error
    mse = (error_map ** 2).mean().item()
    health_score = max(0.0, 100.0 - (mse * 10))
    
    # 4. Send all matrices to the frontend for the 4-Panel Visualization
    return {
        "message": "Data Assimilation & Correction Successful",
        "new_system_date": new_date.strftime("%Y-%m-%d"),
        "twin_health_score": f"{round(health_score, 2)}%",
        # Send back the specific arrays for the UI to plot
        "maps_for_ui": {
            "predicted_yesterday": twin.last_hydro_prediction.squeeze().tolist(),
            "actual_observed_today": actual_hydro_day.squeeze().tolist(),
            "calculated_error_map": error_map.squeeze().tolist(),
            "final_corrected_state": corrected_state.squeeze().tolist()
        }
    }
@app.post("/simulate_scenario")
def run_scenario(request: ScenarioRequest):
    """
    ENDPOINT 2: The What-If Engine.
    The UI sends the slider values here.
    """
    try:
        # 1. Clone & Perturb Reality
        scen_t, scen_h = twin.generate_what_if_scenario(delta_t=request.delta_t, delta_h=request.delta_h)
        
        # 2. Run Inference (Mocking the model pass for safety)
        # baseline_pred = model(*twin.get_current_state())
        # scenario_pred = model(scen_t, scen_h)
        baseline_pred = torch.rand(64, 64) # Simulated output grid
        scenario_pred = torch.rand(64, 64) # Simulated output grid
        
        # 3. Calculate impact
        impact_matrix = scenario_pred - baseline_pred
        
        # 4. Package as JSON arrays for the Designer to map
        return {
            "baseline_grid": baseline_pred.tolist(),
            "scenario_grid": scenario_pred.tolist(),
            "impact_heatmap": impact_matrix.tolist(),
            "metrics": {
                "max_rain_increase_mm": round(impact_matrix.max().item(), 2),
                "max_rain_decrease_mm": round(impact_matrix.min().item(), 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))