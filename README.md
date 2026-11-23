# UAV Survey Planner

Python notebooks for planning and simulating drone mapping missions. Compare candidate launch sites, estimate RGB and LiDAR coverage, split survey lines into battery-limited sorties, and visualize the mission from takeoff to landing.

The included example uses a DJI Matrice 400 with a Zenmuse L3 to survey Natadesleen Lake. Use it as a starting point for similar survey missions.

## Quick start

With Conda installed, run:

```bash
git clone https://github.com/joemcconkey/uav-survey-planner.git
cd uav-survey-planner
conda env create -f environment.yml
conda activate uav-survey-planner
```

Open the folder in VS Code, select `uav-survey-planner` as the notebook kernel, and run all cells in the notebook you want to use.

| File | Purpose |
|---|---|
| `rfp02_config.py` | Shared survey geometry, aircraft and sensor settings, and route and sortie planning logic |
| `RFP02_Launch_Site_Optimizer_r03.ipynb` | Compares launch sites by access time, field time, sortie count, and maximum slant distance |
| `RFP02_Mission_Simulation.ipynb` | Simulates the selected mission and exports route, coverage, and timeline plots, along with CSV results |

The mission notebook saves results in `mission_outputs/`. Set `export_video = True` to export an MP4. The Conda environment includes FFmpeg.

## Adapt it to another mission

Edit the survey boundary, candidate launch sites, aircraft and sensor settings, and operating assumptions in `rfp02_config.py`. Rerun the launch-site comparison, set the chosen site in the config, then restart the mission notebook kernel and run all cells.

The model estimates geometric coverage and mission timing under simplified flight and terrain assumptions. Field planning still requires checks for terrain and canopy clearance, command-link performance, and achievable survey data quality.