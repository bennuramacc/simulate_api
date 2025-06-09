# simulate_api

**simulate\_api** is a FastAPI‑based web service that runs a detailed bus route simulation using real-world data and a trained CatBoost model. It exposes a single endpoint (`/simulate`) to generate a dynamic trip schedule, headways, vehicle types, and load estimates for a user‑specified time window and scenario parameters.

---

## 🚀 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/bennuramacc/simulate_api.git
   cd simulate_api
   ```

2. **Prepare data files**: Place the following Excel and model files in the project root:

   * `Travel Times & Dwell Times.xlsx`
   * `passenger_arrival_rates_by_stop.xlsx`
   * `DestinationProbabilities_Corrected.xlsx`
   * `segment_model.cbm`

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**:

   ```bash
   uvicorn simulate_api:app --host 0.0.0.0 --port 8000
   ```

---

## 🔧 Usage

Send a POST request to `/simulate` with scenario parameters:

```http
POST /simulate HTTP/1.1
Host: <your-domain>
Content-Type: application/json

{
  "weather_desc": "Clear",       # One of: Clear, Cloudy, Precipitation, Storm
  "temp": 16.0,                  # Ambient temperature (°C)
  "demand_multiplier": 1.3,      # Demand scaling factor (e.g. rush‑hour)
  "is_school_day": true,         # School in session?
  "is_public_holiday": false,    # Public holiday?
  "is_pandemic": false,          # Pandemic conditions?
  "start": "06:00",            # Simulation start time (HH:MM)
  "end": "23:30"               # Simulation end time (HH:MM)
}
```

Successful response (HTTP 200) returns an array of trip records:

| depart\_time | headway | bus\_type   | capacity | trip\_time | max\_occ | boarded | load\_% |
| ------------ | ------- | ----------- | -------- | ---------- | -------- | ------- | ------- |
| "06:00"      | 30      | Standard    | 90       | 87.23      | 45       | 67      | 74.44   |
| "06:30"      | 15      | Articulated | 120      | 88.12      | 78       | 102     | 65.00   |
| ...          | ...     | ...         | ...      | ...        | ...      | ...     | ...     |

Field descriptions:

* **depart\_time**: Scheduled departure of each trip.
* **headway**: Minutes until next departure.
* **bus\_type**: Vehicle assigned (Standard 90‑seat or Articulated 120‑seat).
* **capacity**: Seat capacity of the vehicle.
* **trip\_time**: Total in‑service minutes.
* **max\_occ**: Maximum onboard passengers during the trip.
* **boarded**: Total boardings over the trip.
* **load\_%**: Peak load factor = (max\_occ / capacity) × 100.

---

## ⚙️ Simulation Logic

1. **Data loading**

   * Build `SEG_DF`, a table of stops with travel times and dwell times.
   * Load arrival‐rate sheets per stop and time interval.
   * Read destination‐probability matrix for passenger boarding.

2. **Expected load estimation**

   * For each planned departure, compute an expected passenger arrival rate along segments.
   * Use this to choose Standard or Articulated vehicle according to a threshold.

3. **Detailed trip simulation**

   * Use SimPy to model each bus’s progression stop-to-stop.
   * Predict segment travel times with CatBoost, conditioned on weather, time, lags.
   * Randomize boarding (Poisson arrivals) and alighting by destination probabilities.
   * Track `max_occ`, `boarded`, and accumulate trip time.

4. **Dynamic headway adjustment**

   * Compute a baseline average trip time over the interval.
   * After each trip, compare actual trip time to baseline:

     * **> 30 min**: headway = 10 min
     * **20–30 min**: headway = 15 min
     * **10–20 min**: headway = 20 min
     * **≤ 10 min**: headway = 30 min
   * If load > 90%, override to Articulated bus and recalculate load %.

---

## 📁 Project Structure

```
simulate_api/
├── .github/               # CI / GitHub Actions workflows
├── Travel Times & Dwell Times.xlsx
├── passenger_arrival_rates_by_stop.xlsx
├── DestinationProbabilities_Corrected.xlsx
├── segment_model.cbm      # Pretrained CatBoost model
├── requirements.txt
├── simulate_api.py        # Main application
└── README.md              # This documentation
```

---

## 🤝 Contributing

Feel free to open issues or pull requests for bug fixes, improvements, or new features. All contributions are welcome!

---

*Generated automatically – please keep this README up to date with any code changes.*
