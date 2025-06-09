# simulate_api.py

import os, re, datetime as dt, numpy as np, pandas as pd, simpy, random
from dataclasses import dataclass, field
from typing import List, Tuple
from catboost import CatBoostRegressor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── 0) File paths ───────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
TRAVEL_XLSX = os.path.join(BASE_DIR, "Travel Times & Dwell Times.xlsx")
ARRIVAL_XLS = os.path.join(BASE_DIR, "passenger_arrival_rates_by_stop.xlsx")
DEST_PATH   = os.path.join(BASE_DIR, "DestinationProbabilities_Corrected.xlsx")
MODEL_PATH  = os.path.join(BASE_DIR, "segment_model.cbm")

# ─── 1) Build travel + dwell DataFrame ───────────────────────────────
def build_stops() -> pd.DataFrame:
    df = pd.read_excel(TRAVEL_XLSX, header=1)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip()
    trav = df[["From","To","KM","Duration (sec)"]].rename(
        columns={"From":"stop","To":"next","KM":"km","Duration (sec)":"travel_min"})
    dcol = next(c for c in df.columns if c.endswith("(sec).1"))
    dwell = df[["Station", dcol]].rename(columns={"Station":"stop", dcol:"dwell_sec"})
    return trav.merge(dwell, on="stop").dropna().reset_index(drop=True)

SEG_DF = build_stops()

# ─── 2) Passenger arrival map ────────────────────────────────────────
arrival_sheets = pd.read_excel(ARRIVAL_XLS, sheet_name=None)
time_re = re.compile(r"^\d{2}:\d{2}-\d{2}:\d{2}$")
ARRIVAL_MAP = {}
for name, df in arrival_sheets.items():
    df = df.set_index(df.columns[0])
    cols = [c for c in df.columns if time_re.match(c)]
    ARRIVAL_MAP[name] = df[cols]

DAY_MAP = {
    "Monday":"Pazartesi","Tuesday":"Salı","Wednesday":"Çarşamba",
    "Thursday":"Perşembe","Friday":"Cuma","Saturday":"Cumartesi","Sunday":"Pazar"
}
ARR_SCALE = 600.0

def get_lambda(stop: str, now: dt.datetime) -> float:
    df = ARRIVAL_MAP.get(stop)
    if df is None: return 0.0
    day = DAY_MAP[now.strftime("%A")]
    if day not in df.index: return 0.0
    row = df.loc[day]
    m = now.hour*60 + now.minute
    for iv,val in row.items():
        s,e = iv.split("-")
        smin = int(s[:2])*60 + int(s[3:])
        emin = int(e[:2])*60 + int(e[3:])
        if (smin <= m < emin) or (smin>emin and (m>=smin or m<emin)):
            return float(val)
    return float(row.iloc[-1])

# ─── 3) Destination probabilities ─────────────────────────────────────
dp_raw = pd.read_excel(DEST_PATH, index_col=0)
dp     = dp_raw.div(dp_raw.sum(axis=1), axis=0).fillna(0)
DEST_MAP = {o:(list(dp.columns), dp.loc[o].tolist()) for o in dp.index}

# ─── 4) CatBoost + feature list ──────────────────────────────────────
CAT = CatBoostRegressor()
CAT.load_model(MODEL_PATH)

FEATS = [
 'weather_temp','HOUR',
 'DAY_OF_WEEK_1','DAY_OF_WEEK_2','DAY_OF_WEEK_3',
 'DAY_OF_WEEK_4','DAY_OF_WEEK_5','DAY_OF_WEEK_6',
 'HOLIDAY_CATEGORY_Normal','HOLIDAY_CATEGORY_Holiday',
 'MONTH_1','MONTH_2','MONTH_3','MONTH_4','MONTH_5','MONTH_6',
 'MONTH_7','MONTH_8','MONTH_9','MONTH_10','MONTH_11','MONTH_12',
 'PANDEMIC_CONDITION_Pandemic','SCHOOL_STATUS_School Open',
 'weather_description_Cloudy','weather_description_Low Visibility',
 'weather_description_Precipitation','weather_description_Storm',
 'HATSURESI_LAG_1','HATSURESI_LAG_2','HATSURESI_LAG_3',
 'HATSURESI_LAG_4','HATSURESI_LAG_5'
]

def make_feats(now: dt.datetime, sc: "Scenario", km: float, lags: List[float]) -> List[float]:
    f = {k:0 for k in FEATS}
    f["HOUR"] = now.hour
    f["weather_temp"] = sc.temp
    f[f"DAY_OF_WEEK_{now.weekday()+1}"] = 1
    f[f"MONTH_{now.month}"] = 1
    f["HOLIDAY_CATEGORY_Holiday"] = int(sc.is_public_holiday)
    f["HOLIDAY_CATEGORY_Normal"]  = int(not sc.is_public_holiday)
    f["SCHOOL_STATUS_School Open"]   = int(sc.is_school_day)
    f["PANDEMIC_CONDITION_Pandemic"] = int(sc.is_pandemic)
    wd = f"weather_description_{sc.weather_desc}"
    if wd in f: f[wd] = 1
    lags = (lags + [0]*5)[:5]
    for i,v in enumerate(lags,1):
        f[f"HATSURESI_LAG_{i}"] = v
    return [f[c] for c in FEATS]

# ─── 5) BusType & Scenario ───────────────────────────────────────────
@dataclass(frozen=True)
class BusType:
    name:     str
    capacity: int

STD   = BusType("Standard", 90)
ARTIC = BusType("Körüklü", 120)

@dataclass
class Scenario:
    weather_desc:      str
    temp:              float
    demand_multiplier: float
    is_school_day:     bool
    is_public_holiday: bool
    is_pandemic:       bool               = False
    bus_type:          BusType            = STD    # ← default added

# ─── 6) Run ALL slots dynamically ────────────────────────────────────
def run_dynamic(sc: Scenario, start: str, end: str) -> pd.DataFrame:
    print(f"Simülasyon başlıyor: {start} - {end}")
    # compute baseline avg trip time
    base_times = []
    cur = dt.datetime.combine(dt.date.today(), pd.to_datetime(start).time())
    end_dt =    dt.datetime.combine(dt.date.today(), pd.to_datetime(end).time())
    while cur <= end_dt:
        # quick one‐segment estimate to build “base”
        base_times.append( float(CAT.predict(
            pd.DataFrame([make_feats(cur, sc, SEG_DF.km.iloc[0], [0]*5)], columns=FEATS)
        )[0]) / 60 )
        cur += dt.timedelta(minutes=30)
    base = sum(base_times)/len(base_times)
    print(f"Base ortalama trip time: {base:.2f} dk")

    # now run full sim on rolling slots
    LOG = []
    dep = dt.datetime.combine(dt.date.today(), pd.to_datetime(start).time())
    headway = 30
    while dep <= end_dt:
        # choose bus type by expected load
        exp = sum(get_lambda(r.stop, dep)*ARR_SCALE*sc.demand_multiplier * 
                  (r.travel_min/60) for r in SEG_DF.itertuples())

        sc.bus_type = ARTIC if exp > STD.capacity*0.8 else STD

        # spawn one Bus
        def bus_proc(env, name, dep, sc, headway):
            curr = dep
            lags = [0]*5
            pax  = []
            max_occ = 0
            boarded = 0
            trip_time = 0.0
            # depart delay
            start_delay = (dep - dt.datetime.combine(dep.date(), dt.time())).seconds/60
            yield env.timeout(start_delay)

            for r in SEG_DF.itertuples():
                now, stop, km = curr, r.stop, r.km
                # travel
                df_feat = pd.DataFrame([make_feats(now, sc, km, lags)], columns=FEATS)
                sec = float(CAT.predict(df_feat)[0]) * random.uniform(0.9,1.1)
                dur = sec/60
                lags = [dur] + lags[:4]
                trip_time += dur
                yield env.timeout(dur)
                curr += dt.timedelta(minutes=dur)

                # alight
                out = [p for p in pax if p==stop]
                pax = [p for p in pax if p!=stop]

                # board
                lam = get_lambda(stop, now)*ARR_SCALE*sc.demand_multiplier
                if sc.is_public_holiday: lam *= 0.8
                nin = np.random.poisson(lam*dur)
                dests,probs = DEST_MAP.get(stop, ([],[]))
                new = (np.random.choice(dests, size=nin, p=probs).tolist()
                       if dests else [stop]*nin)
                pax += new
                boarded += nin
                max_occ = max(max_occ, len(pax))

                # dwell
                dwell = (r.dwell_sec/60) + len(out)*0.03 + nin*0.01
                if sc.is_school_day: dwell *= 1.1
                if sc.is_public_holiday: dwell *= 1.2
                trip_time += dwell
                yield env.timeout(dwell)
                curr += dt.timedelta(minutes=dwell)

            LOG.append({
                "depart_time": dep.strftime("%H:%M"),
                "bus_type":    sc.bus_type.name,
                "capacity":    sc.bus_type.capacity,
                "trip_time":   round(trip_time,2),
                "max_occ":     max_occ*3,     # ×3 as requested
                "boarded":     boarded*3,     # ×3 as requested
                "headway":     headway
            })

        env = simpy.Environment()
        env.process(bus_proc(env, "Trip", dep, sc, headway))
        env.run()

        # compute load%
        rec = LOG[-1]
        rec["load_%"] = round(100 * rec["max_occ"] / rec["capacity"], 2)

        # adjust headway relative to how far trip_time deviates from base
        diff = rec["trip_time"] - base
        if   diff > 30: headway = 10
        elif diff > 20: headway = 15
        elif diff > 10: headway = 20
        else:           headway = 30

        dep += dt.timedelta(minutes=headway)

    df = pd.DataFrame(LOG)
    print(f"Simülasyon tamamlandı, {len(df)} kayıt döndü.")
    return df

# ─── 7) FastAPI setup ────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimRequest(BaseModel):
    weather_desc:      str
    temp:              float
    demand_multiplier: float
    is_school_day:     bool
    is_public_holiday: bool
    is_pandemic:       bool   = False
    start:             str
    end:               str

@app.post("/simulate")
def simulate(req: SimRequest):
    print(f"[SIM REQUEST] {req}")
    try:
        sc = Scenario(
            weather_desc=req.weather_desc,
            temp=req.temp,
            demand_multiplier=req.demand_multiplier,
            is_school_day=req.is_school_day,
            is_public_holiday=req.is_public_holiday,
            is_pandemic=req.is_pandemic,
            bus_type=STD
        )
        df = run_dynamic(sc, req.start, req.end)
        print(f"[SIM RESULT] Üretilen kayıt sayısı: {len(df)}")
        if df.empty:
            print("⚠️ Uyarı: Simülasyondan hiç kayıt gelmedi!")
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"[SIM ERROR] {e}")
        raise
