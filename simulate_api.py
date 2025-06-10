# -*- coding: utf-8 -*-
import os, re, datetime as dt, numpy as np, pandas as pd, simpy, random
from dataclasses import dataclass
from typing import Literal
from catboost import CatBoostRegressor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── 0) Dosya yolları ────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
TRAVEL_XLSX  = os.path.join(BASE_DIR, "Travel Times & Dwell Times.xlsx")
ARRIVAL_XLS  = os.path.join(BASE_DIR, "passenger_arrival_rates_by_stop.xlsx")
DEST_PATH    = os.path.join(BASE_DIR, "DestinationProbabilities_Corrected.xlsx")
MODEL_PATH   = os.path.join(BASE_DIR, "segment_model.cbm")

# ─── FastAPI & CORS ──────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 1) Travel + Dwell tablosu ──────────────────────────────────────
def build_stops() -> pd.DataFrame:
    df = pd.read_excel(TRAVEL_XLSX, header=1)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip()
    trav = df[["From","To","KM","Duration (sec)"]].rename(
        columns={"From":"stop","To":"next","KM":"km","Duration (sec)":"travel_min"})
    dcol = [c for c in df.columns if c.endswith("(sec).1")][0]
    dwell = df[["Station", dcol]].rename(
        columns={"Station":"stop", dcol:"dwell_sec"})
    return trav.merge(dwell, on="stop").dropna().reset_index(drop=True)

SEG_DF = build_stops()

# ─── 2) Arrival‐rate haritası ────────────────────────────────────────
arrival_sheets = pd.read_excel(ARRIVAL_XLS, sheet_name=None)
time_re = re.compile(r"^\d{2}:\d{2}-\d{2}:\d{2}$")
ARRIVAL_MAP = {}
for name, df in arrival_sheets.items():
    df = df.set_index(df.columns[0])
    cols = [c for c in df.columns if time_re.match(c)]
    ARRIVAL_MAP[name] = df[cols]

DAY_MAP   = {"Monday":"Pazartesi","Tuesday":"Salı","Wednesday":"Çarşamba",
             "Thursday":"Perşembe","Friday":"Cuma","Saturday":"Cumartesi","Sunday":"Pazar"}
ARR_SCALE = 600.0

def get_lambda(stop: str, now: dt.datetime) -> float:
    df = ARRIVAL_MAP.get(stop)
    if df is None: return 0.0
    day = DAY_MAP[now.strftime("%A")]
    row = df.loc[day] if day in df.index else None
    if row is None: return 0.0
    m = now.hour*60 + now.minute
    for iv,val in row.items():
        s,e = iv.split("-")
        smin = int(s[:2])*60 + int(s[3:])
        emin = int(e[:2])*60 + int(e[3:])
        if (smin<=m<emin) or (smin>emin and (m>=smin or m<emin)):
            return float(val)
    return float(row.iloc[-1])

# ─── 3) Beklenen load hesaplama ────────────────────────────────────
def estimate_expected_load(dep: dt.datetime, sc: "Scenario") -> float:
    now, tot = dep, 0.0
    for r in SEG_DF.itertuples():
        lam = get_lambda(r.stop, now) * ARR_SCALE * sc.demand_multiplier
        tot += lam * (r.travel_min/60)
        now += dt.timedelta(minutes=r.travel_min)
    return tot

# ─── 4) Destination probabilities ───────────────────────────────────
dp_raw = pd.read_excel(DEST_PATH, index_col=0)
dp     = dp_raw.div(dp_raw.sum(axis=1), axis=0).fillna(0)
DEST   = {o:(list(dp.columns), dp.loc[o].tolist()) for o in dp.index}

# ─── 5) CatBoost modeli & feature list ─────────────────────────────
CAT = CatBoostRegressor(); CAT.load_model(MODEL_PATH)
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

def make_feats(now: dt.datetime, sc: "Scenario", km: float, lags: list[float]) -> list[float]:
    f = {k:0 for k in FEATS}
    f["HOUR"] = now.hour; f["weather_temp"] = sc.temp
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

# ─── 6) BusType & Scenario ─────────────────────────────────────────
@dataclass(frozen=True)
class BusType:
    name: str; capacity: int

STD   = BusType("Standard", 90)
ARTIC = BusType("Körüklü", 120)

@dataclass
class Scenario:
    weather_desc:      str
    temp:              float
    demand_multiplier: float
    is_school_day:     bool
    is_public_holiday: bool
    is_pandemic:       bool   = False
    bus_type:          BusType = STD

# ─── 7) Tek bir sefer simülasyonu ─────────────────────────────────
def one_trip(dep: dt.datetime, sc: Scenario) -> dict:
    LOG = []
    env = simpy.Environment()

    class Bus:
        def __init__(self, env, dep, sc):
            self.env, self.dep, self.sc = env, dep, sc
            self.curr      = dep
            self.seg       = SEG_DF.copy()
            self.pax       = []
            self.max_occ   = 0
            self.boarded   = 0
            self.lags      = [0]*5
            self.wait_times = []      # ← Bekleme sürelerini biriktirecek liste
            env.process(self.run())

        def run(self):
            # Kalkış öncesi bekleme
            delay = (self.dep - dt.datetime.combine(self.dep.date(), dt.time())).seconds / 60
            yield self.env.timeout(delay)

            trip_time = 0.0

            for r in self.seg.itertuples():
                now, stop, km = self.curr, r.stop, r.km

                # 1) Seyahat süresi tahmini
                feat_df = pd.DataFrame([make_feats(now, self.sc, km, self.lags)], columns=FEATS)
                sec     = float(CAT.predict(feat_df)[0])
                dur     = sec / 60
                self.lags = [dur] + self.lags[:4]
                trip_time += dur

                yield self.env.timeout(dur)
                self.curr += dt.timedelta(minutes=dur)

                # 2) İnme-bindirme
                out = [p for p in self.pax if p == stop]
                self.pax = [p for p in self.pax if p != stop]

                lam = get_lambda(stop, now) * ARR_SCALE * self.sc.demand_multiplier
                if self.sc.is_public_holiday:
                    lam *= 0.8
                nin = np.random.poisson(lam * dur)

                dests, probs = DEST.get(stop, ([], []))
                new = (np.random.choice(dests, size=nin, p=probs).tolist() if dests else [stop] * nin)
                self.pax.extend(new)
                self.boarded += nin
                self.max_occ = max(self.max_occ, len(self.pax))

                # 3) Bekleme (dwell) süresi
                dwell = (r.dwell_sec / 60) + len(out) * 0.03 + nin * 0.01
                if self.sc.is_school_day:
                    dwell *= 1.1
                if self.sc.is_public_holiday:
                    dwell *= 1.2

                # Bekleme süresini kaydet
                self.wait_times.append(dwell)

                trip_time += dwell
                yield self.env.timeout(dwell)
                self.curr += dt.timedelta(minutes=dwell)

            # Sonuç kaydı: avg_wait eklendi
            LOG.append({
                "depart_time": self.dep.strftime("%H:%M"),
                "bus_type":    self.sc.bus_type.name,
                "capacity":    self.sc.bus_type.capacity,
                "trip_time":   round(trip_time, 2),
                "max_occ":     self.max_occ,
                "boarded":     self.boarded,
                "avg_wait":    round(sum(self.wait_times) / len(self.wait_times), 2)
            })

    Bus(env, dep, sc)
    env.run()
    return LOG[0]

# ─── 8) Ortalama trip zamanı ────────────────────────────────────────
def avg_trip(sc: Scenario, start: str, end: str) -> float:
    times = pd.date_range(start=start,end=end,freq="30min").time
    return np.mean([one_trip(dt.datetime.combine(dt.date.today(),t),sc)["trip_time"]
                    for t in times])

# ─── 9) Dinamik slot-day ──────────────────────────────────────────
CURRENT_THR = 90

def run_dynamic(sc: Scenario, start: str="06:00", end: str="23:00") -> pd.DataFrame:
    base   = avg_trip(sc, start, end)
    recs   = []
    headway = 30
    dep     = dt.datetime.combine(dt.date.today(), pd.to_datetime(start).time())
    end_dt  = dt.datetime.combine(dt.date.today(), pd.to_datetime(end).time())

    while dep <= end_dt:
        # 1) İlk araç tipi seçimi: beklenen talebe göre
        exp = estimate_expected_load(dep, sc)
        sc.bus_type = ARTIC if exp > CURRENT_THR else STD

        # 2) Seferi simüle et
        out = one_trip(dep, sc)
        out["headway"] = headway

        # 3) Yükü raporlamak için ölçekle
        out["max_occ"]  *= 6
        out["boarded"]  *= 5
        out["load_%"]    = round(100 * out["max_occ"] / out["capacity"], 2)

        # --- İKİNCİ AŞAMADAKİ ZORLAMA BLOĞU SİLİNDİ ---

        recs.append(out)

        # 4) Headway’i ayarla
        diff = out["trip_time"] - base
        if   diff > 2.5: headway = 7
        elif diff > 1:   headway = 10
        elif diff > 0.5: headway = 15
        else:            headway = 20

        dep += dt.timedelta(minutes=headway)

    return pd.DataFrame(recs)

# ─── 10) Endpoint ───────────────────────────────────────────────────
class SimRequest(BaseModel):
    weather_desc:      Literal["Clear","Cloudy","Precipitation","Storm"]
    temp:              float
    demand_multiplier: float
    is_school_day:     bool
    is_public_holiday: bool
    is_pandemic:       bool = False
    start:             str  # "06:00"
    end:               str  # "23:30"

@app.post("/simulate")
def simulate(req: SimRequest):
    sc = Scenario(
        weather_desc      = req.weather_desc,
        temp              = req.temp,
        demand_multiplier = req.demand_multiplier,
        is_school_day     = req.is_school_day,
        is_public_holiday = req.is_public_holiday,
        is_pandemic       = req.is_pandemic,
        bus_type          = STD
    )
    df = run_dynamic(sc, req.start, req.end)
    return df.to_dict(orient="records")
