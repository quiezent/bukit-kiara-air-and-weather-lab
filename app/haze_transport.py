"""Regional air-path context, deliberately separate from local PM prediction.

CAMS already transports satellite-derived fire emissions. This diagnostic adds
time-varying 925 hPa air paths and a 850 hPa height sensitivity check, NOT a
second smoke multiplier. Paths are 2-D, isobaric and coarse-grid: no dispersion,
vertical motion, deposition or source attribution is inferred from them.
"""
from collections import OrderedDict
from contextlib import closing
from datetime import datetime, timezone
import gzip
import json
import math
import sqlite3
import threading
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

VERSION = 'regional_air_paths_v1'
POLL_SECONDS = 3 * 3600
MAX_AGE_SECONDS = 6 * 3600
LATITUDES = (-4.0, -2.0, 0.0, 1.5, 3.1411106257487, 5.0)
LONGITUDES = (99.0, 100.5, 101.62749852676, 103.0, 105.0)
TARGET = (3.1411106257487, 101.62749852676)
FIELDS = ('wind_speed_925hPa', 'wind_direction_925hPa',
          'geopotential_height_925hPa', 'wind_speed_850hPa',
          'wind_direction_850hPa')
# Coarse land-sector polygons, not fire detections or national boundaries.
# Coordinates are (longitude, latitude). Border contacts are ambiguous.
SECTORS = {
    'central_sumatra': ('Central/eastern Sumatra', (
        (99.3, 1.4), (100.3, 2.1), (102.1, 1.1), (103.0, -0.4),
        (101.6, -1.5), (100.4, -0.6))),
    'southern_sumatra': ('Southern Sumatra', (
        (101.6, -1.5), (103.0, -0.4), (104.9, -2.5), (105.8, -5.2),
        (103.7, -4.7))),
}
_cache = OrderedDict()
_lock = threading.Lock()
_latest_cache = {}


def init_archive(path):
    with closing(sqlite3.connect(path, timeout=20)) as con, con:
        con.execute('CREATE TABLE IF NOT EXISTS regional_wind_runs('
                    'fetched_epoch INTEGER PRIMARY KEY, model_version TEXT NOT NULL, '
                    'payload BLOB NOT NULL)')


def fetch_grid(start_date=None, end_date=None):
    locations = [(a,b) for a in LATITUDES for b in LONGITUDES]
    params = {'latitude': ','.join(str(a) for a,b in locations),
              'longitude': ','.join(str(b) for a,b in locations),
              'hourly': ','.join(FIELDS), 'timezone': 'GMT',
              'timeformat': 'unixtime', 'wind_speed_unit': 'kmh',
              'models': 'gfs_global', 'cell_selection': 'nearest'}
    if start_date:
        params.update(start_date=start_date, end_date=end_date)
        endpoint = 'https://historical-forecast-api.open-meteo.com/v1/forecast'
    else:
        params.update(past_hours=72, forecast_hours=48)
        endpoint = 'https://api.open-meteo.com/v1/forecast'
    url = endpoint + '?' + urlencode(params)
    req = Request(url, headers={'User-Agent': 'BukitKiaraDashboard/1.0'})
    with urlopen(req, timeout=45) as response:
        raw = json.load(response)
    if not isinstance(raw, list) or len(raw) != len(locations):
        raise ValueError('Regional wind grid is incomplete')
    times = raw[0].get('hourly', {}).get('time', [])
    if not times or any(r.get('hourly', {}).get('time') != times for r in raw):
        raise ValueError('Regional wind grid time axes do not match')
    points = []
    for (lat,lon), r in zip(locations, raw):
        hourly = r['hourly']
        units = r.get('hourly_units', {})
        if any(units.get(f'wind_speed_{level}hPa') != 'km/h' for level in (925,850)):
            raise ValueError('Unexpected regional wind units')
        if any(len(hourly.get(f, [])) != len(times) for f in FIELDS):
            raise ValueError('Missing regional wind fields')
        # Values are supplied at the model cell, retained as provenance.
        points.append({'latitude':lat, 'longitude':lon,
                       'modelLatitude':r.get('latitude'),
                       'modelLongitude':r.get('longitude'),
                       **{f:hourly[f] for f in FIELDS}})
    return {'version':VERSION, 'fetchedEpoch':int(time.time()),
            'source':'Open-Meteo / NOAA GFS',
            'sourceUrl':'https://open-meteo.com/en/docs/gfs-api',
            'dataRole':('retrospective_stitched_model_diagnostic' if start_date
                        else 'issued_regional_wind_snapshot'),
            'times':times, 'latitudes':list(LATITUDES),
            'longitudes':list(LONGITUDES), 'points':points}


def save_grid(path, payload):
    if payload.get('dataRole') != 'issued_regional_wind_snapshot':
        raise ValueError('Retrospective winds cannot enter the issued archive')
    data = gzip.compress(json.dumps(payload, separators=(',',':'), allow_nan=False).encode())
    with closing(sqlite3.connect(path, timeout=20)) as con, con:
        con.execute('INSERT OR IGNORE INTO regional_wind_runs VALUES(?,?,?)',
                    (payload['fetchedEpoch'],VERSION,data))
        con.execute('DELETE FROM regional_wind_runs WHERE fetched_epoch < ?',
                    (int(time.time())-90*86400,))
    with _lock:
        _latest_cache[str(path)] = payload


def collector(path):
    init_archive(path)
    next_attempt = 0.0
    previous_tick = time.time()
    while True:
        now = time.time()
        if now - previous_tick > 420 or now < previous_tick - 5:
            next_attempt = now
        previous_tick = now
        if now < next_attempt:
            time.sleep(min(5.0, next_attempt - now))
            continue
        delay = POLL_SECONDS
        try:
            existing = latest_grid(path, int(time.time()))
            if existing and time.time()-existing['fetchedEpoch'] < POLL_SECONDS:
                delay = max(60, POLL_SECONDS-(time.time()-existing['fetchedEpoch']))
            else:
                payload = fetch_grid()
                save_grid(path, payload)
                print('[Regional wind] archived', payload['fetchedEpoch'], flush=True)
        except Exception as error:
            print('[Regional wind] ERROR:', error, flush=True)
            delay = 60
        finished = time.time()
        next_attempt = finished + delay
        previous_tick = finished


def latest_grid(path, as_of):
    with _lock:
        cached = _latest_cache.get(str(path))
    if cached and cached['fetchedEpoch'] <= as_of:
        return cached
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=10)
    try:
        try:
            row = con.execute('SELECT payload FROM regional_wind_runs '
                              'WHERE fetched_epoch<=? AND model_version=? '
                              'ORDER BY fetched_epoch DESC LIMIT 1', (as_of,VERSION)).fetchone()
        except sqlite3.OperationalError as exc:
            if 'no such table' in str(exc): return None
            raise
        return json.loads(gzip.decompress(row[0])) if row else None
    finally:
        con.close()


def inside(lat, lon, polygon):
    found = False
    previous = polygon[-1]
    for current in polygon:
        x,y = current; px,py = previous
        if (y > lat) != (py > lat) and lon < (px-x)*(lat-y)/(py-y)+x:
            found = not found
        previous = current
    return found


class WindGrid:
    def __init__(self, payload):
        self.times = np.asarray(payload['times'], dtype=float)
        nlon = len(payload['longitudes'])
        self.lats = np.asarray([p.get('modelLatitude',p['latitude'])
                                for p in payload['points'][::nlon]], dtype=float)
        self.lons = np.asarray([p.get('modelLongitude',p['longitude'])
                                for p in payload['points'][:nlon]], dtype=float)
        if np.any(np.diff(self.lats)<=0) or np.any(np.diff(self.lons)<=0):
            raise ValueError('Regional model cells are not a unique ordered grid')
        self.fields = {}
        shape = (len(self.lats),len(self.lons),len(self.times))
        for level in (925,850):
            speed = np.asarray([p[f'wind_speed_{level}hPa'] for p in payload['points']],dtype=float).reshape(shape)
            angle = np.radians(np.asarray([p[f'wind_direction_{level}hPa'] for p in payload['points']],dtype=float).reshape(shape))
            # Meteorological wind direction is FROM. Interpolate vectors,
            # never degrees (359 and 1 must not average to southerly flow).
            self.fields[f'u{level}'] = -speed*np.sin(angle)
            self.fields[f'v{level}'] = -speed*np.cos(angle)
        self.fields['height'] = np.asarray([p['geopotential_height_925hPa'] for p in payload['points']],dtype=float).reshape(shape)

    def value(self, name, lat, lon, epoch):
        if not (self.lats[0] <= lat <= self.lats[-1] and
                self.lons[0] <= lon <= self.lons[-1] and
                self.times[0] <= epoch <= self.times[-1]):
            return None
        brackets=[]
        for axis, val in ((self.lats,lat),(self.lons,lon),(self.times,epoch)):
            i = min(max(int(np.searchsorted(axis,val))-1,0),len(axis)-2)
            brackets.append((i,(val-axis[i])/(axis[i+1]-axis[i])))
        (i,a),(j,b),(k,c) = brackets
        total = 0.0
        for di,wa in ((0,1-a),(1,a)):
            for dj,wb in ((0,1-b),(1,b)):
                for dk,wc in ((0,1-c),(1,c)):
                    weight=wa*wb*wc
                    if weight <= 1e-12: continue
                    value = self.fields[name][i+di,j+dj,k+dk]
                    if not np.isfinite(value): return None
                    total += weight*value
        return float(total)

    def trace(self, arrival, level=925, max_hours=48):
        lat,lon=TARGET; epoch=float(arrival); hits={}; path=[]
        reason='48_hour_limit'; step=0.5
        for index in range(int(max_hours/step)+1):
            age=index*step
            path.append({'hoursBeforeArrival':age,'latitude':round(lat,3),'longitude':round(lon,3)})
            for key,(label,polygon) in SECTORS.items():
                if key not in hits and inside(lat,lon,polygon): hits[key]=age
            if age >= max_hours: break
            u=self.value(f'u{level}',lat,lon,epoch); v=self.value(f'v{level}',lat,lon,epoch)
            if u is None or v is None:
                reason='grid_or_time_limit'; break
            # Midpoint integration backwards through the changing wind field.
            midlat=lat-v*step/2/111.2
            midlon=lon-u*step/2/(111.2*math.cos(math.radians(lat)))
            mu=self.value(f'u{level}',midlat,midlon,epoch-step*1800)
            mv=self.value(f'v{level}',midlat,midlon,epoch-step*1800)
            if mu is None or mv is None:
                reason='grid_or_time_limit'; break
            lat-=mv*step/111.2
            lon-=mu*step/(111.2*math.cos(math.radians(midlat)))
            epoch-=step*3600
        return {'arrivalEpoch':int(arrival),'levelHpa':level,'sectorContacts':hits,
                'tracedHours':path[-1]['hoursBeforeArrival'], 'stopReason':reason,
                'path':path[::4]+([path[-1]] if len(path)%4 != 1 else [])}


def summarize_paths(payload, start, end, as_of):
    base={'available':False,'method':VERSION,'role':'air_path_context_only',
          'usedForLocalPmPoint':False,'sourceAttributionEstablished':False,
          'label':'Air-path model collecting'}
    if not payload: return base
    age=as_of-int(payload['fetchedEpoch'])
    base.update(fetchedAt=datetime.fromtimestamp(payload['fetchedEpoch'],timezone.utc).isoformat(),
                ageHours=round(age/3600,1),source=payload['source'])
    if age < 0 or age > MAX_AGE_SECONDS:
        base['label']='Air-path model out of date'; return base
    grid=WindGrid(payload)
    arrivals=[start,(start+end)//2,end]
    paths=[grid.trace(t,level) for level in (925,850) for t in arrivals]
    primary=[p for p in paths if p['levelHpa']==925]
    usable=[p for p in primary if p['tracedHours']>=12]
    contacts=[bool(p['sectorContacts']) for p in primary]
    sensitivity=[bool(p['sectorContacts']) for p in paths if p['levelHpa']==850]
    hit_times=[min(p['sectorContacts'].values()) for p in primary if p['sectorContacts']]
    sectors=sorted({key for p in primary for key in p['sectorContacts']})
    if len(usable)<3:
        label='Air-path coverage incomplete'; state='incomplete'
    elif all(contacts):
        label='Modeled air path via Sumatra'; state='sumatra_path'
    elif any(contacts):
        label='Sumatra connection changes during session'; state='mixed_path'
    elif min(p['tracedHours'] for p in primary) < 48:
        label='Source route unresolved'; state='origin_unresolved'
    else:
        label='No modeled Sumatra crossing'; state='no_crossing'
    if len(usable)==3 and contacts != sensitivity:
        label='Sumatra connection uncertain'; state='height_sensitive'
    heights=[grid.value('height',*TARGET,t) for t in arrivals]
    heights=[h for h in heights if h is not None]
    base.update(available=len(usable)==3,state=state,label=label,
                pressureLevelHpa=925,approxHeightM=round(np.mean(heights)) if heights else None,
                primaryPathCount=3,sumatraCrossingCount=sum(contacts),
                sensitivityLevelHpa=850,sensitivityCrossingCount=sum(sensitivity),
                heightSensitive=contacts!=sensitivity,
                sectorLabels=[SECTORS[s][0] for s in sectors],
                transitHoursMin=min(hit_times) if hit_times else None,
                transitHoursMax=max(hit_times) if hit_times else None,
                minimumTracedHours=min(p['tracedHours'] for p in primary),
                paths=paths,dataRole=payload.get('dataRole'),
                limitation='Simplified 2-D air paths, not confirmed smoke origin or plume arrival. No vertical mixing, removal or emissions calculation.')
    return base


def window_paths(path, start, end, as_of):
    payload=latest_grid(path,as_of)
    key=(str(path), (payload or {}).get('fetchedEpoch'),int(start),int(end),int(as_of)//900)
    with _lock:
        if key in _cache: return _cache[key]
    result=summarize_paths(payload,start,end,as_of)
    with _lock:
        _cache[key]=result
        while len(_cache)>12: _cache.popitem(last=False)
    return result
