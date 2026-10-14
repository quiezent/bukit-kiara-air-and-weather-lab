# Dataset schema

`data\models\dataset.parquet` (parquet, index `issue_epoch` = UTC seconds). One row per 5-min issue clock from 2026-08-26 00:00:00 to 2026-10-13 21:50:00 MYT (14087 rows, 585 columns). `eval` marks issues from 2026-09-10 (14 full days of sensor and weather history before it).

Row columns: `issue_myt`, `date_myt`, `canonical` ('07:15', '12:00', '18:05' or ''), `eval`.

## Availability rule

Every feature value was visible at the issue time:
- sensor readings with a receipt (from 2026-10-08 13:12): max(available, confirmed) of the first receipt; older readings: observation + 60 s (measured receipt latency p99 = 60 s; on post-ledger grid issues this rule never admits a reading before its real receipt, see `harness.audit`; production itself assumes 0 s);
- weather / CAMS runs: receipt clock when present, else fetched + 5 s (receipts lag fetch by <= 5 s);
- regional PM: recorded_epoch; regional wind: fetched + 5 s; Subang: first_fetched_epoch;
- neighbours: receipt clock (strict group) or bucket end + 30 min (reconstructed group).

## Targets (production definitions)

- `y_ref`: Fresh 5-min reference at the issue (median of visible readings in (t-300 s, t], newest <= 240 s old). Persistence forecast for every concentration target.
- `y_arrival`: +90 arrival: interpolated 15-min bucket-median proxy at exactly t+90 min (fresh_numeric_model_v4.targets_for_issues / arrival_change_model). NaN if either bucket is missing.
- `y_arrival_complete`: Epoch when the y_arrival buckets have closed (label usable for training only after this).
- `y_ride_mean / y_ride_min / y_ride_max`: Ride window [t+90, t+210]: overlap-weighted mean of 15-min bucket medians; min/max over buckets with positive overlap (partial edge buckets summarise their full 15 min). All NaN if any bucket is missing (targets_for_issues).
- `y_ride_joint_valid`: All four Ridge heads finite (Ridge v4 trains only on these rows).
- `y_ride_complete`: Epoch when the last ride bucket closes.
- `y_delta`: y_arrival - y_ref.
- `y_class5`: Production arrival classes: 0 delta<=-40, 1 (-40,-20], 2 (-20,20), 3 [20,40), 4 >=40; -1 = no label.
- `y_fall20 / y_rise20 / y_fall40 / y_rise40`: Binary versions of y_class5 (fall20 = class 0 or 1, rise20 = 3 or 4, fall40 = 0, rise40 = 4); NaN without label.
- `y_ride_le70`: 1 if y_ride_mean <= 70 (cycling_window_probability.target_label).
- `y_first20`: First crossing within 90 min: first of the six +15..+90 proxies with |proxy - y_ref| >= 20; -1 drop, +1 rise, 0 none (fresh_event_model labels).
- `y_first20_step / y_first20_complete`: Index (0..5) of the first crossing step; epoch when all six proxies are complete.
- `y_morning_* / y_afternoon_*`: Next feasible session (production next_feasible_ride_period): Morning 09:00-11:00, Afternoon 14:00-16:00 MYT, first window starting >= t+90 min. _mean = mean of its 8 closed 15-min bucket medians (window_pm_predictor._session_actual); _start, _complete (= window end), _lead_min, _same_day.

## Feature groups

### meta (5 columns)

Issue metadata: m_fresh_available (production freshness rule), m_fresh_ref_epoch, m_sensor_watermark (newest visible observation), m_input_rows_180min, m_receipts_all_known (every visible input row in the 180-min window has a real receipt clock; False while any pre-ledger row is used, i.e. before about 2026-10-08 16:12).

`m_fresh_available`, `m_fresh_ref_epoch`, `m_sensor_watermark`, `m_input_rows_180min`, `m_receipts_all_known`

### sensor (208 columns)

fresh_sensor_features.FEATURE_COLUMNS computed by the production function on rows visible at the issue: freshPm25, raw lags/deltas 5..60 min, SDs, ranges, slopes, fresh temperature/RH and 30-min deltas, sample counts, time and issue-offset cycles, and 3-min sequence bins of PM2.5/temperature/RH over 180 min (seq_<channel>_<age>_<age+3>min). The 28 non-sequence columns are the arrival logistic's inputs.

### time (8 columns)

MYT hour, day cycle, day of week, weekend, days since grid start.

`t_hour`, `t_day_sin`, `t_day_cos`, `t_dow`, `t_dow_sin`, `t_dow_cos`, `t_weekend`, `t_days_since_start`

### weather (227 columns)

Latest Open-Meteo vintage visible at the issue (age <= 2 h): point values at +0..+360 min (w_<field>_<lead>, precipitation/showers = right-labelled hourly amount), wind u/v at 10 m/180 m/925 hPa, ride-window and next Morning/Afternoon session means and precipitation sums, prior/next 3 h precipitation, max precipitation probability in 6 h, first lead with probability >= 50, and 60-min vintage revisions at +90 (same as production arrival_feature_inputs).

`w_age_min`, `w_temp_0`, `w_temp_30`, `w_temp_60`, `w_temp_90`, `w_temp_120`, `w_temp_180`, `w_temp_240`, `w_temp_300`, `w_temp_360`, `w_rh_0`, `w_rh_30`, `w_rh_60`, `w_rh_90`, `w_rh_120`, `w_rh_180`, `w_rh_240`, `w_rh_300`, `w_rh_360`, `w_feels_0`, `w_feels_30`, `w_feels_60`, `w_feels_90`, `w_feels_120`, `w_feels_180`, `w_feels_240`, `w_feels_300`, `w_feels_360`, `w_pprob_0`, `w_pprob_30`, `w_pprob_60`, `w_pprob_90`, `w_pprob_120`, `w_pprob_180`, `w_pprob_240`, `w_pprob_300`, `w_pprob_360`, `w_precip_0`, `w_precip_30`, `w_precip_60`, `w_precip_90`, `w_precip_120`, `w_precip_180`, `w_precip_240`, `w_precip_300`, `w_precip_360`, `w_showers_0`, `w_showers_30`, `w_showers_60`, `w_showers_90`, `w_showers_120`, `w_showers_180`, `w_showers_240`, `w_showers_300`, `w_showers_360`, `w_cloud_0`, `w_cloud_30`, `w_cloud_60`, `w_cloud_90`, `w_cloud_120`, `w_cloud_180`, `w_cloud_240`, `w_cloud_300`, `w_cloud_360`, `w_pmsl_0`, `w_pmsl_30`, `w_pmsl_60`, `w_pmsl_90`, `w_pmsl_120`, `w_pmsl_180`, `w_pmsl_240`, `w_pmsl_300`, `w_pmsl_360`, `w_wspd10_0`, `w_wspd10_30`, `w_wspd10_60`, `w_wspd10_90`, `w_wspd10_120`, `w_wspd10_180`, `w_wspd10_240`, `w_wspd10_300`, `w_wspd10_360`, `w_gust_0`, `w_gust_30`, `w_gust_60`, `w_gust_90`, `w_gust_120`, `w_gust_180`, `w_gust_240`, `w_gust_300`, `w_gust_360`, `w_wspd180_0`, `w_wspd180_30`, `w_wspd180_60`, `w_wspd180_90`, `w_wspd180_120`, `w_wspd180_180`, `w_wspd180_240`, `w_wspd180_300`, `w_wspd180_360`, `w_wspd925_0`, `w_wspd925_30`, `w_wspd925_60`, `w_wspd925_90`, `w_wspd925_120`, `w_wspd925_180`, `w_wspd925_240`, `w_wspd925_300`, `w_wspd925_360`, `w_blh_0`, `w_blh_30`, `w_blh_60`, `w_blh_90`, `w_blh_120`, `w_blh_180`, `w_blh_240`, `w_blh_300`, `w_blh_360`, `w_cape_0`, `w_cape_30`, `w_cape_60`, `w_cape_90`, `w_cape_120`, `w_cape_180`, `w_cape_240`, `w_cape_300`, `w_cape_360`, `w_wind10_u_0`, `w_wind10_u_30`, `w_wind10_u_60`, `w_wind10_u_90`, `w_wind10_u_120`, `w_wind10_u_180`, `w_wind10_u_240`, `w_wind10_u_300`, `w_wind10_u_360`, `w_wind10_v_0`, `w_wind10_v_30`, `w_wind10_v_60`, `w_wind10_v_90`, `w_wind10_v_120`, `w_wind10_v_180`, `w_wind10_v_240`, `w_wind10_v_300`, `w_wind10_v_360`, `w_wind180_u_0`, `w_wind180_u_30`, `w_wind180_u_60`, `w_wind180_u_90`, `w_wind180_u_120`, `w_wind180_u_180`, `w_wind180_u_240`, `w_wind180_u_300`, `w_wind180_u_360`, `w_wind180_v_0`, `w_wind180_v_30`, `w_wind180_v_60`, `w_wind180_v_90`, `w_wind180_v_120`, `w_wind180_v_180`, `w_wind180_v_240`, `w_wind180_v_300`, `w_wind180_v_360`, `w_wind925_u_0`, `w_wind925_u_30`, `w_wind925_u_60`, `w_wind925_u_90`, `w_wind925_u_120`, `w_wind925_u_180`, `w_wind925_u_240`, `w_wind925_u_300`, `w_wind925_u_360`, `w_wind925_v_0`, `w_wind925_v_30`, `w_wind925_v_60`, `w_wind925_v_90`, `w_wind925_v_120`, `w_wind925_v_180`, `w_wind925_v_240`, `w_wind925_v_300`, `w_wind925_v_360`, `w_ride_temp_mean`, `w_ride_rh_mean`, `w_ride_pprob_mean`, `w_ride_wspd10_mean`, `w_ride_wind10_u_mean`, `w_ride_wind10_v_mean`, `w_ride_wind925_u_mean`, `w_ride_wind925_v_mean`, `w_ride_blh_mean`, `w_ride_cloud_mean`, `w_ride_cape_mean`, `w_ride_precip_sum`, `w_morning_temp_mean`, `w_morning_rh_mean`, `w_morning_pprob_mean`, `w_morning_wspd10_mean`, `w_morning_wind10_u_mean`, `w_morning_wind10_v_mean`, `w_morning_wind925_u_mean`, `w_morning_wind925_v_mean`, `w_morning_blh_mean`, `w_morning_cloud_mean`, `w_morning_cape_mean`, `w_morning_precip_sum`, `w_morning_covered`, `w_afternoon_temp_mean`, `w_afternoon_rh_mean`, `w_afternoon_pprob_mean`, `w_afternoon_wspd10_mean`, `w_afternoon_wind10_u_mean`, `w_afternoon_wind10_v_mean`, `w_afternoon_wind925_u_mean`, `w_afternoon_wind925_v_mean`, `w_afternoon_blh_mean`, `w_afternoon_cloud_mean`, `w_afternoon_cape_mean`, `w_afternoon_precip_sum`, `w_afternoon_covered`, `w_prior3h_precip_sum`, `w_next3h_precip_sum`, `w_next6h_pprob_max`, `w_first_pprob50_lead_min`, `w_rev60_temp_90`, `w_rev60_rh_90`, `w_rev60_pprob_90`, `w_rev60_wspd10_90`

### cams (21 columns)

Latest CAMS (Open-Meteo, cams_anchor_v1/v2) run visible at the issue (age <= 3 h): PM2.5/PM10/AOD/dust at +0/+90/+180/+360, +90 change, ride-window and session means.

`c_age_min`, `c_pm25_0`, `c_pm25_90`, `c_pm25_180`, `c_pm25_360`, `c_pm10_0`, `c_pm10_90`, `c_pm10_180`, `c_pm10_360`, `c_aod_0`, `c_aod_90`, `c_aod_180`, `c_aod_360`, `c_dust_0`, `c_dust_90`, `c_dust_180`, `c_dust_360`, `c_pm25_delta90`, `c_pm25_ride_mean`, `c_pm25_morning_mean`, `c_pm25_afternoon_mean`

### regional_pm (13 columns)

CAMS 3x3 regional grid (2026-10-01 onward): mean, N-S and E-W gradients and max at +0/+90/+180.

`rp_mean_0`, `rp_ns_grad_0`, `rp_ew_grad_0`, `rp_max_0`, `rp_mean_90`, `rp_ns_grad_90`, `rp_ew_grad_90`, `rp_max_90`, `rp_mean_180`, `rp_ns_grad_180`, `rp_ew_grad_180`, `rp_max_180`, `rp_age_min`

### regional_wind (11 columns)

GFS 925/850 hPa winds (2026-09-05 onward): u/v nearest TTDI at +0/+90 and 925 hPa domain-mean u/v.

`rw_925_u_0`, `rw_925_v_0`, `rw_850_u_0`, `rw_850_v_0`, `rw_925_u_90`, `rw_925_v_90`, `rw_850_u_90`, `rw_850_v_90`, `rw_925_domain_u_0`, `rw_925_domain_v_0`, `rw_age_min`

### subang (12 columns)

Latest Subang airport report visible (first-fetch clock, age <= 3 h): temperature, RH, dew point, visibility, wind speed/u/v, rain/thunder/haze flags, precipitation.

`o_age_min`, `o_temp`, `o_rh`, `o_dewpoint`, `o_visibility_km`, `o_wind_kmh`, `o_wind_u`, `o_wind_v`, `o_rain`, `o_thunder`, `o_haze`, `o_precip_mm`

### neighbors_strict (24 columns)

Seven AirGradient stations; value visible by its receipt clock (first request that returned it). The archive starts with a backfill received at 2026-10-05 19:30 MYT, so this group is empty before 19:35 that day.

`n_956_pm25`, `n_956_age_min`, `n_956_chg60`, `n_1087_pm25`, `n_1087_age_min`, `n_1087_chg60`, `n_52426946_pm25`, `n_52426946_age_min`, `n_52426946_chg60`, `n_98982064_pm25`, `n_98982064_age_min`, `n_98982064_chg60`, `n_99667152_pm25`, `n_99667152_age_min`, `n_99667152_chg60`, `n_109266655_pm25`, `n_109266655_age_min`, `n_109266655_chg60`, `n_111145052_pm25`, `n_111145052_age_min`, `n_111145052_chg60`, `n_count`, `n_mean`, `n_mean_minus_ref`

### neighbors_reconstructed (24 columns)

Same stations, visibility = min(receipt, bucket end + 30 min). An assumption about AirGradient cloud latency; not a proven clock.

`nr_956_pm25`, `nr_956_age_min`, `nr_956_chg60`, `nr_1087_pm25`, `nr_1087_age_min`, `nr_1087_chg60`, `nr_52426946_pm25`, `nr_52426946_age_min`, `nr_52426946_chg60`, `nr_98982064_pm25`, `nr_98982064_age_min`, `nr_98982064_chg60`, `nr_99667152_pm25`, `nr_99667152_age_min`, `nr_99667152_chg60`, `nr_109266655_pm25`, `nr_109266655_age_min`, `nr_109266655_chg60`, `nr_111145052_pm25`, `nr_111145052_age_min`, `nr_111145052_chg60`, `nr_count`, `nr_mean`, `nr_mean_minus_ref`
