# Data sources and attribution

This repository contains software and development writing, not a redistributed operational database. Fetches made by a local installation are subject to the providers' current availability, terms and attribution requirements.

## AirGradient

The dashboard uses public TTDI location **86311**, with raw sensor variables obtained from the AirGradient world endpoint:

```text
https://api.airgradient.com/public/api/v1/world/locations/86311/measures/current
```

Credit: **AirGradient and the public station contributor**. See [AirGradient](https://www.airgradient.com/) and its [public API documentation](https://api.airgradient.com/public/docs/api/v1/). The implementation's world endpoint and the documented account/location routes should not be assumed interchangeable. No service-level guarantee or historical recovery capability is implied. The software does not scrape IQAir as its live concentration source.

The server also polls six neighbouring public AirGradient stations around Kuala Lumpur every 300 s through AirGradient's map data API (`https://map-data.airgradient.com/map/api/v1`). Since v25 (14 October 2026) these readings are inputs to the +90 and ride-window forecasts. Each poll keeps its receipt time, so a reading is only used once it was actually available.

## Open-Meteo weather

Credit: **Open-Meteo and its underlying numerical weather providers**. The application stores retrieved forecasts so evaluation can distinguish available-at-issue input from later forecast revisions. A frequently refreshed dashboard does not turn hourly model values into second-by-second local observations. [Weather API documentation](https://open-meteo.com/en/docs).

## Copernicus Atmosphere Monitoring Service

Credit: **Copernicus Atmosphere Monitoring Service (CAMS), its data providers, and Open-Meteo** for access. Kuala Lumpur uses regional/global atmospheric-composition information, not the European domain's resolution. Regional concentration is not a neighbourhood instrument reading. Consult [Open-Meteo's air-quality documentation and attribution requirements](https://open-meteo.com/en/docs/air-quality-api), [CAMS](https://atmosphere.copernicus.eu/) and the [CAMS provider acknowledgement](https://confluence.ecmwf.int/display/CKB/CAMS+Regional%3A+European+air+quality+analysis+and+forecast+data+documentation).

## MET Malaysia / WMO WIS2

Credit: **Malaysian Meteorological Department (MET Malaysia)**. The code retrieves Subang hourly SYNOP reference observations from its WIS2 service, station WIGOS `0-20000-0-48647`. This is a different location from the TTDI sensor. Observations are used as reference context, not relabelled as a forecast. [MET Malaysia](https://www.met.gov.my/) and [its WIS2 service](http://wis2node.met.gov.my/oapi/collections/urn:wmo:md:my-metmalaysia:synop-hourly/items).

## RainViewer radar

Credit: [Weather data by RainViewer](https://www.rainviewer.com/). The server keeps timestamped summary features from recent *past* radar frames around TTDI for research. It stores no radar images, and radar is not an input to the published PM2.5 forecasts. A composite frame's timestamp is not necessarily its local scan time, and radar echo is not a ground-level rain measurement. RainViewer's [public API terms](https://www.rainviewer.com/api.html) permit personal and educational use and require visible credit; the dashboard shows it in its source details. Availability is best effort.

Provider names, logos and datasets are not owned or relicensed by this project. There is no endorsement by the providers. Before a commercial, redistributed or high-volume deployment, check the actual provider terms and access arrangements; this public source snapshot makes no grant over third-party data.
