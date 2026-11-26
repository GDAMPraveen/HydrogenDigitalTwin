
import os
import requests
from pathlib import Path

api_key = os.environ.get("NLR_API_KEY")

if not api_key:
    raise RuntimeError("NLR_API_KEY is not set.")

url = (
    "https://developer.nlr.gov/api/nsrdb/v2/"
    "solar/msg-iodc-download.csv"
)

params = {"api_key": api_key}

data = {
    "wkt": "POINT(77.5946 12.9716)",
    "names": "2017",
    "interval": "15",
    "utc": "true",
    "leap_day": "true",
    "attributes": (
        "ghi,dni,dhi,air_temperature,wind_speed,"
        "relative_humidity,surface_pressure,solar_zenith_angle"
    ),
    "email": "p2452293@gmail.com"
}

response = requests.post(
    url,
    params=params,
    data=data,
    timeout=180
)

response.raise_for_status()

output_dir = Path("data/solar")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "bengaluru_2017_meteosat.csv"
output_file.write_bytes(response.content)

print("Download completed!")
print(f"File: {output_file}")
print(f"Size: {output_file.stat().st_size:,} bytes")
