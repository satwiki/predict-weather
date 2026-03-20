import re
from urllib.parse import urlparse, parse_qs


def extract_coords_from_google_maps_url(url):
    """
    Extract (latitude, longitude) from various Google Maps URL formats.

    Supported:
      - https://www.google.com/maps/@lat,lon,zoom
      - https://www.google.com/maps/place/.../@lat,lon,zoom/...
      - https://maps.google.com/?q=lat,lon
      - https://www.google.com/maps?q=lat,lon

    Returns (lat, lon) tuple or None if parsing fails.
    """
    url = url.strip()

    # Pattern: /@lat,lon
    match = re.search(r"@(-?\d+\.?\d*),(-?\d+\.?\d*)", url)
    if match:
        return float(match.group(1)), float(match.group(2))

    # Pattern: ?q=lat,lon or &q=lat,lon
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    if "q" in params:
        q = params["q"][0]
        coord_match = re.match(r"(-?\d+\.?\d*),\s*(-?\d+\.?\d*)", q)
        if coord_match:
            return float(coord_match.group(1)), float(coord_match.group(2))

    # Pattern: /place/Name/lat,lon
    match = re.search(r"/place/[^/]+/(-?\d+\.?\d*),(-?\d+\.?\d*)", url)
    if match:
        return float(match.group(1)), float(match.group(2))

    return None
