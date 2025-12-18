import os
from pathlib import Path
from time import sleep

import astropy.units as u
import click
import numpy as np
import requests
from astropy.coordinates import ICRS, AltAz, Angle, EarthLocation, SkyCoord
from astropy.table import QTable, Table
from astropy.time import Time

try:
    from colorama import init

    init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

DEFAULT_STELLARIUM_URL = "http://localhost:8090"
CAMERA_CHOICES = ["LSSTCam", "ComCam", "LATISS"]
DEFAULT_RSP_SERVER = "https://usdf-rsp.slac.stanford.edu"
DEFAULT_RSP_TOKEN_PATH = "~/.lsst/rsp_token"
DEFAULT_EFD_CLIENT = "usdf_efd"
# Using Stellarium values for Cerro Pachón
RUBIN_OBSERVATORY = EarthLocation(
    lat=Angle("-30d14m27.00s"),
    lon=Angle("-70d44m11.99s"),
    height=2722 * u.m,
)


# Cribbing some items from https://github.com/lsst-sims/rubin_nights
def get_access_token(tokenfile: str | Path | None = None) -> str:
    """Retrieve RSP access token.

    Parameters
    ----------
    tokenfile : `str`, `Path`, or None
        Path to token file.
        Default None will fall back to environment variable,
        ACCESS_TOKEN and then try lsst.rsp.get_access_token().

    Returns
    -------
    token : `str`
        Token value.
    """
    token = None
    try:
        # Try using lsst-rsp first
        import lsst.rsp.get_access_token as rsp_get_access_token

        token = rsp_get_access_token(tokenfile=tokenfile)
    except ImportError:
        # No lsst-rsp available
        token_path = (
            Path(tokenfile) if tokenfile is not None else Path(DEFAULT_RSP_TOKEN_PATH)
        ).expanduser()
        if token_path.exists():
            with open(token_path, "r") as f:
                token = f.read().strip()
        if token is None:
            # Try using environment variable
            token = os.environ.get("ACCESS_TOKEN", None)

    if token is None:
        print("No RSP token available.")
        token = ""
    return token


class ConsDB:
    """Class to query ConsDB.

    Uses the ConsDbClient if available, otherwise falls back to
    using the REST API.

    Parameters
    ----------
    server : str
        ConsDB server URL.
    token : str
        RSP access token.
    """

    def __init__(self, server, token):
        self.server = server
        self.token = token
        try:
            from lsst.summit.utils import ConsDbClient

            server = server.replace("https://", f"https://user:{token}@") + "/consdb"
            self.client = ConsDbClient(server)
        except ImportError:
            self.client = None

    def query(self, query):
        """Query ConsDB.

        Parameters
        ----------
        query : str
            SQL query to execute.

        Returns
        -------
        astropy.table.Table
            Table with results of query.
        """
        if self.client is not None:
            return self.client.query(query)
        else:
            auth = "user", self.token
            params = {
                "query": query,
            }
            response = requests.post(
                self.server + "/consdb/query", auth=auth, json=params
            )
            columns = response.json()["columns"]
            data = response.json()["data"]
            table = Table(names=columns, data=np.array(data))
            return table


class EFD:
    """Class to query EFD.

    Uses lsst.efd.client.EFDClient if available, otherwise falls back to
    using the REST API.

    Parameters
    ----------
    server : str
        EFD server URL.
    """

    def __init__(self, server):
        if "usdf" in server:
            site = "usdf_efd"
        else:
            site = "summit_efd"
        try:
            from lsst.summit.utils.efdUtils import getMostRecentRowWithDataBefore
            from lsst_efd_client import EfdClient

            self.client = EfdClient(site)
        except ImportError:
            self.client = None
            creds_service = f"https://roundtable.lsst.codes/segwarides/creds/{site}"
            efd_creds = requests.get(creds_service).json()
            self.auth = efd_creds["username"], efd_creds["password"]
            self.url = f"https://{efd_creds['host']}{efd_creds['path']}query"

    def get_most_recent_row_before(self, topic, time):
        if self.client is not None:
            from aiohttp.client_exceptions import ServerDisconnectedError
            from lsst.summit.utils.efdUtils import getMostRecentRowWithDataBefore

            try:
                out = getMostRecentRowWithDataBefore(
                    self.client, topic, timeToLookBefore=time, maxSearchNMinutes=5
                )
            except ServerDisconnectedError:
                raise ValueError()
            return out
        else:
            import pandas as pd

            # Stealing from summit_utils and rubin_nights
            earliest = Time("2019-01-01")
            if time < earliest:
                raise ValueError(f"Time {time} is before EFD start time.")
            earliest = max(earliest, time - 5 * u.min)
            query = f'select * from "{topic}" '
            query += f"where time > '{earliest.isot}Z' and time <= '{time.isot}Z'"
            params = {"db": "efd", "q": query}
            while True:
                try:
                    response = requests.get(
                        self.url, auth=self.auth, params=params
                    ).json()
                except ValueError:
                    print("Waiting for EFD...")
                    sleep(1.0)
                    continue
                else:
                    break

            statement = response["results"][0]
            if "series" not in statement:
                raise ValueError(f"No data found for topic {topic} at time {time}.")
            series = statement["series"][0]
            result = pd.DataFrame(series.get("values", []), columns=series["columns"])
            if "time" in result:
                result = result.set_index(pd.to_datetime(result["time"]))
                result = result.drop("time", axis=1)
            return result.iloc[-1]


def parallactic_angle(coord, time):
    """Calculate the parallactic angle.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Coordinates of the object.  Must have alt and az attributes.
    time : astropy.time.Time
        Time of the observation.

    Returns
    -------
    astropy.coordinates.Angle
        Parallactic angle.
    """
    loc = RUBIN_OBSERVATORY
    coord = coord.transform_to(AltAz(obstime=time, location=loc))

    ha, dec = coord.hadec.ha, coord.hadec.dec
    lat = loc.lat

    sinq = np.sin(ha)
    cosq = np.tan(lat) * np.cos(dec) - np.sin(dec) * np.cos(ha)
    return Angle(np.arctan2(sinq, cosq))


def set_camera(url, camera):
    """Set the camera in Stellarium.

    Parameters
    ----------
    url : str
        URL of Stellarium API.
    camera : str
        Camera to set.
    """
    url = f"{url}/api/stelproperty/set"
    requests.post(url, data={"id": "MosaicCamera.currentCamera", "value": camera})
    requests.post(url, data={"id": "MosaicCamera.visible", "value": "True"})


def slew_to(url, camera, ra, dec, rsp):
    """Send camera repositioning command to Stellarium.

    Parameters
    ----------
    url : str
        URL of Stellarium API.
    camera : str
        Camera to slew.
    ra : Angle
        Right ascension.
    dec : Angle
        Declination.
    rsp : Angle
        Rotator sky position (E of N).
    """
    if not np.isfinite(ra):
        cprint(f"ra is not finite: {ra}", "bright_red")
    if not np.isfinite(dec):
        cprint(f"dec is not finite: {dec}", "bright_red")
    if not np.isfinite(rsp):
        cprint(f"rsp is not finite: {rsp}", "bright_red")

    url = f"{url}/api/stelproperty/set"
    for data in [
        {"id": "MosaicCamera.currentCamera", "value": camera},
        {"id": "MosaicCamera.visible", "value": "True"},
        {"id": "MosaicCamera.ra", "value": ra.deg},
        {"id": "MosaicCamera.dec", "value": dec.deg},
        {"id": "MosaicCamera.rotation", "value": rsp.deg},
    ]:
        requests.post(url, data=data)


def parse_angle(value, unit=u.deg):
    """Parse an angle with units.

    Parameters
    ----------
    value : str
        Angle string.
    unit : astropy.units.Unit, optional
        Unit of the angle if not inferable from value.  Default is degrees.

    Returns
    -------
    astropy.coordinates.Angle
        Angle object.
    """
    try:
        angle = Angle(value)
    except u.UnitsError:
        angle = Angle(value, unit=unit)
    return angle


def get_stellarium_attributes(url):
    """Get particular Stellarium attributes of interest.

    Parameters
    ----------
    url : str
        URL of Stellarium API.

    Returns
    -------
    dict
        Dictionary with structure:
            time: astropy.time.Time
                Time in Stellarium.
            last: astropy.coordinates.Angle
                Local apparent sidereal time at Rubin Observatory.
            current_camera: str
                Current camera in Stellarium.
            az: astropy.coordinates.Angle
                Azimuth of current camera.
            alt: astropy.coordinates.Angle
                Altitude of current camera.
            rsp: astropy.coordinates.Angle
                Sky rotation (RotSkyPos) of current camera.
            q: astropy.coordinates.Angle
                Parallactic angle of current camera.
            ra: astropy.coordinates.Angle
                Right ascension of current camera.
            dec: astropy.coordinates.Angle
                Declination of current camera.
            rtp: astropy.coordinates.Angle
                Rotator position (RotTelPos) of current camera.
            ha: astropy.coordinates.Angle
                Hour angle of current camera.
    """
    time = Time(
        requests.get(f"{url}/api/main/status").json()["time"]["jday"], format="jd"
    )
    properties = requests.get(f"{url}/api/stelproperty/list").json()
    ra = properties["MosaicCamera.ra"]["value"]
    dec = properties["MosaicCamera.dec"]["value"]
    rsp = properties["MosaicCamera.rotation"]["value"]
    current_camera = properties["MosaicCamera.currentCamera"]["value"]
    while any(x is None for x in [ra, dec, rsp, current_camera]):
        print("Waiting for Stellarium...")
        sleep(0.5)
        properties = requests.get(f"{url}/api/stelproperty/list").json()
        ra = properties["MosaicCamera.ra"]["value"]
        dec = properties["MosaicCamera.dec"]["value"]
        rsp = properties["MosaicCamera.rotation"]["value"]
        current_camera = properties["MosaicCamera.currentCamera"]["value"]

    ra = Angle(ra * u.deg).wrap_at("360d")
    dec = Angle(dec * u.deg)
    rsp = Angle(rsp, unit=u.deg).wrap_at("360d")
    coord = SkyCoord(ra=ra, dec=dec)
    aa = coord.transform_to(AltAz(obstime=time, location=RUBIN_OBSERVATORY))
    last = Time(time, location=RUBIN_OBSERVATORY).sidereal_time("apparent")
    ha = last - ra
    alt = aa.alt
    az = aa.az.wrap_at("180d")
    q = parallactic_angle(coord, time).wrap_at("360d")
    rtp = q - rsp - 90 * u.deg
    rtp = rtp.wrap_at("90d")  # Should always be between -90 and +90 degrees
    return dict(
        time=time,
        last=last,
        current_camera=current_camera,
        az=az,
        alt=alt,
        rsp=rsp,
        q=q,
        ra=ra,
        dec=dec,
        rtp=rtp,
        ha=ha,
    )


def cprint(text, color=None):
    if HAS_COLORAMA:
        from colorama import Fore, Style

        # All color names
        colors = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]

        # Generate the color dictionary programmatically
        color_dict = {color: getattr(Fore, color.upper()) for color in colors}
        color_dict.update(
            {
                f"bright_{color}": getattr(Fore, f"LIGHT{color.upper()}_EX")
                for color in colors
            }
        )

        # Get the color code from the dictionary
        color_code = color_dict.get(color.lower(), "") if color else ""
        print(f"{color_code}{text}{Style.RESET_ALL}")

    else:
        print(text)


def print_state(api_url):
    data = get_stellarium_attributes(api_url)

    to_string_kwargs = dict(
        unit=u.deg,
        precision=1,
        pad=True,
        format="unicode",
        alwayssign=True,
    )

    print(f"Camera: {data['current_camera']}")
    cprint(f"Time: {data['time'].iso} UTC", "yellow")
    cprint(
        f"parallactic angle:                                 {data['q'].deg:8.3f} deg",
        "bright_magenta",
    )
    az_str = data["az"].to_string(**to_string_kwargs)
    alt_str = data["alt"].to_string(**to_string_kwargs)
    rtp_str = f"{data['rtp'].deg:8.3f} deg"
    cprint(
        f"Az/Alt/RotTelPos:   {az_str:>13s}   {alt_str:>12s}   {rtp_str:>11s}",
        "bright_green",
    )

    ra_str = data["ra"].to_string(unit="hour", precision=1, pad=True)
    dec_str = data["dec"].to_string(**to_string_kwargs)
    rsp_str = f"{data['rsp'].deg:8.3f} deg"
    cprint(
        f"RA/Dec/RotSkyPos:   {ra_str:>13s}   {dec_str:>12s}   {rsp_str:>11s}",
        "bright_cyan",
    )
    rot_speed = (
        Angle("360d")
        / u.day
        * np.cos(RUBIN_OBSERVATORY.lat)
        * np.cos(data["az"])
        / np.cos(data["alt"])
    )
    alt_speed = (
        Angle("360d")
        / u.day
        * np.cos(RUBIN_OBSERVATORY.lat)
        * np.sin(data["az"])
    )
    az_speed = (
        Angle("360d")
        / u.day
        * (
            np.sin(RUBIN_OBSERVATORY.lat)
            - np.cos(RUBIN_OBSERVATORY.lat) * np.tan(data["alt"]) * np.cos(data["az"])
        )
    )

    rot_vel_str = f"{rot_speed.to(u.arcsec/u.s).value:>7.2f}"
    alt_vel_str = f"{alt_speed.to(u.arcsec/u.s).value:>7.2f}"
    az_vel_str = f"{az_speed.to(u.arcsec/u.s).value:>7.2f}"
    cprint(
        f"Az/Alt/Rot rate [\" / s]: "
        f"{rot_vel_str:7s}        {alt_vel_str:7s}     {az_vel_str:7s}",
        "bright_red"
    )


def set_view_to_camera(api_url):
    """Set Stellarium view to camera.

    Parameters
    ----------
    api_url : str
        URL of Stellarium API.
    """
    requests.post(f"{api_url}/api/stelaction/do", data={"id": "actionSetViewToCamera"})


def set_stellarium_time(api_url, time, timerate=None):
    """Set Stellarium time.

    Parameters
    ----------
    api_url : str
        URL of Stellarium API.
    time : astropy.time.Time
        Time to set.
    timerate : float, optional
        Timerate to set in days per second.  Default is None, which does not set the
        timerate.
    """
    data = {"time": time.mjd + 2400000.5}
    if timerate is not None:
        data["timerate"] = timerate
    requests.post(f"{api_url}/api/main/time", data=data)


@click.group()
@click.option(
    "--url",
    default=DEFAULT_STELLARIUM_URL,
    help=f"URL of Stellarium API. [default: {DEFAULT_STELLARIUM_URL}]",
)
@click.pass_context
def cli(ctx, url):
    """Track Vera Rubin Observatory in Stellarium!

    See subcommand help for details.
    """
    ctx.ensure_object(dict)
    ctx.obj["URL"] = url


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument("lon", type=str)
@click.argument("lat", type=str)
@click.argument("rot", required=False, type=str)
@click.option(
    "--time",
    type=str,
    default=None,
    help="Explicitly set time in stellarium to this.  Accepts any format that"
    " astropy.time.Time can parse together with the --timeformat option."
    " [default: None]",
)
@click.option(
    "--timeformat",
    type=str,
    default=None,
    help="Time format to use.  Passed as the format argument to astropy.time.Time when"
    " explicitly setting the time.",
)
@click.option(
    "--camera",
    type=click.Choice(CAMERA_CHOICES, case_sensitive=True),
    help="Specify camera to slew. [default: LSSTCam]",
    default="LSSTCam",
)
@click.option(
    "--horizon",
    is_flag=True,
    help="Use horizon coordinates (Az/Alt/RotTelPos) instead of equitorial "
    "(RA/Dec/RotSkyPos).",
)
@click.option(
    "--no-follow",
    is_flag=True,
    help="Do not follow the slew with a Stellarium view change.",
)
@click.pass_context
def slew(ctx, lon, lat, rot, time, timeformat, camera, horizon, no_follow):
    """Slew mosaic to given position and rotation.

    LON is the longitudinal angle (RA or Az).

    LAT is the latitudinal angle (Dec or Alt).

    ROT is the rotation angle (RotSkyPos (E of N) or RotTelPos (CCW from +alt)).

    These may be specified in any format that astropy.units.Angle can parse.

    Some examples:

        python tvs.py slew 12:34:56.7 -12:34:56.7 45

        python tvs.py slew 12:34:56.7 -12:34:56.7 45 --horizon

        python tvs.py slew 12h34m56.7 -12d34m56.7 45deg

        python tvs.py slew 12.5h -12.5d 45deg

        python tvs.py slew "185 deg" "12 deg" "0.5 rad" --camera LATISS

        python tvs.py slew 12:34:56.7 -12:34:56.7 45 --no-follow

        python tvs.py slew 0 0 0 --time 2024-04-08T00:00:00

        python tvs.py slew 0 0 0 --time 60408.0 --timeformat mjd
    """
    api_url = ctx.obj["URL"]
    extra_args = ctx.args
    if extra_args:
        lon = extra_args[0] if len(extra_args) > 0 else lon
        lat = extra_args[1] if len(extra_args) > 1 else lat
        rot = extra_args[2] if len(extra_args) > 2 else rot

    camera = "LSSTCam" if camera == "ComCam" else camera

    if time is not None:
        t = Time(time, format=timeformat)
        set_stellarium_time(api_url, t)

    if horizon:
        # Interpret lon/lat/rot as Az/Alt/RotTelPos
        alt = parse_angle(lat)
        az = parse_angle(lon)

        set_camera(api_url, camera)
        data = get_stellarium_attributes(api_url)
        coord = SkyCoord(
            alt=alt,
            az=az,
            frame=AltAz(obstime=data["time"], location=RUBIN_OBSERVATORY),
        )
        q = parallactic_angle(coord, data["time"])
        ra = coord.icrs.ra
        dec = coord.icrs.dec
        if rot is None:  # Keep current rtp
            rtp = data["rtp"]
        else:
            rtp = parse_angle(rot)
        rsp = q - rtp - 90 * u.deg
    else:
        # Interpret lon/lat/rot as RA/Dec/RotSkyPos
        ra = parse_angle(lon, unit=u.hour)
        dec = parse_angle(lat)
        if rot is None:  # Keep current rsp
            rsp = get_stellarium_attributes(api_url)["rsp"]
        else:
            rsp = parse_angle(rot)

    slew_to(api_url, camera, ra, dec, rsp)
    if not no_follow:
        set_view_to_camera(api_url)

    print_state(api_url)


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument("name", type=str)
@click.argument("rot", type=str, required=False)
@click.option(
    "--time",
    type=str,
    default=None,
    help="Explicitly set time in stellarium to this.  Accepts any format that"
    " astropy.time.Time can parse together with the --timeformat option."
    " [default: None]",
)
@click.option(
    "--timeformat",
    type=str,
    default=None,
    help="Time format to use.  Passed as the format argument to astropy.time.Time when"
    " explicitly setting the time.  [default: None]",
)
@click.option(
    "--camera",
    type=click.Choice(CAMERA_CHOICES, case_sensitive=True),
    help="Specify camera to slew. [default: LSSTCam]",
    default="LSSTCam",
)
@click.option(
    "--horizon",
    is_flag=True,
    help="Use horizon rot (RotTelPos) instead of equatorial rot (RotSkyPos).",
)
@click.option(
    "--no-follow",
    is_flag=True,
    help="Do not follow the slew with a Stellarium view change.",
)
@click.pass_context
def target(ctx, name, rot, time, timeformat, camera, horizon, no_follow):
    """Slew to target by name.

    NAME is the name of the target (e.g., 'M20' or 'Trifid Nebula').

    ROT is the rotation angle (RotSkyPos (E of N) or RotTelPos (CCW from +alt)).

    The rotation angle may be specified in any format that astropy.units.Angle can
    parse.

    Some examples:

        python tvs.py target M20

        python tvs.py target M20 45

        python tvs.py target Trifid_Nebula 45 --horizon

        python tvs.py target M20 20deg

        python tvs.py target M20 "20 deg"

        python tvs.py target "Trifid Nebula" 45deg --camera LATISS

        python tvs.py target M20 45 --no-follow

        python tvs.py target M20 --time 2024-04-08T00:00:00

        python tvs.py target M20 --time 60408.0 --timeformat mjd
    """
    from astroquery.simbad import Simbad

    api_url = ctx.obj["URL"]
    extra_args = ctx.args
    if extra_args:
        name = extra_args[0] if len(extra_args) > 0 else name
        rot = extra_args[1] if len(extra_args) > 1 else rot

    if time is not None:
        t = Time(time, format=timeformat)
        set_stellarium_time(api_url, t)

    simbad = Simbad()
    simbad.add_votable_fields("ra", "dec")
    result = simbad.query_object(name)
    if result is None:
        raise ValueError(f"Object {name} not found in Simbad.")
    ra = Angle(result["ra"].data[0] * u.deg)
    dec = Angle(result["dec"].data[0] * u.deg)

    if horizon:
        data = get_stellarium_attributes(api_url)
        time = data["time"]
        coord = SkyCoord(ra=ra, dec=dec, frame=ICRS)
        q = parallactic_angle(coord, time)
        if rot is None:  # Keep current rtp
            rtp = data["rtp"]
        else:
            rtp = parse_angle(rot)
        rsp = q - rtp - 90 * u.deg
    else:
        if rot is None:  # Keep current rsp
            rsp = get_stellarium_attributes(api_url)["rsp"]
        else:
            rsp = parse_angle(rot)

    slew_to(api_url, "LSSTCam" if camera == "ComCam" else camera, ra, dec, rsp)
    if not no_follow:
        set_view_to_camera(api_url)
    print_state(api_url)


@cli.command()
@click.argument("visit", type=int)
@click.option(
    "--camera",
    type=click.Choice(CAMERA_CHOICES, case_sensitive=True),
    help="Specify camera to slew. [default: LSSTCam or inferred from date]",
    default=None,
)
@click.option(
    "--force_consdb",
    is_flag=True,
    help="Force use of ConsDB for coordinates.",
)
@click.option(
    "--rsp-token",
    type=str,
    help=f"Path to RSP token file. [default: {DEFAULT_RSP_TOKEN_PATH}]",
    default=DEFAULT_RSP_TOKEN_PATH,
)
@click.option(
    "--rsp-server",
    type=str,
    help=f"RSP server URL. [default: {DEFAULT_RSP_SERVER}]",
    default=DEFAULT_RSP_SERVER,
)
@click.option(
    "--no-follow",
    is_flag=True,
    help="Do not follow the slew with a Stellarium view change.",
)
@click.option(
    "--rot-offset",
    type=float,
    help="Offset to apply to the RotSkyPos (E of N) angle.",
    default=None,
)
@click.pass_context
def visit(
    ctx, visit, camera, force_consdb, rsp_token, rsp_server, no_follow, rot_offset
):
    """Slew to match given visit ID.

    VISIT is the visit ID to slew to.

    Some examples:

        # Infer camera from date

        # ComCam

        python tvs.py visit 2024121100608

        # LATISS

        python tvs.py visit 2025040300825

        # Explicitly specify camera

        python tvs.py visit 2024121100608 --camera ComCam

        python tvs.py visit 2025040300825 --camera LATISS

        python tvs.py visit 2025040300825 --camera LATISS --no-follow
    """
    api_url = ctx.obj["URL"]

    token = get_access_token(rsp_token)
    consdb = ConsDB(rsp_server, token)

    # Override camera by date if possible:
    if camera is None:
        if visit < 2024102300000:
            print("Inferring camera is LATISS")
            camera = "LATISS"
        elif visit < 2024121200000:
            print("Inferring camera is ComCam")
            camera = "ComCam"
        elif visit < 2025041500000:
            print("Inferring camera is LATISS")
            camera = "LATISS"
        else:
            print("Inferring camera is LSSTCam")
            camera = "LSSTCam"

    if camera == "ComCam":
        database = "cdb_lsstcomcam"
    elif camera == "LSSTCam":
        database = "cdb_lsstcam"
    elif camera == "LATISS":
        database = "cdb_latiss"
    else:
        raise ValueError(f"Unknown camera: {camera}")

    if rot_offset is None:
        if visit > 2025041500000 and visit < 2025042200000:
            rot_offset = Angle("-90d")
        else:
            rot_offset = Angle("0d")
    rot_offset = parse_angle(rot_offset, unit=u.deg)

    ra = None
    if not force_consdb:
        try:
            table = QTable.read("visits.ecsv")
        except FileNotFoundError:
            print("visits.ecsv not found.  Using ConsDB for coordinates.")
        else:
            if visit in table["visit"]:
                print("Using visits.ecsv for coordinates")
                row = table[table["visit"] == visit][0]
                ra = row["ra"]
                dec = row["dec"]
                rsp = row["rsp"]
                time = row["time"]
    if ra is None:
        print("Using ConsDB for coordinates")
        query = f"select * from {database}.visit1 where visit_id = {visit}"
        data = consdb.query(query)[0]
        ra = Angle(data["s_ra"] * u.deg)
        dec = Angle(data["s_dec"] * u.deg)
        rsp = Angle(data["sky_rotation"] * u.deg)
        time = Time(data["exp_midpt_mjd"], format="mjd", scale="tai")

    rsp = rsp + rot_offset
    slew_to(
        api_url,
        "LSSTCam" if camera == "ComCam" else camera,
        ra,
        dec,
        rsp,
    )
    # Set visit time and pause with timerate=0
    set_stellarium_time(api_url, time, timerate=0)
    if not no_follow:
        set_view_to_camera(api_url)
    print_state(api_url)


@cli.command()
@click.argument("day_obs_begin", type=int)
@click.argument("day_obs_end", type=int, required=False)
@click.option(
    "--camera",
    type=click.Choice(CAMERA_CHOICES, case_sensitive=True),
    help="Specify camera to replay. [default: LSSTCam or inferred from date]",
    default=None,
)
@click.option(
    "--rsp-token",
    type=str,
    help=f"Path to RSP token file. [default: {DEFAULT_RSP_TOKEN_PATH}]",
    default=DEFAULT_RSP_TOKEN_PATH,
)
@click.option(
    "--rsp-server",
    type=str,
    help=f"RSP server URL. [default: {DEFAULT_RSP_SERVER}]",
    default=DEFAULT_RSP_SERVER,
)
@click.option(
    "--no-follow",
    is_flag=True,
    help="Do not follow the replay with a Stellarium view change.",
)
@click.pass_context
def replay(
    ctx,
    day_obs_begin,
    day_obs_end,
    camera,
    rsp_token,
    rsp_server,
    no_follow,
):
    """Replay a range of days.

    DAY_OBS_BEGIN and DAY_OBS_END form a range of days (inclusive) to replay.  If
    DAY_OBS_END is omitted, it is set equal to DAY_OBS_BEGIN.
    """
    api_url = ctx.obj["URL"]

    if day_obs_end is None:
        day_obs_end = day_obs_begin
    if day_obs_end < day_obs_begin:
        raise ValueError("DAY_OBS_END must be greater than or equal to DAY_OBS_BEGIN")

    # Try to infer the camera
    if camera is None:
        if day_obs_end < 20241023:
            print("Inferring camera is LATISS")
            camera = "LATISS"
        elif day_obs_end < 20241212 and day_obs_begin > 20241023:
            print("Inferring camera is ComCam")
            camera = "ComCam"
        elif day_obs_end < 20250415 and day_obs_begin > 20241212:
            print("Inferring camera is LATISS")
            camera = "LATISS"
        else:
            print("Inferring camera is LSSTCam")
            camera = "LSSTCam"

    token = get_access_token(rsp_token)
    consdb = ConsDB(rsp_server, token)

    if camera == "ComCam":
        database = "cdb_lsstcomcam"
    elif camera == "LSSTCam":
        database = "cdb_lsstcam"
    elif camera == "LATISS":
        database = "cdb_latiss"
    else:
        raise ValueError(f"Unknown camera: {camera}")

    visits = consdb.query(
        f"select * from {database}.visit1 where day_obs >= {day_obs_begin}"
        f" and day_obs <= {day_obs_end}"
    )

    # Set stellarium time to first visit with can_see_sky True
    visit0 = visits[visits["can_see_sky"]][0]
    set_stellarium_time(
        api_url,
        Time(visit0["exp_midpt_mjd"], format="mjd"),
        timerate=100 / 86400,
    )

    currentindex = None
    while True:
        time = get_stellarium_attributes(api_url)["time"]
        index = np.searchsorted(visits["exp_midpt_mjd"], time.mjd)
        if index == currentindex:
            continue
        currentindex = index
        try:
            visit = visits[index]
        except IndexError:
            continue
        print("\n\n\n")
        print(
            visit[
                [
                    "visit_id",
                    "day_obs",
                    "seq_num",
                    "band",
                    "exp_time",
                    "img_type",
                    "science_program",
                    "observation_reason",
                    "target_name",
                ]
            ]
        )
        print_state(api_url)
        slew_to(
            api_url,
            "LSSTCam" if camera == "ComCam" else camera,
            Angle(visit["s_ra"] * u.deg),
            Angle(visit["s_dec"] * u.deg),
            Angle(visit["sky_rotation"] * u.deg),
        )
        if not no_follow:
            set_view_to_camera(api_url)


def query_efd_retry(efd, topic, time):
    """Query/poll EFD until a valid row is found.

    Parameters
    ----------
    efd : EFD
        EFD client.
    topic : str
        Topic to query.
    time : astropy.time.Time
        Time to query.

    Returns
    -------
    dict
        Mount status.
    """

    while True:
        try:
            out = efd.get_most_recent_row_before(topic, time)
        except (TypeError, ValueError):
            sleep(1.0)
            print("Waiting for efd...")
            continue
        else:
            break
    return out


@cli.command()
@click.argument(
    "camera",
    type=click.Choice(CAMERA_CHOICES, case_sensitive=True),
    default="LSSTCam",
    metavar="CAMERA",
)
@click.option(
    "--rsp-server",
    type=str,
    help=f"RSP server URL. [default: {DEFAULT_RSP_SERVER}]",
    default=DEFAULT_RSP_SERVER,
)
@click.option(
    "--delay",
    type=float,
    help="Delay between updates in seconds. [default: 5.0]",
    default=5.0,
)
@click.option(
    "--no-follow",
    is_flag=True,
    help="Do not follow the replay with a Stellarium view change.",
)
@click.pass_context
def follow(ctx, camera, rsp_server, delay, no_follow):
    """Follow camera pointing in real time with Stellarium.

    CAMERA is the camera to follow (LSSTCam, ComCam, or LATISS).

    To follow both LSSTCam and LATISS, run this command twice with different cameras and
    most likely with --no-follow on one of them.
    """
    api_url = ctx.obj["URL"]

    efd = EFD(rsp_server)

    if camera in ["ComCam", "LSSTCam"]:
        topic = "lsst.sal.MTPtg.mountStatus"
    elif camera == "LATISS":
        topic = "lsst.sal.ATPtg.mountStatus"
    else:
        raise ValueError(f"Unknown camera: {camera}")

    while True:
        tnow = Time.now()
        set_stellarium_time(api_url, tnow, timerate=1 / 86400)
        mount_status = query_efd_retry(efd, topic, tnow)
        try:
            ra = Angle(mount_status["mountRA"] * 15 * u.deg)
            dec = Angle(mount_status["mountDec"] * u.deg)
            rtp = Angle(mount_status["mountRot"] * u.deg)
            tefd = Time(mount_status["timestamp"], format="unix")
        except ValueError:
            print("Waiting for EFD...")
            sleep(delay)
            continue
        coord = SkyCoord(ra=ra, dec=dec)
        q = parallactic_angle(coord, tefd)
        rsp = q - rtp - 90 * u.deg

        slew_to(api_url, "LSSTCam" if camera == "ComCam" else camera, ra, dec, rsp)
        if not no_follow:
            set_view_to_camera(api_url)
        print()
        print_state(api_url)
        sleep(delay)


@cli.command()
@click.argument("camera", type=str, required=False, default="LSSTCam")
@click.pass_context
def status(ctx, camera):
    """Print current Stellarium status."""
    api_url = ctx.obj["URL"]
    set_camera(api_url, camera)
    print_state(api_url)


@cli.command()
@click.option(
    "--rsp-server",
    type=str,
    help=f"RSP server URL. [default: {DEFAULT_RSP_SERVER}]",
    default=DEFAULT_RSP_SERVER,
)
@click.option(
    "--delay",
    type=float,
    help="Delay between updates in seconds. [default: 5.0]",
    default=5.0,
)
@click.pass_context
def dome(ctx, rsp_server, delay):
    """Follow the dome in Stellarium."""
    api_url = ctx.obj["URL"]

    efd = EFD(DEFAULT_RSP_SERVER)

    topic = "lsst.sal.MTDome.azimuth"
    while True:
        tnow = Time.now()
        dome_az = query_efd_retry(efd, topic, tnow)
        az = Angle(dome_az["positionActual"] * u.deg)
        tefd = Time(dome_az["timestamp"], format="unix")
        alt = Angle("50d")
        rtp = Angle("0d")
        coord = SkyCoord(
            alt=alt, az=az, frame=AltAz(obstime=tnow, location=RUBIN_OBSERVATORY)
        )
        ra = coord.icrs.ra
        dec = coord.icrs.dec
        q = parallactic_angle(coord, tnow)
        rsp = q - rtp

        set_camera(api_url, "Dome")
        slew_to(api_url, "Dome", ra, dec, rsp)
        print(f"Time: {tnow.iso}")
        print(f"TEFD: {tefd.iso}")
        print(f"q: {q.deg:.2f} deg")
        print(f"Dome azimuth: {az.deg:.2f} deg")
        print()

        sleep(delay)


if __name__ == "__main__":
    cli()
