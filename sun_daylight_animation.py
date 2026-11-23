#!/usr/bin/env python3
"""Animate the daylight footprint on Earth for three seasonal dates.

The illumination is calculated on a latitude/longitude grid.  For every pixel,
the solar elevation is obtained from the Sun's declination, the pixel latitude,
and the hour angle from the subsolar longitude.  This gives a physically based
terminator while keeping the warm, softly glowing look of the reference image.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.ticker import FixedLocator


@dataclass(frozen=True)
class Season:
    label: str
    declination_deg: float


SEASONS = (
    Season("June solstice", 23.44),
    Season("spring / autumn equinox", 0.0),
    Season("December solstice", -23.44),
)

# Elevation/color stops model the night, astronomical/nautical/civil twilight,
# horizon glow, broad daylight, and the warm patch around the subsolar point.
COLOR_STOPS = (
    (-90.0, "#17001f"),
    (-18.0, "#1b0228"),
    (-12.0, "#27245d"),
    (-6.0, "#5675a1"),
    (-1.0, "#d8e3df"),
    (2.0, "#fff2bc"),
    (18.0, "#ffdf58"),
    (45.0, "#ffc126"),
    (90.0, "#f68b16"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a three-panel, 24-hour global daylight animation."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("sun_daylight_example.mp4"),
        help="Output .mp4 or .gif path (default: beside this script).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=240,
        help="Frames per 24-hour loop (240 = four seconds at 60 FPS).",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=3,
        help="Number of complete 24-hour cycles in the exported video.",
    )
    parser.add_argument("--fps", type=int, default=60, help="Playback frames per second.")
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="Output resolution in DPI; 240 produces 2160 x 3840 portrait 4K.",
    )
    parser.add_argument(
        "--start-hour",
        type=float,
        default=0.0,
        help="GMT hour at the first frame; fractions are allowed.",
    )
    return parser.parse_args()


def subsolar_longitude(utc_hour: float) -> float:
    """Longitude where local solar time is noon, in the [-180, 180) range."""
    return ((15.0 * (12.0 - utc_hour) + 180.0) % 360.0) - 180.0


def solar_elevation(
    lat_rad: np.ndarray,
    lon_rad: np.ndarray,
    declination_deg: float,
    utc_hour: float,
) -> np.ndarray:
    """Return solar elevation in degrees for a lat/lon mesh."""
    declination = np.deg2rad(declination_deg)
    subsolar_lon = np.deg2rad(subsolar_longitude(utc_hour))
    cos_zenith = (
        np.sin(lat_rad) * np.sin(declination)
        + np.cos(lat_rad) * np.cos(declination) * np.cos(lon_rad - subsolar_lon)
    )
    return np.rad2deg(np.arcsin(np.clip(cos_zenith, -1.0, 1.0)))


def elevation_to_rgb(elevation: np.ndarray) -> np.ndarray:
    """Map solar elevation to a custom twilight/daylight RGB gradient."""
    levels = np.asarray([stop[0] for stop in COLOR_STOPS], dtype=float)
    colors = np.asarray([to_rgb(stop[1]) for stop in COLOR_STOPS], dtype=float)
    rgb = np.empty((*elevation.shape, 3), dtype=float)
    for channel in range(3):
        rgb[..., channel] = np.interp(elevation, levels, colors[:, channel])

    # A small luminance lift close to the terminator creates the reference's
    # soft white halo without moving the geometrically correct day/night edge.
    halo = np.exp(-0.5 * (elevation / 4.2) ** 2)[..., None]
    rgb = rgb * (1.0 - 0.14 * halo) + 0.14 * halo
    return np.clip(rgb, 0.0, 1.0)


def format_gmt(utc_hour: float) -> str:
    minutes = int(round((utc_hour % 24.0) * 60.0)) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d} GMT"


def build_animation(args: argparse.Namespace) -> animation.FuncAnimation:
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    if args.loops < 1:
        raise ValueError("--loops must be at least 1")
    if args.fps < 1:
        raise ValueError("--fps must be at least 1")

    # A 0.5-degree calculation grid keeps the terminator smooth while allowing
    # the animation to render quickly on a laptop.
    longitudes = np.linspace(-180.0, 180.0, 721)
    latitudes = np.linspace(-90.0, 90.0, 361)
    lon_grid, lat_grid = np.meshgrid(np.deg2rad(longitudes), np.deg2rad(latitudes))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "medium",
            "axes.titlesize": 13,
        }
    )
    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        3,
        1,
        # 9 x 16 inches at the default 240 DPI gives exact 2160 x 3840
        # portrait 4K output in the standard phone/video aspect ratio.
        figsize=(9.0, 16.0),
        subplot_kw={"projection": projection},
    )
    fig.patch.set_facecolor("#fbfaf6")
    fig.subplots_adjust(left=0.045, right=0.955, top=0.965, bottom=0.045, hspace=0.30)

    start_elevations = [
        solar_elevation(lat_grid, lon_grid, season.declination_deg, args.start_hour)
        for season in SEASONS
    ]
    images = []
    titles = []

    for ax, season, elevation in zip(axes, SEASONS, start_elevations, strict=True):
        ax.set_extent((-180, 180, -90, 90), crs=projection)
        image = ax.imshow(
            elevation_to_rgb(elevation),
            origin="lower",
            extent=(-180, 180, -90, 90),
            transform=projection,
            interpolation="bilinear",
            zorder=0,
        )

        gridlines = ax.gridlines(
            crs=projection,
            draw_labels=False,
            linewidth=0.55,
            color="#6f6d72",
            alpha=0.26,
            linestyle=(0, (2, 3)),
            zorder=1,
        )
        gridlines.xlocator = FixedLocator(np.arange(-180, 181, 30))
        gridlines.ylocator = FixedLocator(np.arange(-60, 61, 30))

        ax.coastlines(resolution="110m", color="#55515a", linewidth=0.62, zorder=3)
        ax.add_feature(
            cfeature.BORDERS.with_scale("110m"),
            edgecolor="#625f62",
            linewidth=0.38,
            alpha=0.88,
            zorder=3,
        )
        ax.spines["geo"].set_edgecolor("#aaa69e")
        ax.spines["geo"].set_linewidth(0.7)

        title = ax.set_title(
            f"Path of the Sun at {format_gmt(args.start_hour)} · {season.label}",
            color="#1f2024",
            pad=10,
        )
        images.append(image)
        titles.append(title)

    fig.text(
        0.5,
        0.017,
        "Deep purple: night   ·   Blue-white: twilight   ·   Gold: daylight",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#77736d",
    )

    def update(frame: int):
        # Reset the solar time after each complete loop. Repeating the first
        # frame at each boundary keeps all three cycles exactly identical.
        cycle_frame = frame % args.frames
        utc_hour = (args.start_hour + 24.0 * cycle_frame / args.frames) % 24.0
        for image, title, season in zip(images, titles, SEASONS, strict=True):
            elevation = solar_elevation(
                lat_grid, lon_grid, season.declination_deg, utc_hour
            )
            image.set_data(elevation_to_rgb(elevation))
            title.set_text(f"Path of the Sun at {format_gmt(utc_hour)} · {season.label}")
        return [*images, *titles]

    return animation.FuncAnimation(
        fig,
        update,
        frames=args.frames * args.loops,
        interval=1000 / args.fps,
        blit=False,
        repeat=True,
    )


def save_animation(movie: animation.FuncAnimation, args: argparse.Namespace) -> None:
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()

    if suffix == ".mp4":
        writer = animation.FFMpegWriter(
            fps=args.fps,
            codec="libx264",
            bitrate=8000,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            metadata={"title": "Global daylight through the seasons"},
        )
    elif suffix == ".gif":
        writer = animation.PillowWriter(fps=args.fps)
    else:
        raise ValueError("Output filename must end in .mp4 or .gif")

    total_frames = args.frames * args.loops
    print(
        f"Rendering {total_frames} frames "
        f"({args.loops} x 24-hour loops) to {output}"
    )
    movie.save(output, writer=writer, dpi=args.dpi)
    plt.close(movie._fig)
    print(f"Done: {output}")


def main() -> None:
    args = parse_args()
    movie = build_animation(args)
    save_animation(movie, args)


if __name__ == "__main__":
    main()
