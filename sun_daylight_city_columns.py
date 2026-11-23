#!/usr/bin/env python3
"""Landscape 4K daylight animation with Sydney and Taipei sunlight gauges.

The centre column shows the same three seasonal daylight maps as the portrait
version.  Each outer column contains gauges aligned with those maps.  Relative
sunlight is the direct-beam projection onto a horizontal surface:

    relative intensity = max(0, sin(solar elevation))

The displayed W/m² value is an intentionally simple clear-sky estimate using
1000 W/m² at a 90-degree solar elevation; it is not a weather forecast.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.ticker import FixedLocator

from sun_daylight_animation import (
    SEASONS,
    elevation_to_rgb,
    format_gmt,
    solar_elevation,
)


@dataclass(frozen=True)
class City:
    name: str
    coordinate_label: str
    latitude_deg: float
    longitude_deg: float


SYDNEY = City("SYDNEY", "33.9° S · 151.2° E", -33.8688, 151.2093)
TAIPEI = City("TAIPEI", "25.0° N · 121.6° E", 25.0330, 121.5654)

FIGURE_BG = "#090c13"
CARD_BG = "#121824"
CARD_EDGE = "#30394a"
MUTED = "#8c96a8"
WHITE = "#f5f6f8"
NIGHT = "#62708e"
SUN_GOLD = "#ffbd35"
SUN_ORANGE = "#ff8a1f"
SYDNEY_COLOR = "#45c7ff"
TAIPEI_COLOR = "#ff6577"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 4K landscape daylight animation with city gauges."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("sun_daylight_cities_4k_60fps_3loops.mp4"),
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=240,
        help="Frames per 24-hour cycle (240 = four seconds at 60 FPS).",
    )
    parser.add_argument("--loops", type=int, default=3, help="Number of 24-hour cycles.")
    parser.add_argument("--fps", type=int, default=60, help="Output frame rate.")
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="240 DPI with the 16 x 9 figure produces exact 3840 x 2160 output.",
    )
    parser.add_argument(
        "--start-hour",
        type=float,
        default=11.4167,
        help="GMT hour at the first frame; default is 11:25 GMT.",
    )
    return parser.parse_args()


def city_solar_elevation(city: City, declination_deg: float, utc_hour: float) -> float:
    lat = np.asarray([[np.deg2rad(city.latitude_deg)]])
    lon = np.asarray([[np.deg2rad(city.longitude_deg)]])
    return float(solar_elevation(lat, lon, declination_deg, utc_hour)[0, 0])


def sunlight_metrics(elevation_deg: float) -> tuple[float, int]:
    relative = max(0.0, float(np.sin(np.deg2rad(elevation_deg))))
    irradiance = int(round(1000.0 * relative))
    return relative, irradiance


def mix_color(start: str, end: str, amount: float) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    a = np.asarray(to_rgb(start))
    b = np.asarray(to_rgb(end))
    return tuple(a * (1.0 - amount) + b * amount)


def intensity_color(relative: float) -> tuple[float, float, float]:
    if relative <= 0.0:
        return mix_color(NIGHT, NIGHT, 0.0)
    return mix_color(SUN_GOLD, SUN_ORANGE, min(1.0, relative))


def status_for(elevation_deg: float) -> str:
    if elevation_deg > 0.0:
        return "DAYLIGHT"
    if elevation_deg > -18.0:
        return "TWILIGHT"
    return "NIGHT"


def create_gauge(ax: plt.Axes, season_label: str) -> dict[str, object]:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    card = FancyBboxPatch(
        (0.025, 0.035),
        0.95,
        0.91,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        transform=ax.transAxes,
        facecolor=CARD_BG,
        edgecolor=CARD_EDGE,
        linewidth=1.0,
        zorder=0,
    )
    ax.add_patch(card)

    ax.text(
        0.08,
        0.86,
        season_label.upper(),
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8.5,
        weight="bold",
        va="center",
    )
    status = ax.text(
        0.92,
        0.86,
        "NIGHT",
        transform=ax.transAxes,
        color=NIGHT,
        fontsize=8.5,
        weight="bold",
        ha="right",
        va="center",
    )
    value = ax.text(
        0.08,
        0.64,
        "0%",
        transform=ax.transAxes,
        color=WHITE,
        fontsize=35,
        weight="bold",
        va="center",
    )
    ax.text(
        0.085,
        0.51,
        "RELATIVE SUNLIGHT",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8,
        weight="bold",
        va="center",
    )

    bar_x, bar_y, bar_width, bar_height = 0.08, 0.405, 0.84, 0.042
    ax.add_patch(
        FancyBboxPatch(
            (bar_x, bar_y),
            bar_width,
            bar_height,
            boxstyle="round,pad=0,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor="#252d3b",
            edgecolor="none",
        )
    )
    fill = Rectangle(
        (bar_x, bar_y),
        0.001,
        bar_height,
        transform=ax.transAxes,
        facecolor=NIGHT,
        edgecolor="none",
        zorder=3,
    )
    ax.add_patch(fill)
    marker = Circle(
        (bar_x, bar_y + bar_height / 2),
        radius=0.014,
        transform=ax.transAxes,
        facecolor=NIGHT,
        edgecolor=WHITE,
        linewidth=0.8,
        zorder=4,
    )
    ax.add_patch(marker)

    elevation = ax.text(
        0.08,
        0.265,
        "SUN ELEVATION  −90.0°",
        transform=ax.transAxes,
        color=WHITE,
        fontsize=9.5,
        weight="medium",
        va="center",
    )
    irradiance = ax.text(
        0.08,
        0.155,
        "CLEAR-SKY EST.  ≈ 0 W/m²",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8.5,
        va="center",
    )

    return {
        "status": status,
        "value": value,
        "fill": fill,
        "marker": marker,
        "elevation": elevation,
        "irradiance": irradiance,
        "bar_x": bar_x,
        "bar_width": bar_width,
    }


def update_gauge(gauge: dict[str, object], elevation_deg: float) -> list[object]:
    relative, irradiance = sunlight_metrics(elevation_deg)
    color = intensity_color(relative)
    status = status_for(elevation_deg)
    bar_x = float(gauge["bar_x"])
    bar_width = float(gauge["bar_width"])

    gauge["status"].set_text(status)
    gauge["status"].set_color(color)
    gauge["value"].set_text(f"{relative * 100:.0f}%")
    gauge["value"].set_color(color if relative > 0.0 else WHITE)
    gauge["fill"].set_width(max(0.001, bar_width * relative))
    gauge["fill"].set_facecolor(color)
    gauge["marker"].center = (bar_x + bar_width * relative, 0.426)
    gauge["marker"].set_facecolor(color)
    gauge["elevation"].set_text(f"SUN ELEVATION  {elevation_deg:+05.1f}°")
    gauge["irradiance"].set_text(f"CLEAR-SKY EST.  ≈ {irradiance:,d} W/m²")

    return [
        gauge["status"],
        gauge["value"],
        gauge["fill"],
        gauge["marker"],
        gauge["elevation"],
        gauge["irradiance"],
    ]


def build_animation(args: argparse.Namespace) -> animation.FuncAnimation:
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    if args.loops < 1:
        raise ValueError("--loops must be at least 1")
    if args.fps < 1:
        raise ValueError("--fps must be at least 1")

    longitudes = np.linspace(-180.0, 180.0, 721)
    latitudes = np.linspace(-90.0, 90.0, 361)
    lon_grid, lat_grid = np.meshgrid(np.deg2rad(longitudes), np.deg2rad(latitudes))

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(16.0, 9.0), facecolor=FIGURE_BG)
    grid = fig.add_gridspec(
        3,
        3,
        left=0.018,
        right=0.982,
        top=0.918,
        bottom=0.032,
        width_ratios=(1.0, 1.08, 1.0),
        hspace=0.12,
        wspace=0.038,
    )

    projection = ccrs.PlateCarree()
    map_axes = [fig.add_subplot(grid[row, 1], projection=projection) for row in range(3)]
    left_axes = [fig.add_subplot(grid[row, 0]) for row in range(3)]
    right_axes = [fig.add_subplot(grid[row, 2]) for row in range(3)]

    fig.text(
        0.17,
        0.963,
        SYDNEY.name,
        ha="center",
        va="center",
        color=SYDNEY_COLOR,
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.17,
        0.938,
        SYDNEY.coordinate_label,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=8,
    )
    time_text = fig.text(
        0.50,
        0.956,
        format_gmt(args.start_hour),
        ha="center",
        va="center",
        color=WHITE,
        fontsize=17,
        weight="bold",
    )
    fig.text(
        0.50,
        0.930,
        "GLOBAL DAYLIGHT · THREE SEASONS",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=8,
        weight="bold",
    )
    fig.text(
        0.83,
        0.963,
        TAIPEI.name,
        ha="center",
        va="center",
        color=TAIPEI_COLOR,
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.83,
        0.938,
        TAIPEI.coordinate_label,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=8,
    )

    images: list[object] = []
    left_gauges: list[dict[str, object]] = []
    right_gauges: list[dict[str, object]] = []

    for ax, season in zip(map_axes, SEASONS, strict=True):
        elevation = solar_elevation(
            lat_grid, lon_grid, season.declination_deg, args.start_hour
        )
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
            linewidth=0.45,
            color="#646977",
            alpha=0.28,
            linestyle=(0, (2, 3)),
            zorder=1,
        )
        gridlines.xlocator = FixedLocator(np.arange(-180, 181, 30))
        gridlines.ylocator = FixedLocator(np.arange(-60, 61, 30))
        ax.coastlines(resolution="110m", color="#56545c", linewidth=0.55, zorder=3)
        ax.add_feature(
            cfeature.BORDERS.with_scale("110m"),
            edgecolor="#605d63",
            linewidth=0.33,
            alpha=0.9,
            zorder=3,
        )
        ax.plot(
            SYDNEY.longitude_deg,
            SYDNEY.latitude_deg,
            marker="o",
            markersize=3.6,
            markerfacecolor=SYDNEY_COLOR,
            markeredgecolor=WHITE,
            markeredgewidth=0.55,
            transform=projection,
            zorder=5,
        )
        ax.plot(
            TAIPEI.longitude_deg,
            TAIPEI.latitude_deg,
            marker="o",
            markersize=3.6,
            markerfacecolor=TAIPEI_COLOR,
            markeredgecolor=WHITE,
            markeredgewidth=0.55,
            transform=projection,
            zorder=5,
        )
        ax.spines["geo"].set_edgecolor("#5e6470")
        ax.spines["geo"].set_linewidth(0.7)
        ax.text(
            0.5,
            0.955,
            season.label.upper(),
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=WHITE,
            fontsize=7.5,
            weight="bold",
            zorder=6,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": FIGURE_BG,
                "edgecolor": "none",
                "alpha": 0.76,
            },
        )
        images.append(image)

    for ax, season in zip(left_axes, SEASONS, strict=True):
        left_gauges.append(create_gauge(ax, season.label))
    for ax, season in zip(right_axes, SEASONS, strict=True):
        right_gauges.append(create_gauge(ax, season.label))

    def update(frame: int):
        utc_hour = (args.start_hour + 24.0 * frame / args.frames) % 24.0
        time_text.set_text(format_gmt(utc_hour))
        artists: list[object] = [time_text]

        for image, left_gauge, right_gauge, season in zip(
            images, left_gauges, right_gauges, SEASONS, strict=True
        ):
            map_elevation = solar_elevation(
                lat_grid, lon_grid, season.declination_deg, utc_hour
            )
            image.set_data(elevation_to_rgb(map_elevation))
            artists.append(image)

            sydney_elevation = city_solar_elevation(
                SYDNEY, season.declination_deg, utc_hour
            )
            taipei_elevation = city_solar_elevation(
                TAIPEI, season.declination_deg, utc_hour
            )
            artists.extend(update_gauge(left_gauge, sydney_elevation))
            artists.extend(update_gauge(right_gauge, taipei_elevation))

        return artists

    return animation.FuncAnimation(
        fig,
        update,
        frames=args.frames,
        interval=1000 / args.fps,
        blit=False,
        repeat=False,
    )


def save_animation(movie: animation.FuncAnimation, args: argparse.Namespace) -> None:
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("The city-column output filename must end in .mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sun_daylight_") as temp_dir:
        one_cycle = Path(temp_dir) / "one_cycle.mp4"
        writer = animation.FFMpegWriter(
            fps=args.fps,
            codec="libx264",
            bitrate=10000,
            extra_args=[
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-g",
                str(args.frames),
            ],
            metadata={"title": "Global daylight with Sydney and Taipei sunlight"},
        )
        print(f"Rendering one {args.frames}-frame 24-hour cycle at 4K...")
        movie.save(one_cycle, writer=writer, dpi=args.dpi)
        plt.close(movie._fig)

        print(f"Repeating the cycle {args.loops} times...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-stream_loop",
                str(args.loops - 1),
                "-i",
                str(one_cycle),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )

    print(f"Done: {output}")


def main() -> None:
    args = parse_args()
    movie = build_animation(args)
    save_animation(movie, args)


if __name__ == "__main__":
    main()
