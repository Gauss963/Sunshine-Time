# Sun daylight animation

This folder contains a Python recreation of the three-panel daylight animation
shown in `IMG_7056.jpeg`. The terminator is calculated from spherical solar
geometry; the colors are then styled to match the warm day / blue twilight /
purple night look of the reference. The default output is portrait 4K
(2160 x 3840), 60 FPS, and contains three complete 24-hour cycles. Each cycle
lasts four seconds, so the exported video is twelve seconds long.

Run it in the requested conda environment:

```bash
conda run --no-capture-output -n General \
  python code/sun_daylight_animation.py
```

Useful options:

```bash
# A GIF instead of MP4
conda run --no-capture-output -n General \
  python code/sun_daylight_animation.py \
  --output code/sun_daylight.gif --frames 240 --loops 3 --fps 60

# Start at the time seen near the reference screenshot
conda run --no-capture-output -n General \
  python code/sun_daylight_animation.py --start-hour 11.4167
```

The main controls are `--frames` (per cycle), `--loops`, `--fps`, `--dpi`,
`--start-hour`, and `--output`. Run with `--help` for the complete list.

## Landscape version with Sydney and Taipei

`sun_daylight_city_columns.py` creates an additional landscape 4K video. The
left column shows Sydney's sunlight, the centre shows the seasonal world maps,
and the right column shows Taipei's sunlight. Its defaults are 3840 x 2160,
60 FPS, four seconds per 24-hour cycle, and three cycles per video.

```bash
conda run --no-capture-output -n General \
  python code/sun_daylight_city_columns.py
```

The percentage is relative direct sunlight on a horizontal surface, calculated
as `max(0, sin(solar elevation))`. The W/m² number is a simple clear-sky
estimate rather than live weather data.
# Sunshine-Time
