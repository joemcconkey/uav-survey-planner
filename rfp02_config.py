"""Shared engineering inputs, geometry and route solver for MECH 5005F RFP-02.

Edit the inputs below, then restart the notebook kernel and Run All.
Source references and assumption rationale are in both notebooks.
"""

import math
from collections import namedtuple

import numpy as np
import pandas as pd



# Aircraft and sensor settings
aircraft_with_battery_kg = 9.740
l3_kg = 1.600
l3_connector_kg = 0.145
tb100_kg = 4.720
loaded_reference_min = 53  # H30T + AP100 reference, not an L3 endurance test
airborne_endurance_min = 45
reduced_endurance_min = 35
landing_reserve_fraction = 0.25
flight_height_above_base_m = 100  # Flat proxy for the slant-distance screen
height_min_m, height_max_m = 80, 120
preferred_survey_speed_mps = 10
transit_speed_mps = 12
lidar_pulse_rate_hz = 350_000
lidar_side_overlap = 0.50
lidar_half_fov_deg = 40
rgb_photo_interval_s = 1.0
rgb_side_overlap = 0.70
rgb_forward_overlap = 0.80
rgb_horizontal_fov_deg, rgb_vertical_fov_deg = 62, 41.2
rgb_width_px, rgb_height_px = 12288, 8192
camera_roll_deg = 22.5  # Symmetric split of DJI's 45-degree optical-axis separation
rgb_outer_angle_deg = 50  # Use the central portion of the combined footprint
rgb_gsd_target_cm = 2.5
pulse_density_target_m2 = 8
pulse_spacing_target_m = 0.35

# Route and operations
climb_descent_min = 2.5
run_in_out_m = 25
turn_delay_min = 0.10
c2_slant_limit_m = 3000
bearings_deg = range(0, 180, 15)
cross_grid = False
pieces_per_strip = 1
daily_window_h = 8
ready_battery_sets_per_day = 9     # Nine nominal + one separate contingency TB100; no field recharge
turnaround_min = 7                 # Per completed sortie, including field QA
setup_packdown_min = 35
survey_control_min_per_day = 60     # Include walking to control/checkpoints here
delay_allowance_min_per_day = 25    # General delays; no extra repeat battery reserved
loaded_walk_kmh = 2.5
unloaded_walk_kmh = 3.5
carry_trips_each_way = 1           # Entire crew transports the inventory in one wave
boat_speed_kmh = 4
boat_setup_min = 30                # Extra handling, assumed road-to-shore portage
include_boat_in_selection = False
disabled_sites = set()             # Example: {"LZ-01", "LZ-02"}
trail_fractions = {"LZ-02": 0.25, "LZ-03": 0.50, "LZ-04": 0.75}
site_daily_extra_min = {}          # Optional site-specific allowance, keyed by LZ ID

assert 0 < height_min_m <= flight_height_above_base_m <= height_max_m
assert 0 <= landing_reserve_fraction < 1
assert min(airborne_endurance_min, reduced_endurance_min, preferred_survey_speed_mps,
           transit_speed_mps, daily_window_h, c2_slant_limit_m, loaded_walk_kmh,
           unloaded_walk_kmh, boat_speed_kmh) > 0
assert min(pieces_per_strip, carry_trips_each_way, ready_battery_sets_per_day) >= 1
assert all(0 < fraction < 1 for fraction in trail_fractions.values())
assert all(0 < overlap < 1 for overlap in (rgb_side_overlap, rgb_forward_overlap, lidar_side_overlap))

# Confirmed mission and simulation assumptions
mission_site_id = "LZ-01"
rfp_area_km2 = 2.865
rfp_perimeter_km = 6.869
delivery_limit_business_days = 10
coverage_grid_m = 5
rgb_sensitivity_overlap = 0.75
link_screen_cases_m = (1750, 1850, 2000, 2200, 2350, 2400)
slide_terminus_corner_distances_m = (1717, 1412, 823, 1255)

# Derived sensor geometry and budgeting
camera_roll = np.radians(camera_roll_deg)
theta_inner = camera_roll - np.radians(rgb_horizontal_fov_deg / 2)
theta_outer = np.radians(rgb_outer_angle_deg)
assert theta_inner <= 0 < theta_outer <= camera_roll + np.radians(rgb_horizontal_fov_deg / 2)
half_v_fov = np.radians(rgb_vertical_fov_deg / 2)

lidar_width_m = 2 * height_min_m * np.tan(np.radians(lidar_half_fov_deg))
rgb_camera_width_m = height_min_m * (np.tan(theta_outer) - np.tan(theta_inner))
rgb_combined_width_m = 2 * height_min_m * np.tan(theta_outer)
theta = np.linspace(theta_inner, theta_outer, 201)
rgb_min_along_m = 2 * height_min_m * np.tan(half_v_fov) * np.min(
    np.cos(theta - camera_roll) / np.cos(theta)
)
lidar_spacing_limit_m = lidar_width_m * (1 - lidar_side_overlap)
rgb_spacing_limit_m = rgb_camera_width_m * (1 - rgb_side_overlap)
line_spacing_m = math.floor(min(lidar_spacing_limit_m, rgb_spacing_limit_m))
assert line_spacing_m > 0
rgb_speed_limit_mps = rgb_min_along_m * (1 - rgb_forward_overlap) / rgb_photo_interval_s
survey_speed_mps = min(preferred_survey_speed_mps, rgb_speed_limit_mps)

f_x_px = rgb_width_px / (2 * np.tan(np.radians(rgb_horizontal_fov_deg / 2)))
f_y_px = rgb_height_px / (2 * np.tan(half_v_fov))
down = np.cos(theta_outer) / np.cos(theta_outer - camera_roll)
scale = height_max_m / down**2
projection_jacobian = np.array([
    [scale / f_x_px, 0],
    [scale * np.tan(half_v_fov) * np.sin(camera_roll) / f_x_px, height_max_m / (f_y_px * down)],
])
gsd_max_cm = 100 * np.linalg.svd(projection_jacobian, compute_uv=False)[0]

takeoff_mass_kg = aircraft_with_battery_kg + l3_kg + l3_connector_kg
usable_flight_min = airborne_endurance_min * (1 - landing_reserve_fraction)
# RFP corners, concept-slide digitization and candidate sites
corner_names = list("ABCD")
corners_ll = np.array([
    [57.4930229934, -130.2497217308],
    [57.4887345988, -130.2291135545],
    [57.4709225568, -130.2393209947],
    [57.4778843715, -130.2630132224],
])
origin_ll = corners_ll.mean(axis=0)
latitude = np.deg2rad(origin_ll[0])
a, e2 = 6378137.0, 0.00669437999014
w = np.sqrt(1 - e2 * np.sin(latitude)**2)
metres_per_degree = np.pi / 180 * np.array([
    a * (1 - e2) / w**3, a * np.cos(latitude) / w,
])
corners = ((corners_ll - origin_ll) * metres_per_degree)[:, ::-1]
polygon = np.vstack([corners, corners[0]])

corner_px = np.array([[1111, 272], [1424, 391], [1269, 896], [910, 701]])
trail_px = np.array([  # Highway junction to trail terminus
    [1460, 747], [1427, 729], [1394, 711], [1364, 696],
    [1336, 686], [1310, 677], [1293, 676], [1275, 674],
    [1263, 680], [1243, 681], [1229, 687], [1222, 695],
])
def homogeneous(points):
    return np.column_stack([points, np.ones(len(points))])

transform = np.linalg.lstsq(homogeneous(corner_px), corners, rcond=None)[0]
fit_errors = np.linalg.norm(homogeneous(corner_px) @ transform - corners, axis=1)
trail = homogeneous(trail_px) @ transform

# Main lake outline traced on the 1320 px-wide render of concept slide 11.
# Convert to the 1600 px reference used for the affine fit.
lake_px = np.array([
    [969, 249], [975, 262], [978, 291], [987, 313], [985, 335],
    [992, 349], [1010, 342], [1014, 320], [1012, 298], [1017, 279],
    [1030, 266], [1024, 291], [1024, 312], [1034, 335], [1030, 351],
    [1015, 369], [997, 384], [992, 411], [997, 429], [1011, 439],
    [1030, 435], [1040, 421], [1048, 396], [1058, 386], [1070, 392],
    [1074, 410], [1063, 430], [1060, 462], [1049, 497], [1039, 524],
    [1021, 545], [1009, 561], [1002, 579], [997, 598], [980, 616],
    [957, 626], [931, 628], [908, 624], [889, 625], [871, 616],
    [857, 607], [840, 586], [829, 566], [817, 550], [802, 524],
    [805, 497], [807, 475], [820, 446], [844, 432], [871, 418],
    [876, 393], [893, 370], [902, 347], [909, 325], [909, 299],
    [902, 282], [913, 273], [936, 268], [949, 259],
])
island_px = np.array([
    [888, 492], [895, 501], [895, 516], [891, 530], [883, 539],
    [879, 530], [879, 514], [882, 501],
])
lake_xy = homogeneous(lake_px * (1600 / 1320)) @ transform
island_xy = homogeneous(island_px * (1600 / 1320)) @ transform
trail_station = np.r_[0, np.cumsum(np.linalg.norm(np.diff(trail, axis=0), axis=1))]
boat = (corners[0] + corners[2]) / 2
boat_distance = np.linalg.norm(boat - trail[-1])
Site = namedtuple("Site", "xy walk_m paddle_m")
parking_id, terminus_id, boat_id = "LZ-01", "LZ-05", "LZ-06"
site_names = {parking_id: "Hwy 37 parking lot"}
site_names.update({site_id: f"Trail {fraction:.0%}" for site_id, fraction in trail_fractions.items()})
site_names.update({terminus_id: "Trail terminus", boat_id: "Boat proxy"})
sites = {parking_id: Site(trail[0], 0, 0)}
for site_id, fraction in trail_fractions.items():
    station = fraction * trail_station[-1]
    xy = np.array([np.interp(station, trail_station, trail[:, axis]) for axis in range(2)])
    sites[site_id] = Site(xy, station, 0)
sites[terminus_id] = Site(trail[-1], trail_station[-1], 0)
sites[boat_id] = Site(boat, trail_station[-1], boat_distance)


def site_label(site_id, include_prefix=True):
    return f"{site_id} · {site_names[site_id]}" if include_prefix else site_names[site_id]


area = abs(np.sum(polygon[:-1,0]*polygon[1:,1] - polygon[1:,0]*polygon[:-1,1])) / 2
perimeter = np.linalg.norm(np.diff(polygon, axis=0), axis=1).sum()

# Routing and scheduling
def scan_grid(bearing, spacing):
    angle = np.deg2rad(bearing)
    axis = np.array([np.sin(angle), np.cos(angle)])
    across = np.array([axis[1], -axis[0]])
    local = polygon @ np.column_stack([axis, across])
    low, high = local[:, 1].min(), local[:, 1].max()
    count = math.ceil((high - low) / spacing)
    boundaries = np.linspace(low, high, count + 1)
    strips = []
    for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        y = (lower + upper) / 2
        x_values = list(local[(local[:, 1] >= lower) & (local[:, 1] <= upper), 0])
        for (x0, y0), (x1, y1) in zip(local[:-1], local[1:]):
            for boundary in (lower, upper):
                if min(y0, y1) < boundary < max(y0, y1):
                    x_values.append(x0 + (x1 - x0) * (boundary - y0) / (y1 - y0))
        ends = [min(x_values), max(x_values)]
        if index % 2:
            ends.reverse()
        edges = np.linspace(*ends, num=pieces_per_strip + 1)
        direction = np.sign(ends[1] - ends[0])
        for start, end in zip(edges[:-1], edges[1:]):
            survey_endpoints = np.array([start - direction * run_in_out_m,
                                         end + direction * run_in_out_m])
            strips.append(survey_endpoints[:, None] * axis + y * across)
    return np.array(strips)


def access_minutes(site):
    loaded_km = 2 * carry_trips_each_way * site.walk_m / 1000
    unloaded_km = 2 * (carry_trips_each_way - 1) * site.walk_m / 1000
    walk_min = 60 * (loaded_km / loaded_walk_kmh + unloaded_km / unloaded_walk_kmh)
    paddle_min = 60 * 2 * site.paddle_m / 1000 / boat_speed_kmh
    return walk_min + paddle_min + (boat_setup_min if site.paddle_m else 0)


def sortie_times(strips, site, endurance, slant_limit):
    start, end = strips[:,0], strips[:,1]
    distance = np.linalg.norm(strips - site.xy, axis=-1)
    range_m = float(np.hypot(distance.max(), flight_height_above_base_m))
    if range_m > slant_limit:
        return None, range_m
    survey = np.linalg.norm(end-start, axis=1) / (60*survey_speed_mps)
    connectors = np.linalg.norm(start[1:]-end[:-1], axis=1) / (60*transit_speed_mps) + turn_delay_min
    prefix = np.r_[0, np.cumsum(survey + np.r_[0, connectors])]
    n = len(strips)
    times = np.full((n, n+1), np.inf)
    for first in range(n):
        last = np.arange(first, n)
        strip_time = prefix[last+1] - prefix[first]
        if first:
            strip_time -= connectors[first-1]
        transit = (distance[first,0] + distance[last,1]) / (60*transit_speed_mps)
        flight = strip_time + transit + climb_descent_min
        times[first,last+1] = np.where(flight <= endurance*(1-landing_reserve_fraction),
                                       flight, np.inf)
    return times, range_m


def schedule(times, daily_fixed):
    n = len(times)
    available = 60*daily_window_h - daily_fixed
    best = {0: (0, 0., [])}  # covered segments -> days, elapsed minutes, daily sortie lists
    for first in range(n):
        if first not in best:
            continue
        days, elapsed, history = best[first]
        if n in best and days >= best[n][0]:
            continue
        current = {first: (0., [])}
        for battery in range(ready_battery_sets_per_day):
            next_sortie = {}
            for start, (used, sorties) in current.items():
                possible = np.flatnonzero(times[start] + turnaround_min + used <= available)
                for end in possible:
                    flight = times[start,end]
                    minutes = used + flight + turnaround_min
                    if end not in next_sortie or minutes < next_sortie[end][0]:
                        next_sortie[end] = (minutes, sorties + [(start,int(end),flight)])
            if not next_sortie:
                break
            for end, (minutes, sorties) in next_sortie.items():
                score = (days + 1, elapsed + daily_fixed + minutes)
                if end not in best or score < best[end][:2]:
                    best[end] = (*score, history + [sorties])
            current = {end: item for end, item in next_sortie.items() if end < n}
        if first == 0 and n in best:
            break
    return best.get(n)


def compare(spacing, endurance, slant_limit=c2_slant_limit_m):
    winners, rows = {}, []
    routes = {}
    for bearing in bearings_deg:
        grid = scan_grid(bearing, spacing)
        if cross_grid:
            grid = np.concatenate([grid, scan_grid(bearing+90, spacing)])
        routes[bearing] = grid
    for name in sites:
        site = sites[name]
        access = access_minutes(site)
        fixed = access + setup_packdown_min + survey_control_min_per_day + delay_allowance_min_per_day
        fixed += site_daily_extra_min.get(name, 0)
        minimum_range = np.inf
        within_range = False
        choice = None
        for bearing, grid in routes.items():
            for reverse in (False, True):
                route = grid[::-1,::-1] if reverse else grid
                times, reach = sortie_times(route, site, endurance, slant_limit)
                minimum_range = min(minimum_range, reach)
                if times is None:
                    continue
                within_range = True
                result = schedule(times, fixed)
                if result is not None and (choice is None or result[:2] < choice["result"][:2]):
                    choice = dict(result=result, route=route, bearing=bearing, reach=reach, fixed=fixed)
        eligible = name not in disabled_sites and (name != boat_id or include_boat_in_selection)
        row = dict(site=name, location=site_names[name], selectable=eligible, access_min=access, days=np.nan,
                   field_h=np.nan, sorties=np.nan, flight_min=np.nan, peak_day_h=np.nan,
                   bearing_deg=np.nan, slant_m=minimum_range,
                   status="Daily time or airborne budget" if within_range else "Slant range screen")
        if choice:
            days, minutes, history = choice["result"]
            day_flights = [sum(flight for _,_,flight in day) for day in history]
            day_times = [fixed + flight + len(day)*turnaround_min
                         for flight, day in zip(day_flights, history)]
            row.update(days=days, field_h=minutes/60, sorties=sum(map(len,history)),
                       flight_min=sum(day_flights), peak_day_h=max(day_times)/60,
                       bearing_deg=choice["bearing"], slant_m=choice["reach"], status="Fits scenario")
            winners[name] = choice
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["days","field_h"]), winners
# r04 mission-only settings (legacy optimizer remains unchanged)
active_aircraft = 1
reserve_aircraft = 1
reserve_batteries = 1
shoreline_water_buffer_m = 60  # provisional RGB context, not client specified
exclude_deep_water = True
lidar_scan_mode = "Star-shaped"
lidar_returns = 16
imu_allowance_min_per_sortie = 1.0  # placeholder calibration allowance
