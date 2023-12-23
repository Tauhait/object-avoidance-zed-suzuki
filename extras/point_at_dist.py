import random, numpy as np, math
import os
import logging
import datetime
import constant as const
import pyzed.sl as sl
from scipy.interpolate import CubicSpline
from math import asin, atan2, cos, degrees, radians, sin, sqrt

def get_point_at_distance(lat1, lon1, d, bearing, R=6371):
    """
    lat: initial latitude, in degrees
    lon: initial longitude, in degrees
    d: target distance from initial
    bearing: (true) heading in degrees
    R: optional radius of sphere, defaults to mean radius of earth

    Returns new lat/lon coordinate {d}km from initial, in degrees
    """
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    a = radians(bearing)
    lat2 = asin(sin(lat1) * cos(d/R) + cos(lat1) * sin(d/R) * cos(a))
    lon2 = lon1 + atan2(
        sin(a) * sin(d/R) * cos(lat1),
        cos(d/R) - sin(lat1) * sin(lat2)
    )
    return (degrees(lat2), degrees(lon2))


if __name__ == '__main__':
    distance = [0.004, 0.001, 0.001, 0.001, 0.001, 0.0005]
    # bearing = [22.24 , 22.24, 22.24, 21.24, 18.24] *****
    
    # bearing = [30.24 , 22.24, 22.24, 21.24, 21.24, 18.24]

    bearing = [22.24 , 18.24, 21.24, 22.24, 19.24, 18.24]

    # Number of waypoints in the sequence
    num_waypoints = const.N

    overtake_lat = 17.602029105524696
    overtake_lon = 78.12708292791199
    print(f"{overtake_lat}, {overtake_lon}")

    for n in range(1, num_waypoints + 1):
        overtake_lat, overtake_lon = get_point_at_distance(overtake_lat, overtake_lon, distance[n], bearing[n])
        print(f"{overtake_lat}, {overtake_lon}")