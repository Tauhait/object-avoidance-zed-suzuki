from math import asin, atan2, cos, degrees, radians, sin
import util

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


def calc_avg_lat_lng(coord_list):
    lat = 0
    lng = 0
    len_pts = len(coord_list)
    lat_tot = 0
    lng_tot = 0
    for coord in coord_list:
        lat_tmp, lng_tmp = coord[0], coord[1]
        lat_tot += lat_tmp
        lng_tot += lng_tmp
        
    lat = lat_tot / len_pts
    lng = lng_tot / len_pts
    return lat, lng

# Starting point
lat, lon = calc_avg_lat_lng(util.get_coordinates("/Users/TUHIN/repos/object-avoidance-zed-suzuki/code/waypoints_2023-12-18-12_30 OP pt1.txt"))

# Distance between points
distance = 3.625 

# Bearing for the first waypoint
bearing = 86.24 

# Number of waypoints in the sequence
num_waypoints = 10


# Generate a sequence of points with the first waypoint having a bearing of given degrees
# for i in range(num_waypoints):
#     lat, lon = get_point_at_distance(lat, lon, distance, bearing if i == 0 else 0)
#     print(lat, lon)

# Open a file in write mode
pt = "pt1"
filename = f"waypoints_output_generated_{pt}.txt"
with open(filename, 'w') as file:
    # for i in range(num_waypoints):
    #     lat, lon = get_point_at_distance(lat, lon, distance, bearing if i == 0 else 0)
    #     file.write(f"{lat}, {lon}\n")
    gen_lat, gen_lon = get_point_at_distance(lat, lon, distance, bearing)
    file.write(f"[{gen_lat}, {gen_lon}]")

