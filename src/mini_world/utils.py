import numpy as np


# --- Defining function that gets the travel ticks
def travel_time(bfrom, bto, db, speed):

    # --- Getting coordinates
    xfrom = db[bfrom]["x"]
    yfrom = db[bfrom]["y"]
    xto = db[bto]["x"]
    yto = db[bto]["y"]

    # --- Getting the distance
    distance = np.sqrt((xfrom - xto)**2 + (yfrom - yto)**2)

    # --- Getting velocity in meters per minute
    velocity = speed * (100.0 / 6.0)

    # --- Getting travel time in minutes
    ttime = distance / velocity

    # --- Getting ticks
    tticks = round(ttime / 5.0) + 1

    # --- Returning
    return tticks


# --- Creating function that gets the dwell time
def getting_dwell_time(btype, params):

    # --- Checking bytpe
    if (btype == 1):

        dwell_ticks = round(params["time_spent_at_home"] / 5.0) + 1

    elif (btype == 2):

        dwell_ticks = round(params["time_spent_at_work"] / 5.0) + 1

    elif (btype == 3):

        dwell_ticks = round(params["time_spent_at_restaurant"] / 5.0) + 1

    elif (btype == 4):

        dwell_ticks = round(params["time_spent_at_school"] / 5.0) + 1

    elif (btype == 5):

        dwell_ticks = round(params["time_spent_at_recreation"] / 5.0) + 1

    elif (btype == 7):

        dwell_ticks = round(params["time_spent_at_errands"] / 5.0) + 1

    else:

        print("Error assigning time!")
        raise (SystemExit)

    # --- Returning
    return dwell_ticks
