NHTS_LOC_TYPES = {
    1: "Home",
    2: "Work from Home",        # Home
    3: "Work",                  # Work
    4: "Work-related",          # Work
    5: "Volunteer",             # Recreation
    6: "Drop-off/Pick-up",      # Other
    7: "Change transport",      # Other
    8: "School-student",        # School
    9: "Child care",            # Other
    10: "Adult care",           # Other
    11: "Buy goods",            # Grocer
    12: "Buy services",         # Grocer
    13: "Buy meals",            # Restau
    14: "General",              # Other
    15: "Recreation",           # Recreation
    16: "Exercise",             # Recreation
    17: "Visit friends",        # Other
    18: "Health care",          # Other
    19: "Religious",            # Other
    97: "Other",                # Other
}

COMMON_LOC_TYPES = {
    1: "Home",
    2: "Work",
    3: "Education",
    4: "Commercial",
    5: "Recreation",
    6: "Social",
    7: "Other",
}

NEW_LOC_TYPES = {
    1: "Home",
    2: "Work",
    3: "Eat Meal",
    4: "Education",
    5: "Recreational",
    6: "Shopping",
    7: "Care",
    8: "Community",
    9: "Other",
    10: "Social Visit",
}

# Canonical NHTS WHYTO -> new runtime action taxonomy (10 categories).
# Notes:
# - WHYTO=9 ("Attend child care") maps to Education by design.
# - WHYTO=6/7 (drop-off / transport-change) map to Other for now.
NHTS_WHYTO_TO_NEW_ACTION = {
    1: 1,    # Regular home activities
    2: 1,    # Work from home
    3: 2,    # Work
    4: 2,    # Work-related meeting/trip
    5: 8,    # Volunteer activities -> Community
    6: 9,    # Drop off / pick up someone -> Other
    7: 9,    # Change type of transportation -> Other
    8: 4,    # Attend school as a student -> Education
    9: 4,    # Attend child care -> Education
    10: 7,   # Attend adult care -> Care
    11: 6,   # Buy goods -> Shopping
    12: 6,   # Buy services -> Shopping
    13: 3,   # Buy meals -> Eat Meal
    14: 9,   # Other general errands -> Other
    15: 5,   # Recreational activities -> Recreational
    16: 5,   # Exercise -> Recreational
    17: 10,  # Visit friends or relatives -> Social Visit
    18: 7,   # Health care visit -> Care
    19: 8,   # Religious or community -> Community
    97: 9,   # Something else -> Other
    -7: 9,
    -8: 9,
    -9: 9,
}

# Placeholder crosswalks for deferred eval migration work.
# These will be finalized when scorecards / POL comparison are updated.
NEW_ACTION_TO_LEGACY6 = {
    1: 1,   # Home -> Home
    2: 2,   # Work -> Work
    3: 3,   # Eat Meal -> Restaurant
    4: 4,   # Education -> School
    5: 5,   # Recreational -> Recreation
    6: 7,   # Shopping -> Other/Errands
    7: 7,   # Care -> Other/Errands
    8: 7,   # Community -> Other/Errands
    9: 7,   # Other -> Other/Errands
    10: 7,  # Social Visit -> Other/Errands
}
NEW_ACTION_TO_POL = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 7,
    7: 7,
    8: 7,
    9: 7,
    10: 7,
}

# Parse simulation activity_log `location_type` labels from both
# the new 10-category runtime and legacy 6-category runtime into the
# canonical new10 analysis taxonomy, then optionally collapse further.
SIM_LOG_LABEL_TO_NEW_ACTION_NEW_RUNTIME = {
    "home": 1,
    "work": 2,
    "eat_meal": 3,
    "education": 4,
    "recreational": 5,
    "shopping": 6,
    "care": 7,
    "community": 8,
    "other": 9,
    "social_visit": 10,
}
SIM_LOG_LABEL_TO_NEW_ACTION_LEGACY_RUNTIME = {
    "residential": 1,
    "work": 2,
    "restaurant": 3,
    "school": 4,
    "recreation": 5,
    "other": 9,
}
SIM_LOG_LABEL_TO_NEW_ACTION = {
    **SIM_LOG_LABEL_TO_NEW_ACTION_LEGACY_RUNTIME,
    **SIM_LOG_LABEL_TO_NEW_ACTION_NEW_RUNTIME,
}

LEGACY6_STATE_CODES = (1, 2, 3, 4, 5, 7)
NEW10_STATE_CODES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

DEFAULT_TEMPORAL_ACTIVITY_CODES_LEGACY6 = (2, 4)
DEFAULT_TEMPORAL_ACTIVITY_CODES_NEW10 = (2, 4)

SIM_LOC_TYPES = {
    1: "Home",
    2: "Work",
    3: "Restaurant",
    4: "School",
    5: "Recreation",
    6: "NA",
    7: "Errands",
}

# Canonical NHTS purpose -> MMv4 simulation activity mapping.
# Keep this aligned with the scorecard / realism contract evaluation.
NHTS_TO_SIM_ACTIVITY = {
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 5,
    6: 7,
    7: 7,
    8: 4,
    9: 7,
    10: 7,
    11: 7,
    12: 7,
    13: 3,
    14: 7,
    15: 5,
    16: 5,
    17: 7,
    18: 7,
    19: 7,
    97: 7,
    -7: 7,
    -8: 7,
    -9: 7,
}

# Simulation activity code -> dwell calibration key.
SIM_ACTIVITY_TO_DWELL_KEY = {
    1: "home",
    2: "work",
    3: "restaurant",
    4: "school",
    5: "recreation",
    7: "errands",
}

AGENT_TYPE_LABELS = {
    1: "Worker",
    2: "Student",
}

NHTS_MODE_OF_TRANSPORT = {
    1: "Walk",
    2: "Bicycle",
    3: "Car",
    4: "SUV",
    5: "Van",
    6: "Pickup truck",
    7: "Golf cart / Segway",
    8: "Motorcycle / Moped",
    9: "RV (motor home, ATV, snowmobile)",
    10: "School bus",
    11: "Public or commuter bus",
    12: "Paratransit / Dial-a-ride",
    13: "Private / Charter / Tour / Shuttle bus",
    14: "City-to-city bus (Greyhound, Megabus)",
    15: "Amtrak / Commuter rail",
    16: "Subway / elevated / light rail / street car",
    17: "Taxi / limo (including Uber / Lyft)",
    18: "Rental car (Including Zipcar / Car2Go)",
    19: "Airplane",
    20: "Boat / ferry / water taxi",
    97: "Other",
}
