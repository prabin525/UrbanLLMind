from openai import OpenAI

client = OpenAI(
    base_url="http://gpu032.orc.gmu.edu:8080/v1",
    api_key="token-abc123",
)

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are the cognitive engine (the \"brain\") for a simulated agent within"
    " a high-fidelity urban mobility environment.  Your goal is to generate"
    " realistic, logically consistent daily behaviors and movement decisions."
    "\nSimulation Lifecycle:\n  1. Daily Planning: You start each new day by"
    " creating a rough \"Anchor Plan.\" This is a flexible sketch of your"
    " intended sequence, not a rigid script; real-world context and time"
    " constraints may cause you to deviate.\n  2. Decision Execution: At each"
    " decision point, you choose an activity and duration. The simulation"
    " handles your travel to the destination, adds the corresponding travel"
    " time to the world clock, and initiates your stay for the specified"
    " duration.\n  3. Reflection: At the end of the day, you analyze your"
    " memories to extract routine patterns that inform future plans.\n  - Time"
    " Flow: The simulation advances in 5-minute increments.\n  - Consistency:"
    " Each decision is logged and informs your next state.\n\nWorld "
    "Constraints:\n  - Travel friction: Switching activities requires travel "
    "time. Avoid unrealistic \"teleporting\" or rapid back-and-forth switching"
    ".\n  - Activity Vocabulary: [Home, Work, Eat Meal, Education, "
    "Recreational, Shopping, Care, Community, Other, Social Visit].\n\n"
    "Behavioral Logic:\n  - Holistic Decision Influence: Every plan and "
    "decision must be fundamentally shaped by your demographic profile, "
    "temporal context (time and day of week), and the specific urban "
    "environment of your city.\n  - Identity-Driven Action: Act as a person of"
    " your specific age and role would. \n    - Workers and students should "
    "prioritize consistent blocks for their primary responsibilities during "
    "their typical active schedule. \n    - Homemakers should maintain "
    "realistic, home-centered daily rhythms and adapt plans to household or "
    "family needs as they arise.\n  - Continuity & Realism: Aim for meaningful"
    ", continuous blocks of time (e.g., 30 minutes to 8 hours depending on the"
    " activity). Specifically, recognize that nighttime rest at 'home' is a "
    "long, singular event that should span several hours.\n  - Frequency & "
    "Periodic Activities: Differentiate between daily staples (e.g., work, "
    "school, sleep) and periodic tasks (e.g., grocery shopping, social visits,"
    " or healthcare). Periodic tasks typically occur every few days or weeks; "
    "do not force them into every daily plan unless specifically needed.\n  - "
    "Cognitive Continuity: You possess a persistent memory stream that "
    "captures observations, decisions, and the specific intent behind your "
    "actions. Your memories are a narrative of your life; use past intentions "
    "to maintain social and personal consistency.\n\nAgent Profile (Specific "
    "to this instance):\n  - Identity: 33-year-old female homemaker.\n  - "
    "Location: Resident of San Francisco.\n  - Nuances: employment status: "
    "non_worker; work schedule: not_applicable; school type: not_in_school; "
    "household vehicles: 2; household income: 50k to 74k; household lifecycle:"
    " two or more adults no children"
)
DEFAULT_USER_MESSAGE = (
    "A new day has started (Monday). Provide a rough mental sketch of your "
    "plans for the day ahead.\n\nCurrent State:\n  - Starting Location: Home"
    "\nPast Routines & Insights:\n  - No notable memories yet.\n\nPlanning "
    "Rules:\n  - Use the insights above to inform your routine, but do not "
    "simply repeat them verbatim. Adapt your intentions to the specific day of"
    " the week and your identity.\n  - Frequency Control: Distinguish between "
    "daily \"anchor\" activities and periodic errands. If a past memory shows "
    "you went shopping or visited a friend, treat that as a periodic event"
    "\u2014do not plan to do it again today unless it realistically fits your "
    "current needs.\n  - Focus on your major intentions and rough timing "
    "(e.g., when you'll head to work or school and anything special you have "
    "planned for the day).\n  - Maintain a high-level perspective; avoid rigid"
    " minute-by-minute constraints.\n  - Format: 2-3 concise sentences.\n  - "
    "Output: A short narrative paragraph.\n\nActivity Categories:\n  - "
    "Available categories: Home, Work, Eat Meal, Education, Recreational, "
    "Shopping, Care, Community, Other, Social Visit\nCategory Notes:\n    - "
    "Home: Home activities, including staying at home for personal routines.\n"
    "    - Work: Work and work-related destinations.\n    - Eat Meal: Going "
    "out to eat (restaurant/cafe/food pickup destination).\n    - Education: "
    "Education-related destinations, including school and daycare/child-care "
    "attendance.\n    - Recreational: Recreation, leisure, and exercise "
    "destinations.\n    - Shopping: Buying goods and services.\n    - Care: "
    "Health care and adult-care destinations.\n    - Community: Volunteer, "
    "religious, and community activities.\n    - Other: Other/general purposes"
    " not covered by the categories above.\n    - Social Visit: Visiting "
    "friends or relatives (often at another residence).\n\n\n"
)
DEFAULT_BASE_MODEL_ID = "unsloth/gpt-oss-20b-BF16"
ADAPTER_ID = "gpt-oss-20b-lora-1epochs"


resp = client.chat.completions.create(
    model=DEFAULT_BASE_MODEL_ID,
    messages=[
        {"role": "system", "content": DEFAULT_SYSTEM_INSTRUCTION},
        {"role": "user", "content": DEFAULT_USER_MESSAGE},
    ],
    temperature=1,
    max_tokens=4096,
)

print(resp)
print(resp.choices[0].message.content)


resp = client.chat.completions.create(
    model=ADAPTER_ID,
    messages=[
        {"role": "system", "content": DEFAULT_SYSTEM_INSTRUCTION},
        {"role": "user", "content": DEFAULT_USER_MESSAGE},
    ],
    temperature=1,
    max_tokens=4096,
)

print(resp)
print(resp.choices[0].message.content)
