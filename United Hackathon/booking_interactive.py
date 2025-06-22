# booking_interactive.py — final version for Fast MCP

import uuid

def run(input_data: dict) -> dict:
    """
    Fast MCP-compatible run function for booking agent.
    Uses 'activities' from the previous agent, not a file.
    """
    activities = input_data.get("activities", [])
    
    if not activities:
        return {
            "error": "No activities provided for booking."
        }

    bookings = []
    for i, activity in enumerate(activities, 1):
        bookings.append({
            "activity": activity,
            "status": "booked",
            "booking_id": str(uuid.uuid4())
        })

    return {"booking_summary": bookings}
