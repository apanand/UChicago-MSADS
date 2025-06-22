import uuid

def run(input_data: dict) -> dict:
    activities = input_data.get("activities", [])
    if not activities:
        return {"error": "No activities provided for booking."}
    bookings = []
    for activity in activities:
        bookings.append({
            "activity": activity,
            "status": "booked",
            "booking_id": str(uuid.uuid4())
        })
    return {"booking_summary": bookings}