# orchestrator.py

import streamlit as st
st.set_page_config(page_title="🌍 Smart Travel Planner", layout="wide")
from datetime import date
from pre_trip_reqs import run as run_pre_trip
from flight_search import run as run_flight
from itinerary_plan import run as run_itinerary
from booking_interactive import run as run_booking

GOOGLE_API_KEY = "AIzaSyBDaj8tEzHBD6m3DhlTI04BcslTBmHflO0"  # Use your real key

def get_photo_url(photo_reference, api_key):
    return f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={api_key}"

def get_maps_link(lat, lng):
    return f"https://www.google.com/maps/dir/?api=1&destination={lat},{lng}"

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1
if "trip_data" not in st.session_state:
    st.session_state.trip_data = {}
if "itinerary" not in st.session_state:
    st.session_state.itinerary = []
if "selected_bookings" not in st.session_state:
    st.session_state.selected_bookings = []
if "booking_summary" not in st.session_state:
    st.session_state.booking_summary = []
if "payment_method" not in st.session_state:
    st.session_state.payment_method = None

# Set up page
st.title("🌍 Smart Travel Planner")
st.markdown("Use AI agents to plan your next international trip.")

# Step 1: Trip details form
if st.session_state.step == 1:
    with st.form("trip_form"):
        destination = st.text_input("Destination", "Chicago")
        nationality = st.text_input("Nationality", "US")
        trip_purpose = st.selectbox("Trip Purpose", ["Tourism", "Business", "Work", "Study"], index=0)
        start_date = st.date_input("Start Date", date.today())
        end_date = st.date_input("End Date", date.today())
        activity_pref = st.text_input("Activity preference (optional)", "")
        food_pref = st.text_input("Food preference (optional)", "")
        submitted = st.form_submit_button("Next: Generate Itinerary")
    if submitted:
        st.session_state.trip_data = {
            "destination": destination,
            "nationality": nationality,
            "trip_purpose": trip_purpose,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "activity_pref": activity_pref,
            "food_pref": food_pref,
        }
        st.session_state.step = 2
        st.rerun()

# Step 2: Show day-by-day itinerary with booking options
elif st.session_state.step == 2:
    st.header("Step 2: Your Day-by-Day Itinerary")
    itinerary_result = run_itinerary(st.session_state.trip_data)
    itinerary = itinerary_result.get("itinerary", [])
    st.session_state.itinerary = itinerary
    selected_bookings = []
    if not itinerary:
        st.error("No itinerary found.")
    else:
        for day in itinerary:
            st.subheader(f"Day {day['day']}: {day['date']}")
            cols = st.columns(2)
            # Activity
            activity = day.get("activity")
            with cols[0]:
                if activity:
                    st.markdown(f"### Activity: {activity['name']}")
                    if activity["photo_reference"]:
                        st.image(get_photo_url(activity["photo_reference"], GOOGLE_API_KEY), use_column_width=True)
                    st.write(activity["address"])
                    st.write(f"⭐ {activity['rating']}")
                    st.write(activity["description"])
                    st.markdown(f"[Get Directions]({get_maps_link(activity['lat'], activity['lng'])})")
                    if activity.get("requires_booking"):
                        checked = st.checkbox(f"Book this activity: {activity['name']}", key=f"book_activity_{day['day']}")
                        if checked:
                            selected_bookings.append({"type": "activity", "name": activity["name"], "details": activity})
            # Restaurant
            restaurant = day.get("restaurant")
            with cols[1]:
                if restaurant:
                    st.markdown(f"### Restaurant: {restaurant['name']}")
                    if restaurant["photo_reference"]:
                        st.image(get_photo_url(restaurant["photo_reference"], GOOGLE_API_KEY), use_column_width=True)
                    st.write(restaurant["address"])
                    st.write(f"⭐ {restaurant['rating']}")
                    st.write(restaurant["description"])
                    st.markdown(f"[Get Directions]({get_maps_link(restaurant['lat'], restaurant['lng'])})")
                    if restaurant.get("requires_booking"):
                        checked = st.checkbox(f"Book this restaurant: {restaurant['name']}", key=f"book_restaurant_{day['day']}")
                        if checked:
                            selected_bookings.append({"type": "restaurant", "name": restaurant["name"], "details": restaurant})
        if st.button("Next: Book Selected Items"):
            st.session_state.selected_bookings = selected_bookings
            st.session_state.step = 3
            st.rerun()
    if st.button("Back"):
        st.session_state.step = 1
        st.rerun()

# Step 3: Booking and payment
elif st.session_state.step == 3:
    st.header("Step 3: Booking & Payment")
    selected_bookings = st.session_state.selected_bookings
    if not selected_bookings:
        st.info("No activities or restaurants selected for booking.")
        if st.button("Back to Itinerary Selection"):
            st.session_state.step = 2
            st.rerun()
    else:
        st.subheader("You are booking the following:")
        for item in selected_bookings:
            st.markdown(f"- **{item['type'].capitalize()}: {item['name']}** ({item['details']['address']})")
        st.markdown("---")
        payment_method = st.selectbox("Choose payment method", ["Google Pay", "Apple Pay", "Credit Card"])
        st.session_state.payment_method = payment_method
        payment_ok = False
        if payment_method == "Credit Card":
            card_number = st.text_input("Card Number")
            expiry = st.text_input("Expiry Date (MM/YY)")
            cvv = st.text_input("CVV")
            if card_number and expiry and cvv:
                payment_ok = True
        else:
            st.info(f"You will be redirected to {payment_method} for payment (mockup)")
            payment_ok = True
        if st.button("Confirm Booking") and payment_ok:
            # Only pass names to booking agent for now
            booking_input = {"activities": [item["name"] for item in selected_bookings]}
            booking_result = run_booking(booking_input)
            st.session_state.booking_summary = booking_result.get("booking_summary", [])
            st.session_state.step = 4
            st.rerun()
        if st.button("Back to Itinerary Selection"):
            st.session_state.step = 2
            st.rerun()

# Step 4: Confirmation
elif st.session_state.step == 4:
    st.header("Step 4: Booking Confirmation")
    if st.session_state.booking_summary:
        st.success("Your bookings are confirmed!")
        for booking in st.session_state.booking_summary:
            st.markdown(f"**Item:** {booking['activity']}")
            st.markdown(f"**Status:** {booking['status']}")
            st.markdown(f"**Booking ID:** `{booking['booking_id']}`")
            st.markdown("---")
    else:
        st.info("No bookings made.")
    if st.button("Back to Start"):
        st.session_state.step = 1
        st.session_state.selected_bookings = []
        st.session_state.booking_summary = []
        st.rerun()
