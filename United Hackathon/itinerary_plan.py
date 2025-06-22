#Grok + Planning
import streamlit as st
import requests
import json
from datetime import datetime, timedelta
from dateutil.parser import parse

# ---- KEYS ----
GOOGLE_API_KEY = "AIzaSyBDaj8tEzHBD6m3DhlTI04BcslTBmHflO0"   # Replace with your real key
groq_api_key = "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"     # Replace with your Groq API key


# ---- Groq API Setup ----
def generate_activities(city):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "system",
                "content": "You are a travel planner. Suggest fun activities for travelers."
            },
            {
                "role": "user",
                "content": f"Suggest 10 fun and diverse activities to do in {city}. Just return a numbered list."
            }
        ],
        "temperature": 0.8
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        suggestions = response.json()["choices"][0]["message"]["content"]
        return [line.split('. ', 1)[1] for line in suggestions.split('\n') if '. ' in line]
    else:
        st.error("Failed to fetch activity suggestions from Groq.")
        return []

# ---- Google Maps Functions ----
def get_coordinates(destination):
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={destination}&key={GOOGLE_API_KEY}"
    location_data = requests.get(geocode_url).json()
    if not location_data.get("results"):
        return None
    return location_data["results"][0]["geometry"]["location"]

def search_places(lat, lng, place_type, keyword=None, num_results=10):
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": 5000,
        "type": place_type,
        "key": GOOGLE_API_KEY
    }
    if keyword:
        params["keyword"] = keyword

    response = requests.get(base_url, params=params).json()
    return response.get("results", [])[:num_results]

def format_place(place):
    return f"**{place.get('name')}**\n{place.get('vicinity', '')}\n⭐ {place.get('rating', 'N/A')}"

def get_groq_description(name, city):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a travel planner. Write a short, friendly, customer-facing description for a travel activity."},
            {"role": "user", "content": f"Write a 1-2 sentence description for the following activity in {city}: {name}"}
        ],
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return f"Enjoy {name} in {city}!"

def requires_booking(place):
    # Heuristic: Museums, restaurants, tours, theaters, stadiums, galleries, cruises, zoos, aquariums, amusement parks, etc. require booking
    # Parks, monuments, public squares, beaches, etc. do not
    booking_keywords = [
        "museum", "restaurant", "tour", "theater", "stadium", "gallery", "cruise", "zoo", "aquarium", "amusement park", "exhibit", "concert", "show", "attraction"
    ]
    name = place.get("name", "").lower()
    types = place.get("types", [])
    for kw in booking_keywords:
        if kw in name or kw.replace(" ", "_") in types or kw in types:
            return True
    return False

# ---- Streamlit UI ----
st.title("✈️ United Travel Planner")

with st.form("travel_form"):
    destination = st.text_input("Where are you going?", "San Francisco")
    start_date = st.date_input("Start Date", datetime.today())
    duration = st.number_input("Trip Duration (days)", min_value=1, max_value=14, value=3)

    st.markdown("#### Preferences")
    activity_pref = st.text_input("Activity preference (e.g., museum, hiking, landmark)", "")
    food_pref = st.text_input("Food preference (e.g., Italian, vegan, sushi)", "")
    
    submitted = st.form_submit_button("Generate Itinerary")

if submitted:
    coords = get_coordinates(destination)
    if not coords:
        st.error("Could not locate that destination. Please try another.")
    else:
        st.success("Itinerary generated below!")

        # Google Places API search
        attractions = search_places(coords['lat'], coords['lng'], "tourist_attraction", keyword=activity_pref)
        restaurants = search_places(coords['lat'], coords['lng'], "restaurant", keyword=food_pref)

        st.markdown("## 🧳 Your Itinerary")
        itinerary_json = []

        for i in range(duration):
            date = parse(str(start_date)) + timedelta(days=i)
            st.markdown(f"### Day {i+1}: {date.strftime('%A, %B %d')}")

            activity = attractions[i % len(attractions)] if attractions else None
            restaurant = restaurants[i % len(restaurants)] if restaurants else None

            if activity:
                st.markdown(f"🏛️ **Activity:** {format_place(activity)}")
            else:
                st.markdown("🏛️ Activity: Explore on your own!")

            if restaurant:
                st.markdown(f"🍽️ **Restaurant:** {format_place(restaurant)}")
            else:
                st.markdown("🍽️ Restaurant: Local dining options available.")

            # Add to JSON structure
            itinerary_json.append({
                "day": i + 1,
                "date": date.strftime("%Y-%m-%d"),
                "activity": {
                    "name": activity.get("name") if activity else "Explore on your own",
                    "address": activity.get("vicinity") if activity else "",
                    "rating": activity.get("rating") if activity else ""
                },
                "restaurant": {
                    "name": restaurant.get("name") if restaurant else "Local dining options",
                    "address": restaurant.get("vicinity") if restaurant else "",
                    "rating": restaurant.get("rating") if restaurant else ""
                }
            })

        # Output itinerary as JSON
        st.markdown("---")
        st.markdown("## 📦 Itinerary Export (for Booking Agent)")
        st.json(itinerary_json)

        # Dynamic Groq Suggestions
        st.markdown("## 💡 More Suggested Activities (via Groq AI)")
        groq_suggestions = generate_activities(destination)
        selected_activities = []
        for i, suggestion in enumerate(groq_suggestions):
            if st.checkbox(suggestion, key=f"activity_{i}"):
                selected_activities.append(suggestion)

        if selected_activities:
            st.markdown("### ✅ Selected Additional Activities")
            for act in selected_activities:
                st.markdown(f"- {act}")

def run(input_data: dict) -> dict:
    """
    Returns a day-by-day itinerary for the trip duration, with each day containing an activity and a restaurant (if available).
    Each activity includes a description and a requires_booking boolean.
    """
    city = input_data.get("destination", "Chicago")
    activity_pref = input_data.get("activity_pref", None)
    food_pref = input_data.get("food_pref", None)
    start_date = input_data.get("start_date", str(datetime.today().date()))
    end_date = input_data.get("end_date", str(datetime.today().date()))
    try:
        start = parse(start_date)
        end = parse(end_date)
    except Exception:
        start = datetime.today()
        end = datetime.today()
    duration = (end - start).days + 1
    coords = get_coordinates(city)
    if not coords:
        return {"error": "Could not locate that destination."}
    attractions = search_places(coords['lat'], coords['lng'], "tourist_attraction", keyword=activity_pref, num_results=duration)
    restaurants = search_places(coords['lat'], coords['lng'], "restaurant", keyword=food_pref, num_results=duration)
    itinerary = []
    for i in range(duration):
        day = start + timedelta(days=i)
        activity = attractions[i % len(attractions)] if attractions else None
        restaurant = restaurants[i % len(restaurants)] if restaurants else None
        # Activity details
        if activity:
            act = {
                "name": activity.get("name", "Unknown Place"),
                "address": activity.get("vicinity", ""),
                "rating": activity.get("rating", "N/A"),
                "lat": activity["geometry"]["location"]["lat"],
                "lng": activity["geometry"]["location"]["lng"],
                "photo_reference": activity["photos"][0]["photo_reference"] if activity.get("photos") else None,
                "description": get_groq_description(activity.get("name", ""), city),
                "requires_booking": requires_booking(activity),
            }
        else:
            act = None
        # Restaurant details
        if restaurant:
            rest = {
                "name": restaurant.get("name", "Unknown Restaurant"),
                "address": restaurant.get("vicinity", ""),
                "rating": restaurant.get("rating", "N/A"),
                "lat": restaurant["geometry"]["location"]["lat"],
                "lng": restaurant["geometry"]["location"]["lng"],
                "photo_reference": restaurant["photos"][0]["photo_reference"] if restaurant.get("photos") else None,
                "description": get_groq_description(restaurant.get("name", ""), city),
                "requires_booking": True,  # Assume restaurants require booking
            }
        else:
            rest = None
        itinerary.append({
            "day": i + 1,
            "date": day.strftime("%A, %B %d, %Y"),
            "activity": act,
            "restaurant": rest
        })
    return {"itinerary": itinerary, "city": city}
