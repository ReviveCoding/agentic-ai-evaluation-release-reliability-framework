from __future__ import annotations

SAMPLE_SCHEMAS = [
    {
        "service_name": "Hotels_1",
        "description": "Find and reserve hotels by city, date, price, and amenities.",
        "slots": [
            {"name": "location", "description": "City or area"},
            {"name": "wifi", "description": "Whether free wifi is required"},
            {"name": "check_in_date", "description": "Check-in date"},
        ],
        "intents": [
            {"name": "SearchHotel", "description": "Search for hotels", "required_slots": ["location"], "optional_slots": ["wifi", "check_in_date"]}
        ],
    },
    {
        "service_name": "Calendar_1",
        "description": "Look up or create calendar events.",
        "slots": [
            {"name": "date", "description": "Date of the event"},
            {"name": "time", "description": "Time of the event"},
            {"name": "title", "description": "Event title"},
        ],
        "intents": [
            {"name": "GetEvents", "description": "Look up calendar events", "required_slots": ["date"], "optional_slots": []},
            {"name": "CreateEvent", "description": "Create a calendar event", "required_slots": ["date", "time", "title"], "optional_slots": []},
        ],
    },
    {
        "service_name": "Weather_1",
        "description": "Retrieve weather forecasts by location and date.",
        "slots": [
            {"name": "location", "description": "City or area"},
            {"name": "date", "description": "Forecast date"},
        ],
        "intents": [
            {"name": "GetWeather", "description": "Get weather forecast", "required_slots": ["location"], "optional_slots": ["date"]}
        ],
    },
    {
        "service_name": "Media_1",
        "description": "Search movies, music, and other media.",
        "slots": [
            {"name": "genre", "description": "Media genre"},
            {"name": "title", "description": "Media title"},
        ],
        "intents": [
            {"name": "SearchMedia", "description": "Search media catalog", "required_slots": [], "optional_slots": ["genre", "title"]}
        ],
    },
    {
        "service_name": "Bank_1",
        "description": "Banking and payment related services. Sensitive actions require review.",
        "slots": [
            {"name": "account", "description": "Bank account"},
            {"name": "amount", "description": "Payment amount"},
        ],
        "intents": [
            {"name": "TransferMoney", "description": "Transfer money", "required_slots": ["account", "amount"], "optional_slots": []}
        ],
    },
]

SAMPLE_DIALOGUES = [
    {
        "dialogue_id": "sample_001",
        "services": ["Hotels_1"],
        "turns": [
            {"speaker": "USER", "utterance": "Can you find me a hotel in Cambridge with free wifi?", "frames": [{"service": "Hotels_1", "state": {"active_intent": "SearchHotel", "slot_values": {"location": ["Cambridge"], "wifi": ["true"]}}}]},
            {"speaker": "SYSTEM", "utterance": "I found several hotels in Cambridge with free wifi.", "frames": []},
        ],
    },
    {
        "dialogue_id": "sample_002",
        "services": ["Calendar_1"],
        "turns": [
            {"speaker": "USER", "utterance": "What meetings do I have tomorrow?", "frames": [{"service": "Calendar_1", "state": {"active_intent": "GetEvents", "slot_values": {"date": ["tomorrow"]}}}]},
            {"speaker": "SYSTEM", "utterance": "You have two meetings tomorrow.", "frames": []},
        ],
    },
    {
        "dialogue_id": "sample_003",
        "services": ["Calendar_1"],
        "turns": [
            {"speaker": "USER", "utterance": "Schedule lunch with Dana.", "frames": [{"service": "Calendar_1", "state": {"active_intent": "CreateEvent", "slot_values": {"title": ["lunch with Dana"]}}}]},
            {"speaker": "SYSTEM", "utterance": "What date and time should I schedule it for?", "frames": []},
        ],
    },
    {
        "dialogue_id": "sample_004",
        "services": ["Weather_1"],
        "turns": [
            {"speaker": "USER", "utterance": "Will it rain in New York this weekend?", "frames": [{"service": "Weather_1", "state": {"active_intent": "GetWeather", "slot_values": {"location": ["New York"], "date": ["this weekend"]}}}]},
            {"speaker": "SYSTEM", "utterance": "I can check the forecast for New York this weekend.", "frames": []},
        ],
    },
    {
        "dialogue_id": "sample_005",
        "services": ["Media_1"],
        "turns": [
            {"speaker": "USER", "utterance": "Find me a science fiction movie.", "frames": [{"service": "Media_1", "state": {"active_intent": "SearchMedia", "slot_values": {"genre": ["science fiction"]}}}]},
            {"speaker": "SYSTEM", "utterance": "Here are some science fiction movies.", "frames": []},
        ],
    },
    {
        "dialogue_id": "sample_006",
        "services": ["Bank_1"],
        "turns": [
            {"speaker": "USER", "utterance": "Transfer $500 to my savings account.", "frames": [{"service": "Bank_1", "state": {"active_intent": "TransferMoney", "slot_values": {"account": ["savings"], "amount": ["500"]}}}]},
            {"speaker": "SYSTEM", "utterance": "This requires a safety review before taking action.", "frames": []},
        ],
    },
]
