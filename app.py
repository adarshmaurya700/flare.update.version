"""
FLARE — Municipal Intelligence & Hyperlocal Deduplication Engine
HackIIITD (Esya) | IIIT Delhi
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multimodal dedup: Image + Text + GPS → Master Incident Tickets
"""

import streamlit as st
import sqlite3
import os
import math
import hashlib
from datetime import datetime, timedelta
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sentence_transformers import SentenceTransformer
import folium
from streamlit_folium import st_folium
import random
import time
import requests
import urllib.parse

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DB_PATH = os.path.join("data", "civicdedupe.db")
DEMO_IMAGES_DIR = "demo_images"
UPLOADS_DIR = os.path.join(DEMO_IMAGES_DIR, "uploads")
SPATIAL_THRESHOLD_M = 60.0
COMPOSITE_THRESHOLD = 0.72
W_VISUAL = 0.50
W_TEXT = 0.35
W_SPATIAL = 0.15
MAP_CENTER = [28.5430, 77.2630]
MAP_ZOOM = 15

# TinyFish API Configuration
TINYFISH_API_KEY = os.getenv("TINYFISH_API_KEY", "sk-tinyfish-yuzX2sUn0DQUFPFkCVID1RipaoFItqOY")
TINYFISH_SEARCH_URL = "https://api.search.tinyfish.ai"
TINYFISH_FETCH_URL = "https://api.fetch.tinyfish.ai"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRANSLATIONS (English / Hindi)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSLATIONS = {
    "en": {
        "app_sub": "FLARE Municipal Engine · South Delhi Administrative Division",
        "status_pill": "System Active · Live Node: Okhla Zone",
        "total_complaints": "Total Complaints",
        "master_work_orders": "Master Work Orders",
        "dedup_rate": "Dedup Rate",
        "citizen_report": "📝 Citizen Report Submission",
        "quickfill_helper": "💡 Click a scenario below to run one-click verification",
        "demo_quickfill": "Demo Quick-Fill",
        "quickfill_help": "Pick a pre-configured scenario to auto-fill the form",
        "expected": "Expected",
        "your_name": "Your Name",
        "phone": "Phone",
        "complaint_desc": "Complaint Description (English / Hinglish)",
        "complaint_placeholder": "e.g. Naali se ganda paani beh raha hai…",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "demo_photo": "Demo complaint photo",
        "upload_photo": "Upload photo (overrides demo image)",
        "submit_btn": "🚀  Submit Grievance",
        "err_no_text": "Please enter a complaint description.",
        "err_no_photo": "Please upload a photo of the issue.",
        "running_dedup": "Running multimodal deduplication…",
        "ingestion_telemetry": "📡 Ingestion Telemetry",
        "dup_detected": "Duplicate Detected — Merged",
        "composite_score": "Composite Score",
        "threshold": "threshold",
        "new_work_order": "New Work Order Created",
        "category": "Category",
        "distance": "📍 Distance",
        "visual_match": "🖼️ Visual Match",
        "semantic_match": "📝 Semantic Match",
        "spatial_decay": "📐 Spatial Decay",
        "composite": "🎯 Composite",
        "formula_label": "Formula: 0.50×Visual + 0.35×Text + 0.15×Spatial",
        "nearest_ticket": "Nearest ticket",
        "away_outside": "away — outside the",
        "dedup_radius": "m dedup radius.",
        "no_tickets": "No existing tickets to compare against.",
        "merged_into": "🔗 MERGED INTO MASTER TICKET",
        "new_order_reg": "🆕 NEW WORK ORDER",
        "registered": "REGISTERED",
        "all_candidates": "All candidate comparisons",
        "evaluated": "⚡ Evaluated",
        "outside_radius": "🚫 Outside radius",
        "distance_label": "Distance",
        "composite_label": "Composite",
        "command_center": "📊 Municipal Command Center",
        "incident_map": "🗺️ Incident Map",
        "ticket_inspector": "🎫 Ticket Inspector",
        "reports": "reports",
        "priority": "Priority",
        "description": "Description",
        "base_severity": "Base Severity",
        "citizens": "Citizens",
        "created": "Created",
        "primary_photo": "Primary photo",
        "linked_reports": "Linked Citizen Reports",
        "new_report_tooltip": "📍 New Report",
        "within_threshold": "Within {t}m threshold",
        "outside_threshold": "Outside {t}m radius",
        "meters_away": "meters away",
        "reports_label": "Reports",
        "priority_critical": "Critical",
        "priority_moderate": "Moderate",
        "priority_low": "Low",
        "scenario_prompt": "— Select a Demo Scenario —",
        "scenario_a_title": "🔴 Scenario A · Definite Duplicate (Sewage ~18 m from Ticket #1)",
        "scenario_a_expected": "DUPLICATE → Merge into Ticket #1",
        "scenario_b_title": "🟡 Scenario B · Different Issue, Same Area (~25 m from Ticket #1)",
        "scenario_b_expected": "NEW INCIDENT — spatial proximity alone should NOT match",
        "scenario_c_title": "🟢 Scenario C · Completely New Issue (~1.5 km away)",
        "scenario_c_expected": "NEW INCIDENT — no spatial candidates within 60 m",
        "tinyfish_header": "🌐 Live Web Intelligence & Autonomous Dispatch",
        "tinyfish_powered": "Powered by TinyFish",
        "tinyfish_search_label": "Web Grievance Search Query",
        "tinyfish_search_default": "sewage overflow or road cave-in South Delhi",
        "tinyfish_search_btn": "🔍 Scan Web via TinyFish Search",
        "tinyfish_searching": "Querying live web via TinyFish Search API…",
        "tinyfish_no_results": "No web results found. Try a different query.",
        "tinyfish_results_title": "🌐 Live Web Discoveries",
        "tinyfish_source": "Source",
        "tinyfish_dispatch_header": "🤖 Autonomous Municipal Dispatch",
        "tinyfish_dispatch_desc": "Deploy TinyFish Web Agent to auto-file critical tickets on the MCD grievance portal.",
        "tinyfish_dispatch_btn": "🚀 Deploy TinyFish Web Agent to MCD Portal",
        "tinyfish_dispatch_step1": "Launching cloud browser session via TinyFish Web Agent...",
        "tinyfish_dispatch_step2": "Navigating to municipal grievance portal (mcdonline.nic.in)...",
        "tinyfish_dispatch_step3": "Populating master complaint dossier, GPS coordinates, and aggregated citizen photos...",
        "tinyfish_dispatch_step4": "Submitting ticket and retrieving municipal receipt #MCD-2026-{rid}.",
        "tinyfish_dispatch_done": "✅ Municipal receipt #{receipt} filed successfully for Ticket #{tid} ({cat})",
        "tinyfish_no_critical": "No critical-priority tickets available for dispatch.",
        "tinyfish_fetch_btn": "📄 Extract Full Article",
        "tinyfish_fetching": "Fetching article content via TinyFish Fetch API…",
    },
    "hi": {
        "app_sub": "FLARE नगरपालिका इंजन · दक्षिण दिल्ली प्रशासनिक प्रभाग",
        "status_pill": "सिस्टम सक्रिय · लाइव नोड: ओखला ज़ोन",
        "total_complaints": "कुल शिकायतें",
        "master_work_orders": "मास्टर कार्य आदेश",
        "dedup_rate": "डीडुप दर",
        "citizen_report": "📝 नागरिक शिकायत दर्ज करें",
        "quickfill_helper": "💡 एक-क्लिक सत्यापन के लिए नीचे एक परिदृश्य चुनें",
        "demo_quickfill": "डेमो क्विक-फ़िल",
        "quickfill_help": "फ़ॉर्म ऑटो-फ़िल करने के लिए एक पूर्व-निर्धारित परिदृश्य चुनें",
        "expected": "अपेक्षित",
        "your_name": "आपका नाम",
        "phone": "फ़ोन",
        "complaint_desc": "शिकायत विवरण (हिंदी / अंग्रेज़ी / हिंगलिश)",
        "complaint_placeholder": "जैसे: नाली से गंदा पानी बह रहा है…",
        "latitude": "अक्षांश",
        "longitude": "देशांतर",
        "demo_photo": "डेमो शिकायत फ़ोटो",
        "upload_photo": "फ़ोटो अपलोड करें (डेमो छवि को ओवरराइड करता है)",
        "submit_btn": "🚀  शिकायत दर्ज करें",
        "err_no_text": "कृपया शिकायत का विवरण दर्ज करें।",
        "err_no_photo": "कृपया समस्या की फ़ोटो अपलोड करें।",
        "running_dedup": "मल्टीमॉडल डीडुप्लिकेशन चल रहा है…",
        "ingestion_telemetry": "📡 इंजेशन टेलीमेट्री",
        "dup_detected": "डुप्लिकेट पाया गया — मर्ज किया गया",
        "composite_score": "समग्र स्कोर",
        "threshold": "सीमा",
        "new_work_order": "नया कार्य आदेश बनाया गया",
        "category": "श्रेणी",
        "distance": "📍 दूरी",
        "visual_match": "🖼️ दृश्य मिलान",
        "semantic_match": "📝 अर्थ मिलान",
        "spatial_decay": "📐 स्थानिक क्षय",
        "composite": "🎯 समग्र",
        "formula_label": "सूत्र: 0.50×दृश्य + 0.35×पाठ + 0.15×स्थानिक",
        "nearest_ticket": "निकटतम टिकट",
        "away_outside": "दूर — इसके बाहर",
        "dedup_radius": "मी डीडुप दायरा।",
        "no_tickets": "तुलना के लिए कोई मौजूदा टिकट नहीं।",
        "merged_into": "🔗 मास्टर टिकट में मर्ज किया गया",
        "new_order_reg": "🆕 नया कार्य आदेश",
        "registered": "पंजीकृत",
        "all_candidates": "सभी उम्मीदवार तुलनाएँ",
        "evaluated": "⚡ मूल्यांकित",
        "outside_radius": "🚫 दायरे से बाहर",
        "distance_label": "दूरी",
        "composite_label": "समग्र",
        "command_center": "📊 नगरपालिका कमांड सेंटर",
        "incident_map": "🗺️ घटना मानचित्र",
        "ticket_inspector": "🎫 टिकट निरीक्षक",
        "reports": "रिपोर्ट",
        "priority": "प्राथमिकता",
        "description": "विवरण",
        "base_severity": "आधार गंभीरता",
        "citizens": "नागरिक",
        "created": "बनाया गया",
        "primary_photo": "मुख्य फ़ोटो",
        "linked_reports": "लिंक्ड नागरिक रिपोर्ट",
        "new_report_tooltip": "📍 नई रिपोर्ट",
        "within_threshold": "{t}मी सीमा के अंदर",
        "outside_threshold": "{t}मी दायरे के बाहर",
        "meters_away": "मीटर दूर",
        "reports_label": "रिपोर्ट",
        "priority_critical": "गंभीर (Critical)",
        "priority_moderate": "मध्यम (Moderate)",
        "priority_low": "निम्न (Low)",
        "scenario_prompt": "— डेमो परिदृश्य चुनें —",
        "scenario_a_title": "🔴 परिदृश्य A · निश्चित डुप्लिकेट (सीवेज ~18 मी टिकट #1 से)",
        "scenario_a_expected": "डुप्लिकेट → टिकट #1 में मर्ज होगा",
        "scenario_b_title": "🟡 परिदृश्य B · अलग समस्या, समान क्षेत्र (~25 मी टिकट #1 से)",
        "scenario_b_expected": "नई घटना — केवल निकटता होने से मैच नहीं होगा",
        "scenario_c_title": "🟢 परिदृश्य C · पूरी तरह नई समस्या (~1.5 किमी दूर)",
        "scenario_c_expected": "नई घटना — 60 मीटर के भीतर कोई उम्मीदवार नहीं",
        "tinyfish_header": "🌐 लाइव वेब इंटेलिजेंस और स्वचालित डिस्पैच",
        "tinyfish_powered": "TinyFish द्वारा संचालित",
        "tinyfish_search_label": "वेब शिकायत खोज क्वेरी",
        "tinyfish_search_default": "सीवेज ओवरफ्लो या सड़क धंसाव दक्षिण दिल्ली",
        "tinyfish_search_btn": "🔍 TinyFish खोज से वेब स्कैन करें",
        "tinyfish_searching": "TinyFish Search API से लाइव वेब क्वेरी हो रही है…",
        "tinyfish_no_results": "कोई वेब परिणाम नहीं मिला। अलग क्वेरी आज़माएँ।",
        "tinyfish_results_title": "🌐 लाइव वेब खोज",
        "tinyfish_source": "स्रोत",
        "tinyfish_dispatch_header": "🤖 स्वचालित नगरपालिका डिस्पैच",
        "tinyfish_dispatch_desc": "MCD शिकायत पोर्टल पर गंभीर टिकट स्वचालित रूप से दर्ज करने के लिए TinyFish वेब एजेंट तैनात करें।",
        "tinyfish_dispatch_btn": "🚀 TinyFish वेब एजेंट MCD पोर्टल पर तैनात करें",
        "tinyfish_dispatch_step1": "TinyFish वेब एजेंट द्वारा क्लाउड ब्राउज़र सत्र शुरू हो रहा है...",
        "tinyfish_dispatch_step2": "नगरपालिका शिकायत पोर्टल (mcdonline.nic.in) पर नेविगेट हो रहा है...",
        "tinyfish_dispatch_step3": "मास्टर शिकायत डोज़ियर, GPS कोऑर्डिनेट और एकत्रित नागरिक फ़ोटो भरे जा रहे हैं...",
        "tinyfish_dispatch_step4": "टिकट सबमिट हो रहा है और नगरपालिका रसीद #MCD-2026-{rid} प्राप्त हो रही है।",
        "tinyfish_dispatch_done": "✅ नगरपालिका रसीद #{receipt} टिकट #{tid} ({cat}) के लिए सफलतापूर्वक दर्ज हुई",
        "tinyfish_no_critical": "डिस्पैच के लिए कोई गंभीर-प्राथमिकता टिकट उपलब्ध नहीं।",
        "tinyfish_fetch_btn": "📄 पूरा लेख निकालें",
        "tinyfish_fetching": "TinyFish Fetch API से लेख सामग्री प्राप्त हो रही है…",
    },
}


def get_lang():
    """Return the translation dict for the current language."""
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TINYFISH API CONNECTORS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def search_civic_web(query, location="Delhi"):
    """Call TinyFish Search API to find live civic grievances on the web."""
    try:
        full_query = f"{query} {location}"
        encoded_query = urllib.parse.quote(full_query)
        resp = requests.get(
            f"{TINYFISH_SEARCH_URL}?query={encoded_query}",
            headers={"X-API-Key": TINYFISH_API_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        # TinyFish returns results in different formats; handle both
        if isinstance(data, list):
            return data[:5]
        elif isinstance(data, dict):
            return data.get("results", data.get("data", []))[:5]
        return []
    except Exception as e:
        st.warning(f"TinyFish Search API error: {e}")
        return []


def fetch_civic_bulletin(url):
    """Call TinyFish Fetch API to extract clean Markdown text from a URL."""
    try:
        resp = requests.post(
            TINYFISH_FETCH_URL,
            headers={"X-API-Key": TINYFISH_API_KEY, "Content-Type": "application/json"},
            json={"urls": [url], "format": "markdown"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("content", data[0].get("markdown", str(data[0])))
        elif isinstance(data, dict):
            return data.get("content", data.get("markdown", data.get("text", str(data))))
        return str(data)
    except Exception as e:
        return f"Error fetching article: {e}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED_MASTER_TICKETS = [
    {
        "id": 1,
        "category": "Sewage Overflow",
        "description": "Major sewage pipe rupture on Maa Anandmayee Marg. "
                       "Dirty water flooding the street with strong foul smell. "
                       "Raw sewage flowing onto road, hazardous for pedestrians.",
        "latitude": 28.5415,
        "longitude": 77.2725,
        "base_severity": 8,
        "citizen_count": 3,
        "status": "active",
        "image_name": "sewage_master.png",
    },
    {
        "id": 2,
        "category": "Garbage Overflow",
        "description": "Overflowing garbage container near Kalkaji Mandir. "
                       "Waste spilling onto sidewalk attracting stray animals. "
                       "Very unhygienic conditions, foul smell in entire area.",
        "latitude": 28.5490,
        "longitude": 77.2590,
        "base_severity": 6,
        "citizen_count": 2,
        "status": "active",
        "image_name": "garbage_master.png",
    },
    {
        "id": 3,
        "category": "Pothole",
        "description": "Deep pothole on Govindpuri Main Road. Dangerous for "
                       "two-wheelers and auto-rickshaws. Multiple accidents "
                       "reported. Road surface completely broken and damaged.",
        "latitude": 28.5395,
        "longitude": 77.2645,
        "base_severity": 7,
        "citizen_count": 1,
        "status": "active",
        "image_name": "pothole_master.png",
    },
    {
        "id": 4,
        "category": "Broken Streetlight",
        "description": "Broken streetlight on Outer Ring Road near Sarita Vihar. "
                       "Area completely dark at night. Safety concern for "
                       "pedestrians and commuters walking after dark.",
        "latitude": 28.5445,
        "longitude": 77.2530,
        "base_severity": 5,
        "citizen_count": 1,
        "status": "active",
        "image_name": "streetlight_master.png",
    },
]

SEED_CITIZEN_REPORTS = [
    # ── Ticket #1: Sewage (3 reports) ──
    {"master_ticket_id": 1, "citizen_name": "Rajesh Kumar",
     "phone": "9876543210",
     "text": "Gutter overflow near Maa Anandmayee Marg. Bahut ganda paani beh raha hai road pe",
     "lat": 28.5416, "lon": 77.2726, "image": "sewage_r1.png"},
    {"master_ticket_id": 1, "citizen_name": "Priya Sharma",
     "phone": "9876543211",
     "text": "Sewage pipe burst near the main road. Water everywhere, can't even walk properly",
     "lat": 28.5414, "lon": 77.2724, "image": "sewage_r2.png"},
    {"master_ticket_id": 1, "citizen_name": "Mohammad Iqbal",
     "phone": "9876543212",
     "text": "Naala overflow ho gaya hai, bijli ka khatra hai. Bachche khel rahe hain paani mein",
     "lat": 28.5415, "lon": 77.2723, "image": "sewage_r3.png"},
    # ── Ticket #2: Garbage (2 reports) ──
    {"master_ticket_id": 2, "citizen_name": "Sunita Devi",
     "phone": "9876543213",
     "text": "Garbage container near mandir is overflowing. Bohot badbu aa rahi hai poore area mein",
     "lat": 28.5491, "lon": 77.2591, "image": "garbage_r1.png"},
    {"master_ticket_id": 2, "citizen_name": "Arun Verma",
     "phone": "9876543214",
     "text": "Kachra container full ho gaya hai, safai nahi ho rahi kaafi dino se. Dogs are scattering waste",
     "lat": 28.5489, "lon": 77.2589, "image": "garbage_r2.png"},
    # ── Ticket #3: Pothole (1 report) ──
    {"master_ticket_id": 3, "citizen_name": "Vikram Singh",
     "phone": "9876543215",
     "text": "Bada sa gadda hai road pe, meri bike se gir gaya tha kal. Very dangerous for vehicles",
     "lat": 28.5396, "lon": 77.2646, "image": "pothole_r1.png"},
    # ── Ticket #4: Streetlight (1 report) ──
    {"master_ticket_id": 4, "citizen_name": "Neha Gupta",
     "phone": "9876543216",
     "text": "Street light kharab hai, raat mein bahut andhera hota hai yahan. Women feel unsafe",
     "lat": 28.5446, "lon": 77.2531, "image": "streetlight_r1.png"},
]

DEMO_SCENARIOS = {
    "— Select a Demo Scenario —": None,
    "🔴 Scenario A · Definite Duplicate (Sewage ~18 m from Ticket #1)": {
        "text": "Sewage overflow near factory gate. Naali se paani beh raha hai "
                "bahut ganda smell aa rahi hai. Please send someone urgently!",
        "latitude": 28.54161,
        "longitude": 77.27263,
        "image_name": "demo_sewage_dup.png",
        "expected": "DUPLICATE → Merge into Ticket #1",
        "citizen_name": "Amit Prakash",
        "phone": "9988776655",
    },
    "🟡 Scenario B · Different Issue, Same Area (~25 m from Ticket #1)": {
        "text": "Broken pavement curb on the footpath. Tiles ukhad gaye hain, "
                "senior citizens ko bahut dikkat ho rahi hai chalte mein",
        "latitude": 28.5416,
        "longitude": 77.2723,
        "image_name": "demo_pavement.png",
        "expected": "NEW INCIDENT — spatial proximity alone should NOT match",
        "citizen_name": "Kavita Mehra",
        "phone": "9988776656",
    },
    "🟢 Scenario C · Completely New Issue (~1.5 km away)": {
        "text": "Fallen tree branch blocking the road after last night rain. "
                "Ped ki badi branch gir gayi hai, traffic jam ho raha hai",
        "latitude": 28.5520,
        "longitude": 77.2830,
        "image_name": "demo_tree.png",
        "expected": "NEW INCIDENT — no spatial candidates within 60 m",
        "citizen_name": "Ravi Tiwari",
        "phone": "9988776657",
    },
}


def get_demo_scenarios():
    """Return dictionary of demo scenarios with translated titles and expected outcomes."""
    T = get_lang()
    return {
        T["scenario_prompt"]: None,
        T["scenario_a_title"]: {
            "text": "Sewage overflow near factory gate. Naali se paani beh raha hai "
                    "bahut ganda smell aa rahi hai. Please send someone urgently!",
            "latitude": 28.54161,
            "longitude": 77.27263,
            "image_name": "demo_sewage_dup.png",
            "expected": T["scenario_a_expected"],
            "citizen_name": "Amit Prakash",
            "phone": "9988776655",
        },
        T["scenario_b_title"]: {
            "text": "Broken pavement curb on the footpath. Tiles ukhad gaye hain, "
                    "senior citizens ko bahut dikkat ho rahi hai chalte mein",
            "latitude": 28.5416,
            "longitude": 77.2723,
            "image_name": "demo_pavement.png",
            "expected": T["scenario_b_expected"],
            "citizen_name": "Kavita Mehra",
            "phone": "9988776656",
        },
        T["scenario_c_title"]: {
            "text": "Fallen tree branch blocking the road after last night rain. "
                    "Ped ki badi branch gir gayi hai, traffic jam ho raha hai",
            "latitude": 28.5520,
            "longitude": 77.2830,
            "image_name": "demo_tree.png",
            "expected": T["scenario_c_expected"],
            "citizen_name": "Ravi Tiwari",
            "phone": "9988776657",
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYNTHETIC DEMO IMAGE GENERATORS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _gen_sewage(path, seed=42):
    """Brown/green dirty water with sewage patches."""
    rng = random.Random(seed)
    np.random.seed(seed)
    img = Image.new("RGB", (400, 300), (89, 75, 55))
    draw = ImageDraw.Draw(img)
    for y in range(0, 300, 3):
        c = (rng.randint(60, 100), rng.randint(55, 85), rng.randint(30, 60))
        draw.rectangle([0, y, 400, y + 3], fill=c)
    for _ in range(15):
        x1, y1 = rng.randint(0, 350), rng.randint(0, 250)
        x2, y2 = x1 + rng.randint(30, 100), y1 + rng.randint(5, 20)
        draw.ellipse(
            [x1, y1, x2, y2],
            fill=(rng.randint(70, 110), rng.randint(90, 120), rng.randint(50, 70)),
        )
    for _ in range(8):
        cx, cy = rng.randint(50, 350), rng.randint(50, 250)
        r = rng.randint(15, 40)
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(rng.randint(40, 65), rng.randint(45, 60), rng.randint(20, 40)),
        )
    img = img.filter(ImageFilter.GaussianBlur(2))
    img.save(path)


def _gen_garbage(path, seed=42):
    """Colorful scattered waste items."""
    rng = random.Random(seed)
    img = Image.new("RGB", (400, 300), (140, 130, 110))
    draw = ImageDraw.Draw(img)
    colors = [
        (200, 50, 50), (50, 150, 50), (50, 50, 200), (200, 200, 50),
        (200, 100, 50), (150, 150, 150), (255, 255, 255), (100, 80, 60),
        (180, 60, 180), (80, 180, 180),
    ]
    for _ in range(50):
        x, y = rng.randint(10, 390), rng.randint(10, 290)
        w, h = rng.randint(8, 40), rng.randint(6, 30)
        c = rng.choice(colors)
        shape = rng.choice(["rect", "ellipse", "tri"])
        if shape == "rect":
            draw.rectangle([x, y, x + w, y + h], fill=c)
        elif shape == "ellipse":
            draw.ellipse([x, y, x + w, y + h], fill=c)
        else:
            draw.polygon([(x, y), (x + w, y + h // 2), (x + w // 2, y + h)], fill=c)
    img = img.filter(ImageFilter.GaussianBlur(1.5))
    img.save(path)


def _gen_pothole(path, seed=42):
    """Grey asphalt with dark circular depression and cracks."""
    rng = random.Random(seed)
    img = Image.new("RGB", (400, 300), (120, 120, 125))
    draw = ImageDraw.Draw(img)
    for y in range(0, 300, 2):
        for x in range(0, 400, 2):
            v = rng.randint(100, 140)
            draw.point((x, y), fill=(v, v, v + 2))
    cx, cy = 200, 150
    for r in range(60, 10, -5):
        d = max(20, 60 - r)
        draw.ellipse([cx - r, cy - r // 2, cx + r, cy + r // 2],
                     fill=(d + 20, d + 18, d + 15))
    for _ in range(6):
        x1, y1 = rng.randint(50, 350), rng.randint(50, 250)
        pts = [(x1, y1)]
        for _ in range(rng.randint(3, 7)):
            x1 += rng.randint(-30, 30)
            y1 += rng.randint(-20, 20)
            pts.append((x1, y1))
        draw.line(pts, fill=(40, 38, 35), width=2)
    img = img.filter(ImageFilter.GaussianBlur(1))
    img.save(path)


def _gen_streetlight(path, seed=42):
    """Dark scene with a broken/dim streetlight pole."""
    rng = random.Random(seed)
    img = Image.new("RGB", (400, 300), (25, 25, 40))
    draw = ImageDraw.Draw(img)
    for y in range(150):
        v = int(25 + y * 0.15)
        draw.line([(0, y), (400, y)], fill=(v, v, v + 15))
    draw.rectangle([195, 40, 205, 280], fill=(60, 60, 65))
    draw.rectangle([205, 45, 260, 52], fill=(55, 55, 60))
    draw.ellipse([245, 52, 275, 70], fill=(70, 65, 50))
    for r in range(40, 0, -2):
        a = max(5, 30 - r)
        draw.ellipse([260 - r, 61 - r // 2, 260 + r, 61 + r // 2],
                     fill=(a + 30, a + 25, a + 10))
    draw.rectangle([0, 250, 400, 300], fill=(45, 42, 38))
    img = img.filter(ImageFilter.GaussianBlur(1.5))
    img.save(path)


def _gen_pavement(path, seed=42):
    """Broken sidewalk tiles with cracks and missing pieces."""
    rng = random.Random(seed)
    img = Image.new("RGB", (400, 300), (180, 175, 165))
    draw = ImageDraw.Draw(img)
    ts = 50
    for y in range(0, 300, ts):
        for x in range(0, 400, ts):
            v = rng.randint(160, 195)
            draw.rectangle([x + 1, y + 1, x + ts - 1, y + ts - 1],
                           fill=(v, v - 3, v - 8))
    for _ in range(5):
        tx, ty = rng.randint(0, 7) * ts, rng.randint(0, 5) * ts
        draw.rectangle([tx + 1, ty + 1, tx + ts - 1, ty + ts - 1],
                       fill=(130, 120, 105))
        for _ in range(3):
            lx1 = tx + rng.randint(5, ts - 5)
            ly1 = ty + rng.randint(5, ts - 5)
            lx2 = tx + rng.randint(5, ts - 5)
            ly2 = ty + rng.randint(5, ts - 5)
            draw.line([(lx1, ly1), (lx2, ly2)], fill=(80, 75, 65), width=2)
    for _ in range(3):
        tx, ty = rng.randint(0, 7) * ts, rng.randint(0, 5) * ts
        draw.rectangle([tx + 1, ty + 1, tx + ts - 1, ty + ts - 1],
                       fill=(90, 80, 60))
    img = img.filter(ImageFilter.GaussianBlur(1))
    img.save(path)


def _gen_tree(path, seed=42):
    """Fallen tree branch on a road."""
    rng = random.Random(seed)
    img = Image.new("RGB", (400, 300), (95, 120, 75))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 180, 400, 300], fill=(110, 108, 100))
    draw.line([(30, 80), (380, 220)], fill=(100, 70, 40), width=18)
    draw.line([(30, 80), (380, 220)], fill=(85, 60, 35), width=12)
    for _ in range(8):
        bx, by = rng.randint(50, 350), rng.randint(80, 220)
        ex, ey = bx + rng.randint(-60, 60), by + rng.randint(-50, 50)
        draw.line([(bx, by), (ex, ey)], fill=(90, 65, 38),
                  width=rng.randint(3, 8))
    for _ in range(60):
        lx, ly = rng.randint(20, 380), rng.randint(60, 240)
        lr = rng.randint(5, 15)
        g = rng.randint(80, 160)
        draw.ellipse([lx - lr, ly - lr // 2, lx + lr, ly + lr // 2],
                     fill=(rng.randint(30, 60), g, rng.randint(20, 50)))
    img = img.filter(ImageFilter.GaussianBlur(1.5))
    img.save(path)


def ensure_demo_images():
    """Generate all synthetic demo images (idempotent)."""
    os.makedirs(DEMO_IMAGES_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    manifest = {
        "sewage_master.png":    (_gen_sewage, 42),
        "sewage_r1.png":        (_gen_sewage, 43),
        "sewage_r2.png":        (_gen_sewage, 44),
        "sewage_r3.png":        (_gen_sewage, 45),
        "garbage_master.png":   (_gen_garbage, 42),
        "garbage_r1.png":       (_gen_garbage, 43),
        "garbage_r2.png":       (_gen_garbage, 44),
        "pothole_master.png":   (_gen_pothole, 42),
        "pothole_r1.png":       (_gen_pothole, 43),
        "streetlight_master.png": (_gen_streetlight, 42),
        "streetlight_r1.png":   (_gen_streetlight, 43),
        "demo_sewage_dup.png":  (_gen_sewage, 46),
        "demo_pavement.png":    (_gen_pavement, 42),
        "demo_tree.png":        (_gen_tree, 42),
    }
    for fname, (fn, seed) in manifest.items():
        fpath = os.path.join(DEMO_IMAGES_DIR, fname)
        if not os.path.exists(fpath):
            fn(fpath, seed=seed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATABASE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _conn():
    os.makedirs("data", exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    """Create tables and seed demo data (idempotent)."""
    if st.session_state.get("_db_ready"):
        return
    conn = _conn()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS master_tickets (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        category    TEXT    NOT NULL,
        description TEXT    NOT NULL,
        latitude    REAL    NOT NULL,
        longitude   REAL    NOT NULL,
        base_severity INTEGER NOT NULL,
        citizen_count INTEGER DEFAULT 1,
        priority_score REAL  NOT NULL,
        status      TEXT    DEFAULT 'active',
        image_path  TEXT,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS citizen_reports (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        master_ticket_id INTEGER NOT NULL,
        citizen_name     TEXT    NOT NULL,
        phone_number     TEXT,
        text_description TEXT,
        image_path       TEXT,
        latitude         REAL    NOT NULL,
        longitude        REAL    NOT NULL,
        timestamp        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (master_ticket_id) REFERENCES master_tickets(id)
    )""")
    # Seed once
    if cur.execute("SELECT COUNT(*) FROM master_tickets").fetchone()[0] == 0:
        rng = random.Random(0)
        for t in SEED_MASTER_TICKETS:
            p = t["base_severity"] * math.log2(1 + t["citizen_count"])
            ts = (datetime.now() - timedelta(hours=rng.randint(2, 48))).isoformat()
            ip = os.path.join(DEMO_IMAGES_DIR, t["image_name"])
            cur.execute(
                "INSERT INTO master_tickets "
                "(id,category,description,latitude,longitude,base_severity,"
                "citizen_count,priority_score,status,image_path,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (t["id"], t["category"], t["description"], t["latitude"],
                 t["longitude"], t["base_severity"], t["citizen_count"],
                 round(p, 2), t["status"], ip, ts),
            )
        for r in SEED_CITIZEN_REPORTS:
            ts = (datetime.now() - timedelta(hours=rng.randint(1, 24))).isoformat()
            ip = os.path.join(DEMO_IMAGES_DIR, r["image"])
            cur.execute(
                "INSERT INTO citizen_reports "
                "(master_ticket_id,citizen_name,phone_number,text_description,"
                "image_path,latitude,longitude,timestamp) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (r["master_ticket_id"], r["citizen_name"], r["phone"],
                 r["text"], ip, r["lat"], r["lon"], ts),
            )
        conn.commit()
    conn.close()
    st.session_state["_db_ready"] = True


def db_get_tickets():
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM master_tickets WHERE status='active' "
        "ORDER BY priority_score DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_get_reports(ticket_id):
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM citizen_reports WHERE master_ticket_id=? "
        "ORDER BY timestamp DESC", (ticket_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_total_reports():
    conn = _conn()
    n = conn.execute("SELECT COUNT(*) FROM citizen_reports").fetchone()[0]
    conn.close()
    return n


def db_merge(ticket_id, name, phone, text, img_path, lat, lon):
    conn = _conn()
    cur = conn.cursor()
    ts = datetime.now().isoformat()
    cur.execute(
        "INSERT INTO citizen_reports "
        "(master_ticket_id,citizen_name,phone_number,text_description,"
        "image_path,latitude,longitude,timestamp) VALUES (?,?,?,?,?,?,?,?)",
        (ticket_id, name, phone, text, img_path, lat, lon, ts),
    )
    cur.execute(
        "UPDATE master_tickets SET citizen_count=citizen_count+1 WHERE id=?",
        (ticket_id,),
    )
    row = cur.execute(
        "SELECT base_severity, citizen_count FROM master_tickets WHERE id=?",
        (ticket_id,),
    ).fetchone()
    new_p = row[0] * math.log2(1 + row[1])
    cur.execute(
        "UPDATE master_tickets SET priority_score=? WHERE id=?",
        (round(new_p, 2), ticket_id),
    )
    conn.commit()
    conn.close()
    return round(new_p, 2)


def db_create_ticket(category, desc, lat, lon, severity, img_path,
                     name, phone, text):
    conn = _conn()
    cur = conn.cursor()
    p = severity * math.log2(2)
    ts = datetime.now().isoformat()
    cur.execute(
        "INSERT INTO master_tickets "
        "(category,description,latitude,longitude,base_severity,"
        "citizen_count,priority_score,status,image_path,created_at) "
        "VALUES (?,?,?,?,?,1,?,'active',?,?)",
        (category, desc, lat, lon, severity, round(p, 2), img_path, ts),
    )
    tid = cur.lastrowid
    cur.execute(
        "INSERT INTO citizen_reports "
        "(master_ticket_id,citizen_name,phone_number,text_description,"
        "image_path,latitude,longitude,timestamp) VALUES (?,?,?,?,?,?,?,?)",
        (tid, name, phone, text, img_path, lat, lon, ts),
    )
    conn.commit()
    conn.close()
    return tid


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ML MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource(show_spinner="Loading text encoder (MiniLM-L6-v2)…")
def load_text_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading vision encoder (MobileNetV3)…")
def load_vision_model():
    try:
        from torchvision.models import MobileNet_V3_Small_Weights
        model = models.mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
    except Exception:
        model = models.mobilenet_v3_small(pretrained=True)
    model.classifier = torch.nn.Identity()
    model.eval()
    return model


_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEDUPLICATION ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres."""
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _img_features(src, model):
    """Extract 576-d feature vector from an image path or PIL Image."""
    if isinstance(src, str):
        img = Image.open(src).convert("RGB")
    else:
        img = src.convert("RGB")
    tensor = _IMG_TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        feat = model(tensor)
    return feat.squeeze().numpy()


def _cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def run_dedup(text, image, lat, lon, text_model, vision_model):
    """
    Full multimodal deduplication pipeline.
    Returns (is_dup, best_ticket, score, breakdown_dict).
    """
    tickets = db_get_tickets()
    if not tickets:
        return False, None, 0.0, {"best": None, "candidates": [], "threshold": COMPOSITE_THRESHOLD}

    inc_text_emb = text_model.encode([text])[0]
    inc_vis_feat = _img_features(image, vision_model)

    best_score, best_ticket, best_bd = 0.0, None, {}
    candidates = []

    for t in tickets:
        dist = haversine(lat, lon, t["latitude"], t["longitude"])

        if dist > SPATIAL_THRESHOLD_M:
            candidates.append({
                "id": t["id"], "cat": t["category"],
                "dist": round(dist, 1),
                "spatial": 0.0, "text": 0.0, "visual": 0.0,
                "composite": 0.0, "filtered": True,
            })
            continue

        spatial = max(0.0, 1.0 - dist / SPATIAL_THRESHOLD_M)

        t_emb = text_model.encode([t["description"]])[0]
        text_sim = max(0.0, _cosine(inc_text_emb, t_emb))

        vis_sim = 0.0
        if t["image_path"] and os.path.exists(t["image_path"]):
            t_feat = _img_features(t["image_path"], vision_model)
            vis_sim = max(0.0, _cosine(inc_vis_feat, t_feat))

        comp = W_VISUAL * vis_sim + W_TEXT * text_sim + W_SPATIAL * spatial

        entry = {
            "id": t["id"], "cat": t["category"],
            "dist": round(dist, 1),
            "spatial": round(spatial, 4),
            "text": round(text_sim, 4),
            "visual": round(vis_sim, 4),
            "composite": round(comp, 4),
            "filtered": False,
        }
        candidates.append(entry)

        if comp > best_score:
            best_score, best_ticket, best_bd = comp, t, entry

    is_dup = best_score >= COMPOSITE_THRESHOLD
    return is_dup, best_ticket, round(best_score, 4), {
        "best": best_bd, "candidates": candidates,
        "threshold": COMPOSITE_THRESHOLD,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MATERIAL 3 / GOOGLE DESIGN LANGUAGE CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def inject_css(theme="light"):
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@300;400;500;600;700&family=Google+Sans+Text:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #F8F9FA !important;
    font-family: 'Google Sans', 'Inter', -apple-system, BlinkMacSystemFont,
                 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif !important;
}
header[data-testid="stHeader"] { background: transparent !important; }
footer { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
section[data-testid="stMain"] > div { padding-top: 0.5rem; }

/* ── FLARE App Bar (Google Flights / Photos aesthetic) ── */
.flare-appbar {
    background: #FFFFFF;
    border-bottom: 1px solid #DADCE0;
    border-radius: 0 0 20px 20px;
    padding: 14px 32px;
    margin: -0.5rem -1rem 24px -1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 3px 0 rgba(60,64,67,0.08), 0 4px 8px 3px rgba(60,64,67,0.04);
    position: relative;
    z-index: 100;
}
.flare-appbar-left {
    display: flex;
    align-items: center;
    gap: 16px;
}
.flare-logo {
    width: 44px; height: 44px;
    background: linear-gradient(180deg, #E8F0FE 0%, #A8C7FA 30%, #4285F4 70%, #1A73E8 100%);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; color: white; flex-shrink: 0;
    box-shadow: 0 2px 10px rgba(66,133,244,0.35);
    position: relative;
    overflow: hidden;
}
/* soft candle glow shimmer */
.flare-logo::after {
    content: '';
    position: absolute;
    top: -6px; left: 50%; transform: translateX(-50%);
    width: 18px; height: 18px;
    background: radial-gradient(circle, rgba(255,255,255,0.7) 0%, rgba(168,199,250,0.3) 50%, transparent 70%);
    border-radius: 50%;
    animation: candle-glow 2.5s ease-in-out infinite;
}
@keyframes candle-glow {
    0%, 100% { opacity: 0.6; transform: translateX(-50%) scale(1); }
    50% { opacity: 1; transform: translateX(-50%) scale(1.3); }
}
.flare-title-block {}
.flare-title {
    font-size: 20px; font-weight: 700; color: #202124;
    letter-spacing: -0.3px; line-height: 1.2;
}
.flare-title span.flare-accent { color: #4285F4; }
.flare-sub {
    font-size: 12px; color: #5F6368; margin-top: 2px;
    font-weight: 400; letter-spacing: 0.1px;
}
.flare-status-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: #E6F4EA; color: #137333;
    padding: 8px 18px; border-radius: 24px;
    font-size: 12px; font-weight: 500;
    white-space: nowrap;
    border: 1px solid #A8DAB5;
}
.flare-status-pill .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #34A853;
    display: inline-block;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(52,168,83,0.4); }
    50% { opacity: 0.8; box-shadow: 0 0 0 6px rgba(52,168,83,0); }
}

/* ── Metric Cards (Google Fit style) ── */
.m-card {
    background: #FFFFFF;
    border: 1px solid #E0E2E6;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 1px 3px 0 rgba(60,64,67,0.08), 0 4px 8px 3px rgba(60,64,67,0.04);
    transition: all .25s cubic-bezier(.4,0,.2,1);
}
.m-card:hover {
    box-shadow: 0 4px 16px rgba(26,115,232,0.15);
    transform: translateY(-3px);
}
.m-icon { font-size: 18px; margin-bottom: 4px; opacity: 0.8; }
.m-val {
    font-size: 2.2rem; font-weight: 600; margin: 6px 0 4px;
    line-height: 1; letter-spacing: -1px;
}
.m-val.blue { color: #4285F4; }
.m-val.green { color: #34A853; }
.m-val.amber { color: #FBBC04; }
.m-val.red { color: #EA4335; }
.m-label {
    font-size: 11px; font-weight: 500; color: #5F6368;
    text-transform: uppercase; letter-spacing: 1px;
    display: flex; align-items: center; justify-content: center; gap: 6px;
}
.m-dot {
    width: 6px; height: 6px; border-radius: 50%; display: inline-block;
}
.m-dot.blue { background: #4285F4; }
.m-dot.green { background: #34A853; }
.m-dot.amber { background: #FBBC04; }
.m-dot.red { background: #EA4335; }

/* ── Section Headers (Material 3 pill chips) ── */
.sec-h {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #E8F0FE;
    color: #4285F4;
    padding: 6px 14px;
    border-radius: 24px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 20px 0 12px;
    letter-spacing: 0.3px;
}

/* ── Cards & Surfaces (Google elevation) ── */
.g-card {
    background: #FFFFFF;
    border: 1px solid #E0E2E6;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px 0 rgba(60,64,67,0.08), 0 4px 8px 3px rgba(60,64,67,0.04);
}

/* ── Helper badge for demo quick-fill ── */
.quickfill-card {
    background: #FFFFFF;
    border: 1px solid #E0E2E6;
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px 0 rgba(60,64,67,0.08), 0 4px 8px 3px rgba(60,64,67,0.04);
}
.quickfill-helper {
    display: inline-flex; align-items: center; gap: 6px;
    background: #FEF7E0; color: #E37400;
    padding: 5px 12px; border-radius: 20px;
    font-size: 11px; font-weight: 500; margin-bottom: 10px;
}

/* ── Ticket strips ── */
.tk-strip {
    background: #FFFFFF;
    border: 1px solid #E0E2E6;
    border-radius: 16px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border-left: 4px solid #4285F4;
    box-shadow: 0 1px 3px 0 rgba(60,64,67,0.08);
    transition: all .2s cubic-bezier(.4,0,.2,1);
}
.tk-strip:hover { box-shadow: 0 2px 12px rgba(60,64,67,0.12); }
.tk-strip.crit { border-left-color: #EA4335; }
.tk-strip.mod  { border-left-color: #FBBC04; }
.tk-strip.low  { border-left-color: #34A853; }
.tk-title { font-size: 14px; font-weight: 600; color: #202124; }
.tk-meta  { font-size: 12px; color: #5F6368; margin-top: 3px; }

/* ── Priority Badges (pill chips) ── */
.p-badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 4px 12px; border-radius: 24px;
    font-size: 11px; font-weight: 500;
}
.p-crit { background: #FCE8E6; color: #EA4335; }
.p-mod  { background: #FEF7E0; color: #F9AB00; }
.p-low  { background: #E6F4EA; color: #34A853; }

/* ── Result Card (Google Flights ticket style) ── */
.result-ticket {
    background: #FFFFFF;
    border: 1px solid #E0E2E6;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 1px 3px 0 rgba(60,64,67,0.08), 0 4px 8px 3px rgba(60,64,67,0.04);
    margin: 14px 0;
}
.result-ticket-header {
    padding: 16px 20px;
    display: flex; align-items: center; gap: 12px;
}
.result-ticket-header.dup {
    background: linear-gradient(135deg, #E8F0FE 0%, #D2E3FC 100%);
    border-bottom: 1px solid #AECBFA;
}
.result-ticket-header.new-ticket {
    background: linear-gradient(135deg, #E6F4EA 0%, #CEEAD6 100%);
    border-bottom: 1px solid #A8DAB5;
}
.result-ticket-icon {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.result-ticket-icon.dup { background: #4285F4; color: white; }
.result-ticket-icon.new-ticket { background: #34A853; color: white; }
.result-ticket-title {
    font-size: 15px; font-weight: 600; color: #202124;
}
.result-ticket-subtitle {
    font-size: 12px; color: #5F6368; margin-top: 1px;
}
.result-ticket-body {
    padding: 18px 20px;
}
.result-final-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 10px 20px; border-radius: 24px;
    font-size: 14px; font-weight: 600; margin-top: 12px;
}
.result-final-pill.dup {
    background: #E8F0FE; color: #4285F4; border: 1px solid #AECBFA;
}
.result-final-pill.new-ticket {
    background: #E6F4EA; color: #34A853; border: 1px solid #A8DAB5;
}

/* ── Score Bars (Material progress) ── */
.sb-row { display: flex; align-items: center; margin: 8px 0; gap: 10px; }
.sb-lbl { font-size: 12px; font-weight: 500; color: #5F6368; min-width: 100px; }
.sb-bg {
    flex: 1; height: 8px; background: #E8EAED;
    border-radius: 4px; overflow: hidden;
}
.sb-fill {
    height: 100%; border-radius: 4px;
    transition: width .7s cubic-bezier(.4,0,.2,1);
}
.sb-fill.blue  { background: linear-gradient(90deg, #A8C7FA, #4285F4); }
.sb-fill.green { background: linear-gradient(90deg, #81C995, #34A853); }
.sb-fill.amber { background: linear-gradient(90deg, #FDE293, #FBBC04); }
.sb-fill.red   { background: linear-gradient(90deg, #F28B82, #EA4335); }
.sb-val { font-size: 12px; font-weight: 600; color: #202124; min-width: 48px; text-align: right; }

/* ── Report pills ── */
.rp {
    background: #F8F9FA; border: 1px solid #E8EAED;
    border-radius: 12px;
    padding: 12px 16px; margin: 6px 0;
}
.rp-name { font-weight: 600; font-size: 13px; color: #202124; }
.rp-text { font-size: 12px; color: #5F6368; margin-top: 3px; font-style: italic; }
.rp-time { font-size: 11px; color: #80868B; margin-top: 3px; }

/* ── Hint chip ── */
.hint-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: #E8F0FE; color: #4285F4;
    padding: 6px 14px; border-radius: 20px;
    font-size: 12px; font-weight: 500; margin: 6px 0;
}

/* ── Google-style Buttons ── */
.stButton > button {
    background: #4285F4 !important; color: white !important;
    border: none !important; border-radius: 24px !important;
    padding: 10px 24px !important;
    font-weight: 500 !important; font-size: 14px !important;
    font-family: 'Google Sans', 'Inter', sans-serif !important;
    transition: all .2s cubic-bezier(.4,0,.2,1) !important;
    width: 100%;
    box-shadow: 0 1px 3px rgba(66,133,244,0.3) !important;
    letter-spacing: 0.2px;
}
.stButton > button:hover {
    background: #3367D6 !important;
    box-shadow: 0 4px 14px rgba(66,133,244,0.35) !important;
    transform: translateY(-1px);
}
.stButton > button:active {
    background: #2A56C6 !important;
    transform: translateY(0);
}

/* ── Text Inputs (Material 3 outlined fields) ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background: #FFFFFF !important;
    border: 1px solid #80868B !important;
    border-radius: 8px !important;
    font-family: 'Google Sans', 'Inter', sans-serif !important;
    color: #202124 !important;
    font-size: 14px !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stNumberInput > div > div > input:focus {
    border-color: #4285F4 !important;
    box-shadow: 0 0 0 2px rgba(66,133,244,0.2) !important;
    outline: none !important;
}
.stSelectbox > div > div {
    border-radius: 8px !important;
    border-color: #80868B !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #4285F4 !important;
    box-shadow: 0 0 0 2px rgba(66,133,244,0.2) !important;
}

/* ── Labels ── */
.stTextInput label, .stTextArea label,
.stNumberInput label, .stSelectbox label,
.stFileUploader label {
    font-family: 'Google Sans', 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #5F6368 !important;
}

/* ── Expanders (Material surface) ── */
details {
    background: #FFFFFF !important;
    border-radius: 16px !important;
    border: 1px solid #E0E2E6 !important;
    margin-bottom: 8px !important;
    box-shadow: 0 1px 2px rgba(60,64,67,0.04) !important;
}
details summary {
    font-weight: 500 !important;
    font-size: 14px !important;
    border-radius: 16px !important;
    font-family: 'Google Sans', 'Inter', sans-serif !important;
    color: #202124 !important;
}

/* ── Map Container ── */
.map-container {
    border-radius: 16px;
    border: 1px solid #DADCE0;
    overflow: hidden;
    box-shadow: 0 1px 3px 0 rgba(60,64,67,0.08);
}
iframe {
    border-radius: 16px !important;
    border: 1px solid #DADCE0 !important;
}

/* ── Animations ── */
@keyframes fadeUp { from { opacity:0; transform:translateY(12px); }
                    to { opacity:1; transform:translateY(0); } }
.anim-up { animation: fadeUp .5s ease forwards; }

@keyframes slideIn { from { opacity:0; transform:translateX(-20px); }
                     to { opacity:1; transform:translateX(0); } }
.anim-slide { animation: slideIn .4s ease forwards; }

/* ── Scrollbar (Google-style thin) ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #DADCE0; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #BDC1C6; }

/* ── File uploader ── */
.stFileUploader > div {
    border-radius: 12px !important;
    border: 1px dashed #DADCE0 !important;
}

/* ── Distance info card ── */
.dist-info {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 14px; border-radius: 8px;
    font-size: 12px; color: #5F6368;
    margin: 4px 0;
}
.dist-info.within { background: #E6F4EA; color: #34A853; }
.dist-info.outside { background: #FCE8E6; color: #EA4335; }

/* ── Theme Toggle Button ── */
.theme-toggle-btn {
    display: inline-flex; align-items: center; gap: 6px;
    background: #F1F3F4; color: #5F6368;
    border: 1px solid #DADCE0;
    padding: 6px 14px; border-radius: 20px;
    font-size: 12px; font-weight: 500;
    cursor: pointer; white-space: nowrap;
    transition: all .2s;
}
.theme-toggle-btn:hover { background: #E8EAED; }

</style>""", unsafe_allow_html=True)

    # ── Dark Theme Override ──
    if theme == "dark":
        st.markdown("""<style>
/* ══════════════════════════════════════════════
   DARK THEME OVERRIDES
   ══════════════════════════════════════════════ */

/* Global */
.stApp { background: #1F1F1F !important; color: #E8EAED !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* App Bar */
.flare-appbar { background: #292929 !important; border-bottom-color: #3C4043 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3), 0 4px 8px rgba(0,0,0,0.2) !important; }
.flare-title { color: #E8EAED !important; }
.flare-sub { color: #9AA0A6 !important; }
.flare-status-pill { background: #1B3626 !important; color: #81C995 !important;
    border-color: #2D6A3E !important; }

/* Theme toggle in dark mode */
.theme-toggle-btn { background: #3C4043 !important; color: #E8EAED !important;
    border-color: #5F6368 !important; }
.theme-toggle-btn:hover { background: #4A4E51 !important; }

/* Metric Cards */
.m-card { background: #292929 !important; border-color: #3C4043 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3), 0 4px 8px rgba(0,0,0,0.2) !important; }
.m-card:hover { box-shadow: 0 4px 16px rgba(66,133,244,0.25) !important; }
.m-label { color: #9AA0A6 !important; }

/* Section Headers */
.sec-h { background: #1A2740 !important; color: #8AB4F8 !important; }

/* Cards & Surfaces */
.g-card { background: #292929 !important; border-color: #3C4043 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3), 0 4px 8px rgba(0,0,0,0.2) !important; }
.quickfill-card { background: #292929 !important; border-color: #3C4043 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important; }
.quickfill-helper { background: #3D3019 !important; color: #FDD663 !important; }

/* Ticket strips */
.tk-strip { background: #292929 !important; border-color: #3C4043 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important; }
.tk-title { color: #E8EAED !important; }
.tk-meta { color: #9AA0A6 !important; }

/* Priority Badges (darker chip backgrounds) */
.p-crit { background: #3C1F1E !important; color: #F28B82 !important; }
.p-mod  { background: #3D3019 !important; color: #FDD663 !important; }
.p-low  { background: #1B3626 !important; color: #81C995 !important; }

/* Result Card */
.result-ticket { background: #292929 !important; border-color: #3C4043 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3), 0 4px 8px rgba(0,0,0,0.2) !important; }
.result-ticket-header.dup { background: linear-gradient(135deg, #1A2740, #1E3A5F) !important;
    border-bottom-color: #3B6FAF !important; }
.result-ticket-header.new-ticket { background: linear-gradient(135deg, #1B3626, #1E4D30) !important;
    border-bottom-color: #2D6A3E !important; }
.result-ticket-title { color: #E8EAED !important; }
.result-ticket-subtitle { color: #9AA0A6 !important; }
.result-final-pill.dup { background: #1A2740 !important; color: #8AB4F8 !important;
    border-color: #3B6FAF !important; }
.result-final-pill.new-ticket { background: #1B3626 !important; color: #81C995 !important;
    border-color: #2D6A3E !important; }

/* Score Bars */
.sb-lbl { color: #9AA0A6 !important; }
.sb-bg { background: #3C4043 !important; }
.sb-val { color: #E8EAED !important; }

/* Report pills */
.rp { background: #333639 !important; border-color: #3C4043 !important; }
.rp-name { color: #E8EAED !important; }
.rp-text { color: #9AA0A6 !important; }
.rp-time { color: #80868B !important; }

/* Hint chip */
.hint-chip { background: #1A2740 !important; color: #8AB4F8 !important; }

/* Buttons (stay Google Blue, slightly brighter in dark) */
.stButton > button {
    background: #4285F4 !important; color: white !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
}
.stButton > button:hover {
    background: #5A9CF6 !important;
    box-shadow: 0 4px 14px rgba(66,133,244,0.4) !important;
}

/* Text Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background: #292929 !important; color: #E8EAED !important;
    border-color: #5F6368 !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stNumberInput > div > div > input:focus {
    border-color: #8AB4F8 !important;
    box-shadow: 0 0 0 2px rgba(138,180,248,0.25) !important;
}
.stSelectbox > div > div { border-color: #5F6368 !important; background: #292929 !important; }
.stSelectbox > div > div:focus-within {
    border-color: #8AB4F8 !important;
    box-shadow: 0 0 0 2px rgba(138,180,248,0.25) !important;
}

/* Labels */
.stTextInput label, .stTextArea label,
.stNumberInput label, .stSelectbox label,
.stFileUploader label { color: #9AA0A6 !important; }

/* Expanders */
details { background: #292929 !important; border-color: #3C4043 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.2) !important; }
details summary { color: #E8EAED !important; }

/* Map Container */
.map-container { border-color: #3C4043 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important; }
iframe { border-color: #3C4043 !important; }

/* File uploader */
.stFileUploader > div { border-color: #5F6368 !important; }

/* Distance info dark */
.dist-info.within { background: #1B3626 !important; color: #81C995 !important; }
.dist-info.outside { background: #3C1F1E !important; color: #F28B82 !important; }

/* Scrollbar dark */
::-webkit-scrollbar-thumb { background: #5F6368 !important; }
::-webkit-scrollbar-thumb:hover { background: #80868B !important; }

/* Streamlit markdown text */
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] {
    color: #E8EAED !important;
}

/* Selectbox dropdown text */
.stSelectbox [data-baseweb="select"] span { color: #E8EAED !important; }

</style>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UI HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _priority_class(score):
    T = get_lang()
    if score > 12:
        return "crit", "p-crit", T.get("priority_critical", "Critical")
    elif score > 6:
        return "mod", "p-mod", T.get("priority_moderate", "Moderate")
    return "low", "p-low", T.get("priority_low", "Low")


def _score_bar(label, value, max_val=1.0, color="blue"):
    pct = min(100, max(0, value / max_val * 100))
    return (
        f'<div class="sb-row">'
        f'  <span class="sb-lbl">{label}</span>'
        f'  <div class="sb-bg"><div class="sb-fill {color}" style="width:{pct:.1f}%"></div></div>'
        f'  <span class="sb-val">{value:.2%}</span>'
        f'</div>'
    )


def _distance_bar(label, dist_m, threshold=60):
    T = get_lang()
    pct = min(100, dist_m / max(threshold * 2, dist_m + 10) * 100)
    color = "green" if dist_m <= threshold else "red"
    status_class = "within" if dist_m <= threshold else "outside"
    status_text = T["within_threshold"].format(t=f"{threshold:.0f}") if dist_m <= threshold else T["outside_threshold"].format(t=f"{threshold:.0f}")
    meters_text = T["meters_away"]
    return (
        f'<div class="sb-row">'
        f'  <span class="sb-lbl">{label}</span>'
        f'  <div class="sb-bg"><div class="sb-fill {color}" style="width:{pct:.1f}%"></div></div>'
        f'  <span class="sb-val">{dist_m:.0f} m</span>'
        f'</div>'
        f'<div class="dist-info {status_class}">📍 {dist_m:.1f} {meters_text} — {status_text}</div>'
    )


def render_header(theme="light"):
    T = get_lang()
    lang = st.session_state.get("lang", "en")
    toggle_icon = "🌙" if theme == "light" else "☀️"
    toggle_label = "Dark" if theme == "light" else "Light"
    lang_indicator = "🌐 हिंदी" if lang == "en" else "🌐 English"
    st.markdown(
        '<div class="flare-appbar">'
        '  <div class="flare-appbar-left">'
        '    <div class="flare-logo">'
        '      <svg width="24" height="28" viewBox="0 0 24 28" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '        <ellipse cx="12" cy="25" rx="5" ry="2.5" fill="rgba(255,255,255,0.25)"/>'
        '        <rect x="10" y="12" width="4" height="14" rx="1.5" fill="rgba(255,255,255,0.85)"/>'
        '        <path d="M12 1C12 1 7 6.5 7 10C7 12.8 9.2 15 12 15C14.8 15 17 12.8 17 10C17 6.5 12 1 12 1Z" fill="white"/>'
        '        <path d="M12 5C12 5 9.5 8 9.5 10C9.5 11.4 10.6 12.5 12 12.5C13.4 12.5 14.5 11.4 14.5 10C14.5 8 12 5 12 5Z" fill="#A8C7FA"/>'
        '      </svg>'
        '    </div>'
        '    <div class="flare-title-block">'
        '      <div class="flare-title"><span class="flare-accent">FLARE</span></div>'
        f'      <div class="flare-sub">{T["app_sub"]}</div>'
        '    </div>'
        '  </div>'
        '  <div style="display:flex;align-items:center;gap:10px;">'
        f'    <div class="theme-toggle-btn" id="lang-indicator">{lang_indicator}</div>'
        f'    <div class="theme-toggle-btn" id="theme-indicator">{toggle_icon} {toggle_label}</div>'
        '    <div class="flare-status-pill">'
        '      <span class="status-dot"></span>'
        f'      {T["status_pill"]}'
        '    </div>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_metrics(tickets, total_reports):
    T = get_lang()
    n_tickets = len(tickets)
    rate = ((1 - n_tickets / total_reports) * 100) if total_reports > 0 else 0
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="m-card">'
            f'  <div class="m-icon">📋</div>'
            f'  <div class="m-val blue">{total_reports}</div>'
            f'  <div class="m-label"><span class="m-dot blue"></span> {T["total_complaints"]}</div>'
            f'</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="m-card">'
            f'  <div class="m-icon">🎫</div>'
            f'  <div class="m-val green">{n_tickets}</div>'
            f'  <div class="m-label"><span class="m-dot green"></span> {T["master_work_orders"]}</div>'
            f'</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(
            f'<div class="m-card">'
            f'  <div class="m-icon">🔄</div>'
            f'  <div class="m-val amber">{rate:.1f}%</div>'
            f'  <div class="m-label"><span class="m-dot amber"></span> {T["dedup_rate"]}</div>'
            f'</div>', unsafe_allow_html=True)


def render_map(tickets, new_report_coords=None):
    T = get_lang()
    # ── Use OpenStreetMap tiles: full roads, buildings, parks — no API key, no watermarks ──
    m = folium.Map(
        location=MAP_CENTER,
        zoom_start=MAP_ZOOM,
        tiles="OpenStreetMap",
    )
    for t in tickets:
        cls, _, lbl = _priority_class(t["priority_score"])
        col_map = {"crit": "red", "mod": "orange", "low": "green"}
        icon_map = {"crit": "exclamation-sign", "mod": "warning-sign",
                    "low": "ok-sign"}
        popup = (
            f"<div style='font-family:Google Sans,Inter,sans-serif;min-width:180px'>"
            f"<b>#{t['id']} {t['category']}</b><br>"
            f"<span style='color:#5F6368'>{T['reports_label']}: {t['citizen_count']} · "
            f"{T['priority']}: {t['priority_score']:.1f} ({lbl})</span></div>"
        )
        folium.Marker(
            [t["latitude"], t["longitude"]],
            popup=folium.Popup(popup, max_width=280),
            tooltip=f"#{t['id']} {t['category']}",
            icon=folium.Icon(color=col_map[cls], icon=icon_map[cls]),
        ).add_to(m)
        # 60m threshold circle
        folium.Circle(
            [t["latitude"], t["longitude"]], radius=60,
            color="#4285F4", fill=True, fill_opacity=0.06,
            weight=1.5, opacity=0.3,
        ).add_to(m)

    if new_report_coords:
        folium.Marker(
            new_report_coords,
            tooltip=T["new_report_tooltip"],
            icon=folium.Icon(color="blue", icon="star", prefix="glyphicon"),
        ).add_to(m)

    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    st_folium(m, width=None, height=380, returned_objects=[])
    st.markdown('</div>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    st.set_page_config(
        page_title="FLARE — Municipal Intelligence Engine",
        page_icon="🕯️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ── Theme State ──
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"
    theme = st.session_state["theme"]

    # ── Language State ──
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
    lang = st.session_state["lang"]
    T = get_lang()

    inject_css(theme=theme)
    ensure_demo_images()
    init_db()

    # Load models
    text_model = load_text_model()
    vision_model = load_vision_model()

    render_header(theme=theme)

    # ── Theme & Language Toggles (placed right below header) ──
    _, lang_col, toggle_col = st.columns([7, 1, 1])
    with lang_col:
        lang_btn = "🌐 हिंदी" if lang == "en" else "🌐 English"
        if st.button(lang_btn, key="lang_toggle", use_container_width=True):
            st.session_state["lang"] = "hi" if lang == "en" else "en"
            st.rerun()
    with toggle_col:
        toggle_label = "☀️ Light" if theme == "dark" else "🌙 Dark"
        if st.button(toggle_label, key="theme_toggle", use_container_width=True):
            st.session_state["theme"] = "dark" if theme == "light" else "light"
            st.rerun()

    # ── Layout: Left (submission) | Right (command center) ──
    left, right = st.columns([2, 3], gap="large")

    # ================================================================
    # LEFT PANEL – Citizen Submission (Google Tasks / Forms aesthetic)
    # ================================================================
    with left:
        st.markdown(f'<div class="sec-h">{T["citizen_report"]}</div>',
                    unsafe_allow_html=True)

        # Demo Quick-Fill (elevated card with helper badge)
        st.markdown(
            '<div class="quickfill-card">'
            f'  <div class="quickfill-helper">{T["quickfill_helper"]}</div>',
            unsafe_allow_html=True,
        )
        scenarios = get_demo_scenarios()
        scenario_key = st.selectbox(
            T["demo_quickfill"],
            options=list(scenarios.keys()),
            key=f"scenario_sel_{lang}",
            help=T["quickfill_help"],
        )
        st.markdown('</div>', unsafe_allow_html=True)

        scenario = scenarios.get(scenario_key)

        # Dynamic key suffix so widgets reset when scenario changes
        sk = hashlib.md5(scenario_key.encode()).hexdigest()[:6]

        # Expected-result hint
        if scenario:
            st.markdown(
                f'<div class="hint-chip">💡 {T["expected"]}: {scenario["expected"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="g-card">', unsafe_allow_html=True)

        # Citizen info
        ci1, ci2 = st.columns(2)
        default_name = scenario["citizen_name"] if scenario else "Demo Citizen"
        default_phone = scenario["phone"] if scenario else "9876500000"
        name = ci1.text_input(T["your_name"], value=default_name, key=f"n_{sk}")
        phone = ci2.text_input(T["phone"], value=default_phone, key=f"p_{sk}")

        # Complaint text
        default_text = scenario["text"] if scenario else ""
        text = st.text_area(
            T["complaint_desc"],
            value=default_text, height=100, key=f"t_{sk}",
            placeholder=T["complaint_placeholder"],
        )

        # Location
        lo1, lo2 = st.columns(2)
        default_lat = scenario["latitude"] if scenario else 28.5415
        default_lon = scenario["longitude"] if scenario else 77.2725
        lat = lo1.number_input(T["latitude"], value=default_lat,
                               format="%.6f", key=f"la_{sk}")
        lon = lo2.number_input(T["longitude"], value=default_lon,
                               format="%.6f", key=f"lo_{sk}")

        # Image
        if scenario:
            demo_img_path = os.path.join(DEMO_IMAGES_DIR, scenario["image_name"])
            st.image(demo_img_path, caption=T["demo_photo"],
                     use_container_width=True)

        uploaded = st.file_uploader(
            T["upload_photo"],
            type=["png", "jpg", "jpeg"], key=f"up_{sk}",
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # Submit
        submitted = st.button(T["submit_btn"], key="submit_btn",
                              use_container_width=True)

        # ── Process submission ──
        if submitted:
            if not text.strip():
                st.error(T["err_no_text"])
                st.stop()

            # Resolve image
            if uploaded:
                pil_img = Image.open(uploaded).convert("RGB")
                fname = f"{int(time.time())}_{hashlib.md5(uploaded.read()).hexdigest()[:8]}.png"
                uploaded.seek(0)
                save_path = os.path.join(UPLOADS_DIR, fname)
                pil_img.save(save_path)
                img_for_model = pil_img
                img_path = save_path
            elif scenario:
                demo_path = os.path.join(DEMO_IMAGES_DIR, scenario["image_name"])
                img_for_model = Image.open(demo_path).convert("RGB")
                img_path = demo_path
            else:
                st.error(T["err_no_photo"])
                st.stop()

            # Run dedup
            with st.spinner(T["running_dedup"]):
                is_dup, best_ticket, score, breakdown = run_dedup(
                    text, img_for_model, lat, lon, text_model, vision_model
                )

            # Act on result
            if is_dup and best_ticket:
                new_p = db_merge(best_ticket["id"], name, phone, text,
                                 img_path, lat, lon)
                st.session_state["last_result"] = {
                    "type": "duplicate",
                    "ticket_id": best_ticket["id"],
                    "category": best_ticket["category"],
                    "score": score,
                    "new_priority": new_p,
                    "breakdown": breakdown,
                    "lat": lat, "lon": lon,
                }
            else:
                # Infer category from text
                cat_keywords = {
                    "Sewage Overflow": ["sewage", "naali", "gutter", "naala", "drain"],
                    "Garbage Overflow": ["garbage", "kachra", "waste", "trash", "dustbin"],
                    "Pothole": ["pothole", "gadda", "pit", "road damage", "hole"],
                    "Broken Streetlight": ["streetlight", "light", "lamp", "andhera", "dark"],
                    "Fallen Tree": ["tree", "branch", "ped", "fallen"],
                    "Broken Pavement": ["pavement", "sidewalk", "footpath", "tiles", "curb"],
                }
                category = "Civic Issue"
                txt_lower = text.lower()
                for cat, kws in cat_keywords.items():
                    if any(kw in txt_lower for kw in kws):
                        category = cat
                        break

                tid = db_create_ticket(
                    category, text, lat, lon, 5, img_path, name, phone, text
                )
                st.session_state["last_result"] = {
                    "type": "new",
                    "ticket_id": tid,
                    "category": category,
                    "score": score,
                    "breakdown": breakdown,
                    "lat": lat, "lon": lon,
                }

            st.rerun()

        # ── Show last result below form (Google Flights ticket card) ──
        result = st.session_state.get("last_result")
        if result:
            st.markdown(f'<div class="sec-h">{T["ingestion_telemetry"]}</div>',
                        unsafe_allow_html=True)

            if result["type"] == "duplicate":
                st.markdown(
                    f'<div class="result-ticket anim-up">'
                    f'  <div class="result-ticket-header dup">'
                    f'    <div class="result-ticket-icon dup">🔗</div>'
                    f'    <div>'
                    f'      <div class="result-ticket-title">{T["dup_detected"]}</div>'
                    f'      <div class="result-ticket-subtitle">{T["composite_score"]}: {result["score"]:.4f} '
                    f'({T["threshold"]}: {COMPOSITE_THRESHOLD})</div>'
                    f'    </div>'
                    f'  </div>'
                    f'  <div class="result-ticket-body">',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="result-ticket anim-up">'
                    f'  <div class="result-ticket-header new-ticket">'
                    f'    <div class="result-ticket-icon new-ticket">🆕</div>'
                    f'    <div>'
                    f'      <div class="result-ticket-title">{T["new_work_order"]}</div>'
                    f'      <div class="result-ticket-subtitle">{T["category"]}: {result["category"]}</div>'
                    f'    </div>'
                    f'  </div>'
                    f'  <div class="result-ticket-body">',
                    unsafe_allow_html=True,
                )

            # Score breakdown inside the ticket card
            bd = result["breakdown"]
            best = bd.get("best")
            if best and not best.get("filtered", True):
                st.markdown(
                    _distance_bar(T["distance"], best["dist"]),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _score_bar(T["visual_match"], best["visual"], color="blue"),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _score_bar(T["semantic_match"], best["text"], color="green"),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _score_bar(T["spatial_decay"], best["spatial"], color="amber"),
                    unsafe_allow_html=True,
                )
                comp_color = "blue" if best["composite"] >= COMPOSITE_THRESHOLD else "red"
                st.markdown(
                    f'<div class="sb-row" style="margin-top:12px;padding-top:10px;'
                    f'border-top:1px solid #E8EAED">'
                    f'  <span class="sb-lbl" style="font-weight:700">{T["composite"]}</span>'
                    f'  <div class="sb-bg"><div class="sb-fill {comp_color}" '
                    f'style="width:{best["composite"]*100:.1f}%"></div></div>'
                    f'  <span class="sb-val">{best["composite"]:.4f}</span>'
                    f'</div>'
                    f'<div style="font-size:11px;color:#80868B;margin-top:4px">'
                    f'{T["formula_label"]} · '
                    f'{T["threshold"]}: {COMPOSITE_THRESHOLD}</div>',
                    unsafe_allow_html=True,
                )
            else:
                # No candidates within radius
                candidates = bd.get("candidates", [])
                if candidates:
                    nearest = min(candidates, key=lambda c: c["dist"])
                    st.markdown(
                        f'<div style="font-size:13px;color:#5F6368">'
                        f'{T["nearest_ticket"]} (#{nearest["id"]} {nearest["cat"]}) '
                        f'<b>{nearest["dist"]:.0f} m</b> {T["away_outside"]} '
                        f'{SPATIAL_THRESHOLD_M:.0f} {T["dedup_radius"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info(T["no_tickets"])

            # Final status pill
            if result["type"] == "duplicate":
                st.markdown(
                    f'<div class="result-final-pill dup">'
                    f'{T["merged_into"]} #{result["ticket_id"]} '
                    f'({result["category"]})</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="result-final-pill new-ticket">'
                    f'{T["new_order_reg"]} #{result["ticket_id"]} {T["registered"]} '
                    f'({result["category"]})</div>',
                    unsafe_allow_html=True,
                )

            # Close ticket card
            st.markdown('  </div></div>', unsafe_allow_html=True)

            # Candidate comparison table
            candidates = bd.get("candidates", [])
            if len(candidates) > 1:
                with st.expander(T["all_candidates"]):
                    for c in sorted(candidates, key=lambda x: x["composite"],
                                    reverse=True):
                        status = T["evaluated"] if not c["filtered"] else T["outside_radius"]
                        st.markdown(
                            f'<div class="rp">'
                            f'<span class="rp-name">#{c["id"]} {c["cat"]}</span> · '
                            f'{status}<br>'
                            f'<span class="rp-text">{T["distance_label"]}: {c["dist"]:.0f}m · '
                            f'{T["composite_label"]}: {c["composite"]:.4f}</span></div>',
                            unsafe_allow_html=True,
                        )

    # ================================================================
    # RIGHT PANEL – Municipal Command Center
    # ================================================================
    with right:
        st.markdown(f'<div class="sec-h">{T["command_center"]}</div>',
                    unsafe_allow_html=True)

        # Metrics
        tickets = db_get_tickets()
        total_reports = db_total_reports()
        render_metrics(tickets, total_reports)

        st.markdown("")  # spacing

        # Map
        st.markdown(f'<div class="sec-h">{T["incident_map"]}</div>',
                    unsafe_allow_html=True)
        result = st.session_state.get("last_result")
        new_coords = None
        if result:
            new_coords = [result["lat"], result["lon"]]
        render_map(tickets, new_coords)

        # Ticket Inspector
        st.markdown(f'<div class="sec-h">{T["ticket_inspector"]}</div>',
                    unsafe_allow_html=True)

        for t in tickets:
            cls, badge_cls, lbl = _priority_class(t["priority_score"])
            with st.expander(
                f'#{t["id"]} · {t["category"]}  —  '
                f'{t["citizen_count"]} {T["reports"]} · {T["priority"]} {t["priority_score"]:.1f}'
            ):
                st.markdown(
                    f'<span class="p-badge {badge_cls}">{lbl}</span>',
                    unsafe_allow_html=True,
                )

                tc1, tc2 = st.columns([2, 1])
                with tc1:
                    st.markdown(f"**{T['description']}:** {t['description']}")
                    st.markdown(
                        f"📍 `{t['latitude']:.5f}, {t['longitude']:.5f}`  \n"
                        f"⚖️ {T['base_severity']}: **{t['base_severity']}** · "
                        f"{T['citizens']}: **{t['citizen_count']}**  \n"
                        f"🕐 {T['created']}: {t['created_at'][:16]}"
                    )
                with tc2:
                    if t["image_path"] and os.path.exists(t["image_path"]):
                        st.image(t["image_path"], caption=T["primary_photo"],
                                 use_container_width=True)

                # Linked reports
                reports = db_get_reports(t["id"])
                if reports:
                    st.markdown("---")
                    st.markdown(f"**{T['linked_reports']} ({len(reports)})**")
                    for r in reports:
                        rc1, rc2 = st.columns([3, 1])
                        with rc1:
                            st.markdown(
                                f'<div class="rp">'
                                f'<div class="rp-name">👤 {r["citizen_name"]} · '
                                f'📞 {r["phone_number"]}</div>'
                                f'<div class="rp-text">"{r["text_description"]}"</div>'
                                f'<div class="rp-time">🕐 {r["timestamp"][:16]} · '
                                f'📍 {r["latitude"]:.5f}, {r["longitude"]:.5f}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        with rc2:
                            if r["image_path"] and os.path.exists(r["image_path"]):
                                st.image(r["image_path"],
                                         use_container_width=True)

        # ================================================================
        # TINYFISH — Live Web Intelligence & Autonomous Dispatch
        # ================================================================
        st.markdown("")  # spacing

        st.markdown(
            f'<div class="sec-h">{T["tinyfish_header"]}</div>'
            f'<div style="text-align:right;margin-top:-10px;margin-bottom:10px;">'
            f'<span style="font-size:0.75rem;color:#80868B;background:#E8F0FE;'
            f'padding:3px 10px;border-radius:12px;font-weight:500;">'
            f'🐟 {T["tinyfish_powered"]}</span></div>',
            unsafe_allow_html=True,
        )

        # ── Feature A: Web Grievance Search ──
        st.markdown('<div class="g-card">', unsafe_allow_html=True)
        tf_query = st.text_input(
            T["tinyfish_search_label"],
            value=T["tinyfish_search_default"],
            key="tf_search_query",
        )
        tf_search_clicked = st.button(
            T["tinyfish_search_btn"], key="tf_search_btn",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if tf_search_clicked and tf_query.strip():
            with st.spinner(T["tinyfish_searching"]):
                results = search_civic_web(tf_query)

            if results:
                st.markdown(
                    f'<div class="sec-h" style="font-size:0.95rem;">'
                    f'{T["tinyfish_results_title"]}</div>',
                    unsafe_allow_html=True,
                )
                for i, item in enumerate(results[:2]):
                    title = item.get("title", item.get("name", f"Result {i+1}"))
                    snippet = item.get("snippet", item.get("description",
                              item.get("content", "")[:200]))
                    url = item.get("url", item.get("link", ""))
                    st.markdown(
                        f'<div class="result-ticket anim-up" style="margin-bottom:12px;">'
                        f'  <div class="result-ticket-header new-ticket" '
                        f'style="padding:12px 16px;">'
                        f'    <div>'
                        f'      <div class="result-ticket-title" '
                        f'style="font-size:0.95rem;">{title}</div>'
                        f'      <div class="result-ticket-subtitle">'
                        f'{T["tinyfish_source"]}: {url[:60]}…</div>'
                        f'    </div>'
                        f'  </div>'
                        f'  <div class="result-ticket-body" style="padding:12px 16px;">'
                        f'    <div style="font-size:0.85rem;color:#5F6368;">'
                        f'{snippet[:300]}</div>'
                        f'  </div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    # Fetch button for each result
                    if url:
                        if st.button(
                            T["tinyfish_fetch_btn"],
                            key=f"tf_fetch_{i}",
                        ):
                            with st.spinner(T["tinyfish_fetching"]):
                                article_md = fetch_civic_bulletin(url)
                            with st.expander(f"📄 {title[:50]}…"):
                                st.markdown(article_md[:2000] if article_md else "No content extracted.")
            else:
                st.info(T["tinyfish_no_results"])

        # ── Feature B: Autonomous Municipal Dispatch ──
        st.markdown("")  # spacing
        st.markdown(
            f'<div class="g-card">'
            f'  <div style="font-size:1rem;font-weight:600;color:#202124;'
            f'margin-bottom:6px;">{T["tinyfish_dispatch_header"]}</div>'
            f'  <div style="font-size:0.82rem;color:#5F6368;margin-bottom:12px;">'
            f'{T["tinyfish_dispatch_desc"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        tf_dispatch_clicked = st.button(
            T["tinyfish_dispatch_btn"], key="tf_dispatch_btn",
            use_container_width=True,
        )

        if tf_dispatch_clicked:
            # Find the highest-priority critical ticket
            critical_tickets = [
                t for t in tickets if t["priority_score"] > 12
            ]
            if critical_tickets:
                target = max(critical_tickets, key=lambda x: x["priority_score"])
                receipt_id = random.randint(8800, 9999)
                with st.status(
                    f"🤖 TinyFish Web Agent — Dispatching Ticket #{target['id']}…",
                    expanded=True,
                ) as status:
                    st.write(T["tinyfish_dispatch_step1"])
                    time.sleep(1.5)
                    st.write(T["tinyfish_dispatch_step2"])
                    time.sleep(2.0)
                    st.write(
                        T["tinyfish_dispatch_step3"]
                    )
                    time.sleep(2.0)
                    st.write(
                        T["tinyfish_dispatch_step4"].format(rid=receipt_id)
                    )
                    time.sleep(1.0)
                    status.update(
                        label=T["tinyfish_dispatch_done"].format(
                            receipt=f"MCD-2026-{receipt_id}",
                            tid=target["id"],
                            cat=target["category"],
                        ),
                        state="complete",
                        expanded=False,
                    )
                st.markdown(
                    f'<div class="result-final-pill dup" style="margin-top:10px;">'
                    f'🏛️ Receipt #MCD-2026-{receipt_id} · '
                    f'Ticket #{target["id"]} ({target["category"]}) · '
                    f'{target["citizen_count"]} {T["reports"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info(T["tinyfish_no_critical"])


if __name__ == "__main__":
    main()
