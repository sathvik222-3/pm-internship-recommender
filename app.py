import streamlit as st
import pandas as pd
from utils.recommender import build_model, recommend

st.set_page_config(page_title="PM Internship Recommender", page_icon="🎯", layout="wide")

st.title("🎯 PM Internship Recommendation Engine")
st.caption("Personalized matches based on your skills, education, interests, and location")

# --- Load data & model ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/internships.csv")
    return df.fillna("")

@st.cache_resource
def load_model(df):
    return build_model(df)

internships = load_data()
vectorizer, corpus_matrix = load_model(internships)

# --- Profile Input ---
st.header("🧑‍🎓 Student Profile")
skills = st.text_input("Skills (comma-separated)", placeholder="e.g., Python, Excel, Content Writing")
interests = st.text_input("Interests / Domains", placeholder="e.g., AI, Finance, Marketing")
education = st.selectbox("Education Level", ["10+2", "Diploma", "UG", "PG", "Other"])
location_pref = st.text_input("Preferred Location", placeholder="e.g., Delhi, Telangana, Remote")

if st.button("🔎 Find Recommendations", use_container_width=True):
    with st.spinner("Finding best matches..."):
        results = recommend(
            internships,
            vectorizer,
            corpus_matrix,
            skills=skills,
            interests=interests,
            education=education,
            location_pref=location_pref,
            top_k=5
        )

    if results.empty:
        st.warning("⚠️ No suitable internships found. Try different skills or location.")
    else:
        st.subheader("✅ Top Recommendations")
        for _, row in results.iterrows():
            with st.container():
                st.markdown(f"### {row['title']} · {row['org']}")
                st.write(f"📍 **Location:** {row['location']}")
                st.write(f"🛠️ **Skills Required:** {row['skills']}")
                st.write(f"📚 **Domain:** {row['domain']}")
                st.progress(min(100, int(row['score'])))
                st.link_button("Apply Now", row['apply_url'])

