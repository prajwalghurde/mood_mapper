import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.title("Mood Mapper")
    st.subheader("Emotion Detection in Social media Text")
    page_bg_img ="""

<style>
    .stApp{
        background-image:url(https://png.pngtree.com/thumb_back/fh260/background/20220206/pngtree-pile-of-social-media-emoji-network-emoticon-background-image_985727.jpg);
        background-size:cover;
        background-repeat: no-repeat;
        opacity=0.1;
    }
</style>

    """
    st.markdown(page_bg_img,unsafe_allow_html=True)

    with st.form(key='my_form'):
        raw_text = st.text_area("Enter the text here", height=150)
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        st.write("---")

        st.subheader("Original Text")
        st.write(raw_text)

        st.subheader("Prediction")
        emoji_icon = emotions_emoji_dict[prediction]
        st.write(f"**{prediction.capitalize()}** {emoji_icon}")
        st.write(f"Confidence: {np.max(probability):.2%}")

        st.write("---")

        st.subheader("Prediction Probability")
        proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
        proba_df_clean = proba_df.T.reset_index()
        proba_df_clean.columns = ["emotions", "probability"]

        fig = alt.Chart(proba_df_clean).mark_bar().encode(
            x='emotions',
            y='probability',
            color='emotions',
            tooltip=['emotions', 'probability']
        ).interactive()

        st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
