
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk

from youtube_comment_downloader import YoutubeCommentDownloader
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="YouTube Comment Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 YouTube Comment Sentiment Analysis")
st.write(
    "Analyze YouTube comments and visualize their sentiment "
    "as Positive, Negative, or Neutral."
)


# --------------------------------------------------
# INITIALIZE MODELS
# --------------------------------------------------

@st.cache_resource
def load_sentiment_analyzer():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


sentiments = load_sentiment_analyzer()
downloader = YoutubeCommentDownloader()


# --------------------------------------------------
# DOWNLOAD COMMENTS
# --------------------------------------------------

def comment_download(video_link):
    comments = downloader.get_comments_from_url(video_link)
    comments_list = list(comments)

    if not comments_list:
        return pd.DataFrame(columns=["Comment"])

    df = pd.DataFrame(comments_list)

    if "text" in df.columns:
        df.rename(columns={"text": "Comment"}, inplace=True)

    if "Comment" not in df.columns:
        raise ValueError(
            "The downloader did not return a comment text column."
        )

    df["Comment"] = df["Comment"].fillna("").astype(str)
    df = df[df["Comment"].str.strip() != ""].copy()

    return df


# --------------------------------------------------
# SENTIMENT ANALYSIS
# --------------------------------------------------

def sentiment_analysis(df1):
    df1 = df1.copy()

    scores = df1["Comment"].apply(sentiments.polarity_scores)

    df1["Positive"] = scores.apply(lambda x: x["pos"])
    df1["Negative"] = scores.apply(lambda x: x["neg"])
    df1["Neutral"] = scores.apply(lambda x: x["neu"])
    df1["Compound"] = scores.apply(lambda x: x["compound"])

    def classify_sentiment(score):
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        return "Neutral"

    df1["Sentiment"] = df1["Compound"].apply(classify_sentiment)

    return df1


# --------------------------------------------------
# DONUT CHART
# --------------------------------------------------

def sentiment_donut_chart(positive, negative, neutral):
    counts = {
        "Positive": positive,
        "Negative": negative,
        "Neutral": neutral,
    }

    chart_counts = {
        label: count
        for label, count in counts.items()
        if count > 0
    }

    total = sum(counts.values())

    if total == 0:
        st.info("No comments are available to display.")
        return

    color_map = {
        "Positive": "#00b300",
        "Negative": "#2e2eb8",
        "Neutral": "#ff6600",
    }

    labels = list(chart_counts.keys())
    values = list(chart_counts.values())
    colors = [color_map[label] for label in labels]

    fig, ax = plt.subplots(figsize=(9, 6))

    wedges, _, autotexts = ax.pie(
        values,
        colors=colors,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        counterclock=False,
        pctdistance=0.78,
        wedgeprops={
            "width": 0.38,
            "edgecolor": "white",
            "linewidth": 2,
        },
    )

    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")

    ax.text(
        0, 0.08, f"{total:,}",
        ha="center", va="center",
        fontsize=25, fontweight="bold",
    )

    ax.text(
        0, -0.15, "Comments",
        ha="center", va="center",
        fontsize=11,
    )

    legend_labels = [
        f"{label}: {count:,} ({count / total:.1%})"
        for label, count in chart_counts.items()
    ]

    ax.legend(
        wedges,
        legend_labels,
        title="Sentiment",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )

    ax.set_title("Sentiment Distribution of All Analyzed Comments")
    ax.axis("equal")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# --------------------------------------------------
# MAIN STREAMLIT APPLICATION
# --------------------------------------------------

def main():
    video_link = st.text_input(
        "YouTube Video URL",
        placeholder="Paste a YouTube video link here",
    )

    analyze_button = st.button(
        "Analyze Comments",
        type="primary",
        width='stretch',
    )

    if analyze_button:
        if not video_link.strip():
            st.warning("Please enter a YouTube video URL.")
            st.stop()

        try:
            # Download comments directly
            with st.spinner(
                "Downloading comments. This may take some time..."
            ):
                df = comment_download(video_link.strip())

            if df.empty:
                st.warning(
                    "No comments were retrieved. The video may have "
                    "disabled comments, or YouTube may be blocking "
                    "automated requests."
                )
                st.stop()

            # Remove unnecessary metadata columns
            columns_to_drop = [
                "cid", "time", "author", "channel", "votes",
                "replies", "photo", "heart", "reply", "time_parsed"
            ]

            df.drop(
                columns=columns_to_drop,
                errors="ignore",
                inplace=True,
            )

            st.success(f"Downloaded {len(df):,} comments.")

            # Analyze every retrieved comment
            with st.spinner("Analyzing comment sentiment..."):
                df2 = sentiment_analysis(df)

            st.success(
                f"Sentiment analysis completed for {len(df2):,} comments."
            )

            # Calculate sentiment counts
            positive = int((df2["Sentiment"] == "Positive").sum())
            negative = int((df2["Sentiment"] == "Negative").sum())
            neutral = int((df2["Sentiment"] == "Neutral").sum())
            total = len(df2)

            # Summary metrics
            st.subheader("📈 Sentiment Summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Comments", f"{total:,}")
            col2.metric(
                "Positive",
                f"{positive:,}",
                f"{positive / total:.1%}" if total else "0%",
            )
            col3.metric(
                "Negative",
                f"{negative:,}",
                f"{negative / total:.1%}" if total else "0%",
            )
            col4.metric(
                "Neutral",
                f"{neutral:,}",
                f"{neutral / total:.1%}" if total else "0%",
            )

            st.divider()

            # Donut chart
            sentiment_donut_chart(positive, negative, neutral)

            st.divider()

            # Analyzed comments
            st.subheader("💬 Analyzed Comments")

            selected_sentiment = st.selectbox(
                "Filter comments by sentiment",
                ["All", "Positive", "Negative", "Neutral"],
            )

            if selected_sentiment == "All":
                display_df = df2
            else:
                display_df = df2[
                    df2["Sentiment"] == selected_sentiment
                ]

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            # Download results
            st.download_button(
                "⬇️ Download All Analyzed Comments (CSV)",
                data=df2.to_csv(index=False).encode("utf-8-sig"),
                file_name="youtube_sentiment_analysis.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as exc:
            st.error(f"Unable to analyze this video: {exc}")
            st.caption(
                "Check the video URL and internet connection. "
                "YouTube may block automated comment downloading."
            )

    else:
        st.info("Enter a YouTube URL and click Analyze Comments.")


if __name__ == "__main__":
    main()
