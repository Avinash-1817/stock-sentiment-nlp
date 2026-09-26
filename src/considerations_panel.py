def render_considerations(d):
    """Lays out the available FACTUAL signals as neutral considerations -
    NOT a recommendation to buy or avoid. This is a deliberate design choice:
    the underlying research found sentiment correlates with SAME-DAY price
    movement only, with no predictive next-day edge, so a 'should you buy'
    verdict would misrepresent what the data actually supports."""
    if d.get("error"):
        return

    st.markdown("#### Things to consider (not a recommendation)")

    points_for = []
    points_against = []
    neutral_points = []

    tone = d["avg_sentiment"]
    if tone > 0.15:
        points_for.append(f"Recent news coverage has been mostly positive in tone.")
    elif tone < -0.15:
        points_against.append(f"Recent news coverage has been mostly negative in tone.")
    else:
        neutral_points.append("Recent news coverage has been mixed or neutral in tone - no strong signal either way.")

    if d["hist_r"] is not None:
        if d["hist_sig"]:
            if d["hist_r"] > 0:
                neutral_points.append(
                    "Historically, this stock's price has tended to move WITH same-day sentiment - "
                    "so today's tone (whatever it is) has historically lined up with same-day moves, "
                    "though this says nothing about tomorrow."
                )
            else:
                neutral_points.append(
                    "Historically, this stock's price has tended to move OPPOSITE to same-day "
                    "sentiment - an unusual pattern worth noting, not assuming."
                )
        else:
            neutral_points.append(
                "Historically, news sentiment hasn't shown a reliable relationship with this "
                "stock's price movement - the tone above may carry less weight for this company."
            )
    else:
        neutral_points.append("This company wasn't part of the historical study, so no track record is available.")

    window_chg = d.get("pct_change_window")
    days = d.get("days_back", 7)
    if window_chg is not None:
        if window_chg > 0.05:
            points_for.append(f"The stock has risen {window_chg:.1%} over the last {days} days.")
        elif window_chg < -0.05:
            points_against.append(f"The stock has fallen {abs(window_chg):.1%} over the last {days} days.")

    if points_for:
        st.markdown("**Points that may be seen as favorable:**")
        for p in points_for:
            st.write(f"- {p}")
    if points_against:
        st.markdown("**Points that may be seen as unfavorable:**")
        for p in points_against:
            st.write(f"- {p}")
    if neutral_points:
        st.markdown("**Context worth knowing:**")
        for p in neutral_points:
            st.write(f"- {p}")

    st.caption(
        "This is a summary of available signals only — not investment advice, and not a "
        "prediction of future performance. The underlying study found no reliable edge for "
        "predicting next-day price movement from sentiment alone."
    )