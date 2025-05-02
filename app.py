from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import os
from analytics import InstagramAnalytics

app = FastAPI()

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

def load_json(filename):
    with open(os.path.join("mockdata", filename), "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Load statistics and analytics
    statistics = load_json("statistics.json")["statistics"]
    posts = load_json("posts.json")["posts"]
    analytics = InstagramAnalytics(statistics, posts)

    # Load account info
    account = load_json("account.json")["accounts"][0]

    # Load comments
    comments = load_json("comments.json")["comments"]

    basic_stats = analytics.get_basic_stats()
    best_growth_period = analytics.find_best_growth_periods()
    weekday_performance = analytics.analyze_weekday_performance()
    anomalies = analytics.detect_anomalies()
    trend = analytics.forecast_trend()
    chart_data = analytics.prepare_chart_data()

    # New: Post/keyword/engagement analytics
    post_type_dist = analytics.post_type_distribution()
    top_keywords = analytics.top_keywords()
    engagement_by_type = analytics.engagement_by_post_type()
    insights = analytics.generate_insights(comments)
    comment_sentiment = analytics.analyze_comment_sentiment(comments)
    post_sentiment = analytics.analyze_post_sentiment()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "account": account,
        "statistics": statistics,
        "posts": posts,
        "comments": comments,
        "basic_stats": basic_stats,
        "best_growth_period": best_growth_period,
        "weekday_performance": weekday_performance,
        "anomalies": anomalies,
        "trend": trend,
        "chart_data": chart_data,
        "post_type_dist": post_type_dist,
        "top_keywords": top_keywords,
        "engagement_by_type": engagement_by_type,
        "insights": insights,
        "comment_sentiment": comment_sentiment,
        "post_sentiment": post_sentiment
    })
