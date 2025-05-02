import pandas as pd
import numpy as np
from datetime import timedelta

class InstagramAnalytics:
    def __init__(self, statistics, posts):
        import pandas as pd
        self.df = pd.DataFrame(statistics)
        self.posts_df = pd.DataFrame(posts) if posts else None
        # Ensure necessary columns exist for analytics
        if not self.df.empty:
            # Convert 'date' to datetime
            self.df['date'] = pd.to_datetime(self.df['date'])
            # Calculate daily_growth
            if 'daily_growth' not in self.df.columns:
                self.df['daily_growth'] = self.df['followers_count'].diff().fillna(0)
            # Calculate growth_percent
            if 'growth_percent' not in self.df.columns:
                self.df['growth_percent'] = self.df['daily_growth'] / self.df['followers_count'].shift(1).replace(0, 1) * 100
                self.df['growth_percent'] = self.df['growth_percent'].fillna(0)
            # Add day_of_week
            if 'day_of_week' not in self.df.columns:
                self.df['day_of_week'] = self.df['date'].dt.day_name()
            # Add rolling_avg_7d
            if 'rolling_avg_7d' not in self.df.columns:
                self.df['rolling_avg_7d'] = self.df['followers_count'].rolling(window=7, min_periods=1).mean()

    # Account category definitions
    ACCOUNT_CATEGORIES = {
        'business': {
            'min_er': 2.0,  # Minimum expected engagement rate
            'optimal_post_frequency': {'min': 3, 'max': 5},  # posts per week
            'key_metrics': ['conversions', 'reach', 'engagement']
        },
        'influencer': {
            'min_er': 3.5,
            'optimal_post_frequency': {'min': 5, 'max': 7},
            'key_metrics': ['engagement', 'reach', 'authenticity']
        },
        'media': {
            'min_er': 1.5,
            'optimal_post_frequency': {'min': 7, 'max': 14},
            'key_metrics': ['reach', 'engagement', 'content_consistency']
        }
    }

    def calculate_engagement_rate(self, post):
        """Calculate engagement rate for a post"""
        # Ensure both sides are tz-naive and compatible
        post_date = pd.to_datetime(post['date'])
        if hasattr(post_date, 'tzinfo') and post_date.tzinfo is not None:
            post_date = post_date.replace(tzinfo=None)
        # Ensure self.df['date'] is tz-naive
        df_dates = self.df['date']
        if hasattr(df_dates.iloc[0], 'tzinfo') and df_dates.iloc[0].tzinfo is not None:
            df_dates = df_dates.apply(lambda d: d.replace(tzinfo=None))
        mask = df_dates >= post_date
        if not mask.any():
            followers = 0
        else:
            followers = self.df.loc[mask].iloc[0]['followers_count']
        if followers == 0:
            return 0
        engagement = post.get('likes', 0) + post.get('comments', 0)
        return (engagement / followers) * 100

    def calculate_content_score(self, post):
        """Calculate content quality score based on multiple factors"""
        er = self.calculate_engagement_rate(post)
        caption_length = len(post.get('caption', ''))
        has_hashtags = '#' in post.get('caption', '')
        has_mentions = '@' in post.get('caption', '')
        has_media = post.get('type') in ['photo', 'video', 'carousel']
        
        score = 0
        if er >= 3.0: score += 40  # Good engagement
        if 50 <= caption_length <= 300: score += 20  # Optimal caption length
        if has_hashtags: score += 15  # Uses hashtags
        if has_mentions: score += 10  # Community engagement
        if has_media: score += 15  # Visual content
        
        return score

    def link_growth_to_posts_comments(self, comments=None):
        """
        Links significant daily follower changes to posts and comments published on or just before those dates.
        Returns a list of dicts: {date, followers_change, post_ids, posts, comments, summary}
        """
        # Identify significant follower changes (spikes/drops)
        threshold = max(10, self.df['daily_growth'].std() * 1.5)
        events = self.df[abs(self.df['daily_growth']) >= threshold]
        results = []
        posts_df = self.posts_df.copy() if self.posts_df is not None else None
        comments_df = pd.DataFrame(comments) if comments is not None else None
        for _, row in events.iterrows():
            date = row['date']
            followers_change = int(row['daily_growth'])
            # Find posts published on this date or the day before
            posts = []
            post_ids = []
            if posts_df is not None:
                mask = (pd.to_datetime(posts_df['date']).dt.date >= (date - pd.Timedelta(days=1)).date()) & (pd.to_datetime(posts_df['date']).dt.date <= date.date())
                posts = posts_df[mask].to_dict(orient='records')
                post_ids = [p['id'] for p in posts]
            # Find comments for those posts, on this date
            linked_comments = []
            if comments_df is not None and post_ids:
                mask = (comments_df['post_id'].isin(post_ids)) & (pd.to_datetime(comments_df['date']).dt.date == date.date())
                linked_comments = comments_df[mask].to_dict(orient='records')
            # Summarize
            summary = f"{'Spike' if followers_change>0 else 'Drop'} of {abs(followers_change)} followers on {date.strftime('%Y-%m-%d')}."
            if posts:
                summary += f" Related post(s): " + ", ".join([f"[{p['type']}] {p['caption'][:30]}..." for p in posts])
            if linked_comments:
                summary += f" | Comments: " + "; ".join([c['text'][:40] for c in linked_comments])
            results.append({
                'date': date.strftime('%Y-%m-%d'),
                'followers_change': followers_change,
                'posts': posts,
                'comments': linked_comments,
                'summary': summary
            })
        return results

    def generate_insights(self, comments=None, account_type='business'):
        """
        Generate up to 4 prioritized, actionable, category-aware insights based on analytics.
        Each insight has a headline, a tailored action, and a priority badge.
        """
        category_config = self.ACCOUNT_CATEGORIES.get(account_type, self.ACCOUNT_CATEGORIES['business'])
        insights = []

        # 1. Best Posting Day Insight
        weekday_info = self.analyze_weekday_performance()
        best_day = weekday_info.get('best_day', None)
        best_day_growth = weekday_info.get('best_day_growth', None)
        if best_day and best_day != 'N/A' and best_day_growth is not None:
            insights.append({
                'priority': 'OPPORTUNITY',
                'insight': f"Лучший день для публикаций: {best_day} (+{best_day_growth} подписчиков в среднем)",
                'action': f"Планируйте ключевые публикации на {best_day}, чтобы максимизировать прирост аудитории.",
                'impact_score': 2.5
            })

        # 2. Growth Trend Insight (with improved action)
        basic = self.get_basic_stats()
        if basic['total_growth'] > 0:
            best_period = self.find_best_growth_periods()
            best_period_str = ""
            if best_period and 'error' not in best_period:
                best_period_str = f" ({best_period['best_growth_period_start']} — {best_period['best_growth_period_end']})"
            insights.append({
                'priority': 'CRITICAL',
                'insight': f"Рост: +{basic['total_growth']} подписчиков ({basic['growth_percent']}%) за {basic['period_days']} дней.",
                'action': f"Повторите формат публикаций или активностей, которые были в лучший период роста{best_period_str}. Проверьте, какие посты или кампании совпали с этим всплеском.",
                'impact_score': 3
            })
        elif basic['total_growth'] < 0:
            insights.append({
                'priority': 'CRITICAL',
                'insight': f"Потеря подписчиков: {basic['total_growth']} за {basic['period_days']} дней.",
                'action': "Срочно проанализируйте контент, вовлеченность и негативные комментарии, чтобы остановить отток.",
                'impact_score': 3
            })

        # 3. Engagement Rate vs. Category Minimum (with more concrete action)
        eng = self.engagement_by_post_type()
        avg_er = 0
        best_type = None
        if eng:
            avg_er = sum([e['engagement_rate'] for e in eng]) / len(eng)
            best_type = max(eng, key=lambda x: x['engagement_rate'])['type'] if eng else None
        min_er = category_config['min_er']
        if avg_er < min_er:
            action = "Используйте более вовлекающий контент: вопросы, опросы, сторис, коллаборации."
            if best_type:
                action += f" Лучше всего работают публикации типа: {best_type}."
            insights.append({
                'priority': 'IMPORTANT',
                'insight': f"Вовлеченность ниже нормы: {avg_er:.2f}% (минимум для вашей категории: {min_er}%)",
                'action': action,
                'impact_score': 2
            })
        else:
            action = "Продолжайте публиковать похожий контент для удержания внимания аудитории."
            if best_type:
                action += f" Формат {best_type} показывает лучшие результаты."
            insights.append({
                'priority': 'OPPORTUNITY',
                'insight': f"Вовлеченность выше нормы: {avg_er:.2f}% (минимум: {min_er}%)",
                'action': action,
                'impact_score': 1
            })

        # 4. Anomaly or Best Growth Period (with explicit attribution)
        anomalies = self.detect_anomalies()
        best = None
        if anomalies and anomalies.get('anomaly_count', 0) > 0:
            anomaly = anomalies['anomalies'][0]
            linked = self.link_growth_to_posts_comments(comments)
            summary = "Изучите публикации и события за этот день — возможны внешние причины или вирусный контент."
            if linked:
                for event in linked:
                    if (event['date'] == anomaly['date'] and event['followers_change'] == anomaly['followers_change']):
                        post_summaries = []
                        for p in event['posts']:
                            post_summaries.append(f"[{p.get('type', 'post')}] {p.get('caption', '')[:40]}...")
                        negative_comment = None
                        for c in event['comments']:
                            if 'negative' in c.get('text', '').lower():
                                negative_comment = c['text']
                                break
                        if post_summaries:
                            summary = f"Рост связан с публикацией: {'; '.join(post_summaries)}"
                        if negative_comment:
                            summary += f" | Обнаружены негативные комментарии: '{negative_comment[:40]}...'"
                        elif event['comments']:
                            summary += f" | В этот день были активные обсуждения в комментариях."
                        break
            insights.append({
                'priority': 'CRITICAL',
                'insight': f"Аномалия: {anomaly['type']} {anomaly['followers_change']} подписчиков {anomaly['date']}",
                'action': summary,
                'impact_score': 3
            })
        else:
            best = self.find_best_growth_periods()
        if best is not None and isinstance(best, dict) and 'error' not in best:
            insights.append({
                'priority': 'OPPORTUNITY',
                'insight': f"Лучший период роста: +{best['best_growth_followers']} подписчиков с {best['best_growth_period_start']} по {best['best_growth_period_end']}",
                'action': "Проанализируйте контент и активность в этот период и повторите успешные практики.",
                'impact_score': 1
            })

        # Only show up to 4 insights, sorted by impact_score (highest first)
        insights = sorted(insights, key=lambda x: -x['impact_score'])[:4]
        return insights

    def forecast_trend(self):
        """Compatibility alias for predict_growth_trend()."""
        return self.predict_growth_trend()

    def prepare_chart_data(self):
        """Compatibility alias for get_chart_data()."""
        return self.get_chart_data()

    def post_type_distribution(self):
        if self.posts_df is None or self.posts_df.empty:
            return {}
        return self.posts_df['type'].value_counts().to_dict()

    def top_keywords(self, top_n=10):
        if self.posts_df is None or self.posts_df.empty:
            return []
        hashtags = self.posts_df['hashtags'].explode().dropna().tolist()
        if not hashtags:
            return []
        from collections import Counter
        counter = Counter(hashtags)
        return counter.most_common(top_n)

    def engagement_by_post_type(self):
        if self.posts_df is None or self.posts_df.empty:
            return {}
        grouped = self.posts_df.groupby('type').agg({
            'likes': 'mean',
            'comments': 'mean',
            'engagement_rate': 'mean',
            'views': 'mean'
        }).reset_index()
        return grouped.to_dict(orient='records')

    def analyze_comment_sentiment(self, comments):
        """Returns sentiment distribution and average score for comments using VADER sentiment analysis."""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        import numpy as np
        if not comments:
            return {"distribution": {}, "average": 0}
        analyzer = SentimentIntensityAnalyzer()
        distr = {"POSITIVE": 0, "NEGATIVE": 0}
        scores = []
        for c in comments:
            score = analyzer.polarity_scores(c['text'])['compound']
            if score >= 0.05:
                distr["POSITIVE"] += 1
                scores.append(score)
            elif score <= -0.05:
                distr["NEGATIVE"] += 1
                scores.append(score)
        total = sum(distr.values())
        if total:
            for k in distr:
                distr[k] = distr[k] / total
        avg_score = float(np.mean(scores)) if scores else 0
        return {"distribution": distr, "average": avg_score}

    def analyze_post_sentiment(self):
        """Returns sentiment distribution and average score for post captions using VADER sentiment analysis."""
        if self.posts_df is None or self.posts_df.empty:
            return {"distribution": {}, "average": 0}
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        import numpy as np
        analyzer = SentimentIntensityAnalyzer()
        captions = self.posts_df['caption'].dropna().tolist()
        if not captions:
            return {"distribution": {}, "average": 0}
        distr = {"POSITIVE": 0, "NEGATIVE": 0}
        scores = []
        for text in captions:
            score = analyzer.polarity_scores(text)['compound']
            if score >= 0.05:
                distr["POSITIVE"] += 1
                scores.append(score)
            elif score <= -0.05:
                distr["NEGATIVE"] += 1
                scores.append(score)
        total = sum(distr.values())
        if total:
            for k in distr:
                distr[k] = distr[k] / total
        avg_score = float(np.mean(scores)) if scores else 0
        return {"distribution": distr, "average": avg_score}

    def get_basic_stats(self):
        if len(self.df) < 2:
            return {"error": "Not enough data for analysis"}
        start_followers = self.df.iloc[0]['followers_count']
        end_followers = self.df.iloc[-1]['followers_count']
        total_growth = end_followers - start_followers
        growth_percent = (total_growth / start_followers) * 100 if start_followers > 0 else 0
        period_days = (self.df.iloc[-1]['date'] - self.df.iloc[0]['date']).days
        avg_daily_growth = total_growth / period_days if period_days > 0 else 0
        return {
            "start_date": self.df.iloc[0]['date'].strftime('%Y-%m-%d'),
            "end_date": self.df.iloc[-1]['date'].strftime('%Y-%m-%d'),
            "period_days": period_days,
            "start_followers": int(start_followers),
            "end_followers": int(end_followers),
            "total_growth": int(total_growth),
            "growth_percent": round(growth_percent, 2),
            "avg_daily_growth": round(avg_daily_growth, 2)
        }

    def find_best_growth_periods(self):
        if len(self.df) < 7:
            return {"error": "Not enough data for period analysis"}
        self.df['rolling_growth_7d'] = self.df['daily_growth'].rolling(window=7, min_periods=1).sum()
        best_growth_idx = self.df['rolling_growth_7d'].idxmax()
        if pd.isna(best_growth_idx):
            return {"error": "Could not determine best growth period"}
        best_start_date = self.df.loc[best_growth_idx - 6 if best_growth_idx >= 6 else 0]['date']
        best_end_date = self.df.loc[best_growth_idx]['date']
        best_growth = self.df.loc[best_growth_idx]['rolling_growth_7d']
        return {
            "best_growth_period_start": best_start_date.strftime('%Y-%m-%d'),
            "best_growth_period_end": best_end_date.strftime('%Y-%m-%d'),
            "best_growth_followers": int(best_growth)
        }

    def analyze_weekday_performance(self):
        weekday_growth = self.df.groupby('day_of_week')['daily_growth'].mean().to_dict()
        best_day = max(weekday_growth.items(), key=lambda x: x[1]) if weekday_growth else ("N/A", 0)
        worst_day = min(weekday_growth.items(), key=lambda x: x[1]) if weekday_growth else ("N/A", 0)
        return {
            "weekday_growth": {day: round(growth, 2) for day, growth in weekday_growth.items()},
            "best_day": best_day[0],
            "best_day_growth": round(best_day[1], 2),
            "worst_day": worst_day[0],
            "worst_day_growth": round(worst_day[1], 2)
        }

    def detect_anomalies(self):
        if len(self.df) < 14:
            return {"error": "Not enough data for anomaly detection"}
        std_growth = self.df['daily_growth'].std()
        mean_growth = self.df['daily_growth'].mean()
        threshold = 2.0
        anomalies = self.df[abs(self.df['daily_growth'] - mean_growth) > threshold * std_growth]
        anomaly_data = []
        for _, row in anomalies.iterrows():
            anomaly_data.append({
                "date": row['date'].strftime('%Y-%m-%d'),
                "followers_change": int(row['daily_growth']),
                "percent_change": round(row['growth_percent'], 2),
                "type": "increase" if row['daily_growth'] > 0 else "decrease"
            })
        return {
            "anomaly_count": len(anomaly_data),
            "anomalies": anomaly_data
        }

    def predict_growth_trend(self):
        if len(self.df) < 14:
            return {"error": "Not enough data for forecasting"}
        recent_data = self.df.tail(14)
        x = np.arange(len(recent_data))
        y = recent_data['followers_count'].values
        z = np.polyfit(x, y, 1)
        slope = z[0]
        last_date = self.df.iloc[-1]['date']
        last_followers = self.df.iloc[-1]['followers_count']
        forecast = []
        for i in range(1, 8):
            next_date = last_date + timedelta(days=i)
            predicted_followers = last_followers + (slope * i)
            forecast.append({
                "date": next_date.strftime('%Y-%m-%d'),
                "predicted_followers": int(predicted_followers)
            })
        if slope > 0:
            trend = "positive"
            trend_description = "восходящий"
        elif slope < 0:
            trend = "negative"
            trend_description = "нисходящий"
        else:
            trend = "neutral"
            trend_description = "стабильный"
        return {
            "trend": trend,
            "trend_description": trend_description,
            "weekly_slope": round(slope * 7, 2),
            "forecast": forecast
        }

    def get_chart_data(self):
        return [
            {
                "date": row['date'].strftime('%Y-%m-%d'),
                "followers": int(row['followers_count']),
                "daily_growth": float(row['daily_growth']) if not pd.isna(row['daily_growth']) else 0,
                "rolling_avg": float(row['rolling_avg_7d'])
            }
            for _, row in self.df.iterrows()
        ]
