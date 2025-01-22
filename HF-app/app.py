import gradio as gr
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Optional, Dict, Any, List, Annotated
from instructor import patch
import instructor
from prompts import sentiments_prompt

# Load model and tokenizer globally for efficiency
model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define sentiment weights for score calculation
SENTIMENT_WEIGHTS = {
    0: 0.0,  # Very Negative
    1: 0.25,  # Negative
    2: 0.5,  # Neutral
    3: 0.75,  # Positive
    4: 1.0  # Very Positive
}

class ExtractProductSentiment(BaseModel):
    """Extracts what people like and dislike about a product based on product reviews and sentiment scores (0-100)"""
    product_likes: List[str] = Field(..., description="What people like about the product. List of 3 sentences AT MOST. Must be aggregated in the order of importance.")
    product_dislikes: List[str] = Field(..., description="What people dislike about the product. List of 3 sentences AT MOST. Must be aggregated in the order of importance.")

    @field_validator("product_likes", "product_dislikes")
    def validate_product_likes_and_dislikes(cls, v, info: ValidationInfo):
        if not v:
            raise ValueError(f"At least one {info.field_name} must be provided. If nothing to say, please enter 'None'")
        
        if len(v) > 3:
            raise ValueError(
                f"{info.field_name} contains {len(v)} points. Please aggregate the points to a maximum of 3 key points "
                "in order of importance. Combine similar points together."
            )
        return v

def predict_sentiment_with_scores(texts):
    """
    Predict sentiment for a list of texts and return both class labels and sentiment scores
    """
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get predicted classes
    sentiment_map = {
        0: "Very Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Very Positive"
    }
    predicted_classes = [sentiment_map[p] for p in torch.argmax(probabilities, dim=-1).tolist()]

    # Calculate sentiment scores (0-100)
    sentiment_scores = []
    for prob in probabilities:
        # Weighted sum of probabilities
        score = sum(prob[i].item() * SENTIMENT_WEIGHTS[i] for i in range(len(prob)))
        # Scale to 0-100
        sentiment_scores.append(round(score * 100, 2))

    return predicted_classes, sentiment_scores

#patch()  # Patch OpenAI client to support response models

def get_product_sentiment(client, reviews: List[str], scores: List[float]) -> ExtractProductSentiment:
    """Extract product likes and dislikes using OpenAI"""
    # Combine reviews and scores for context
    review_context = "\n".join([f"Review (Score: {score}): {review}" 
                               for review, score in zip(reviews, scores)])
    #client = instructor.patch(OpenAI(api_key=openai_api_key))
    prompt = sentiments_prompt.format(review_context=review_context)

    response = client.chat.completions.create(
        model="gpt-4o",
        response_model=ExtractProductSentiment,
        messages=[
            {"role": "system", "content": "You are a helpful product analyst."},
            {"role": "user", "content": prompt}
        ],
        max_retries=3
    )
    return response

def create_comparison_charts(sentiment_results, avg_sentiment_scores):
    """
    Create comparison charts for sentiment analysis across products
    """
    # Create summary DataFrame
    summary_data = []
    for product in sentiment_results.keys():
        counts = sentiment_results[product]
        total = counts.sum()
        row = {
            'Product': product,
            'Average Sentiment Score': avg_sentiment_scores[product],
            'Total Reviews': total,
            'Very Positive %': round((counts.get('Very Positive', 0) / total) * 100, 2),
            'Positive %': round((counts.get('Positive', 0) / total) * 100, 2),
            'Neutral %': round((counts.get('Neutral', 0) / total) * 100, 2),
            'Negative %': round((counts.get('Negative', 0) / total) * 100, 2),
            'Very Negative %': round((counts.get('Very Negative', 0) / total) * 100, 2)
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)

    # Score comparison chart
    score_comparison_fig = px.bar(
        summary_df,
        x='Product',
        y='Average Sentiment Score',
        title='Average Sentiment Scores by Product',
        labels={'Average Sentiment Score': 'Score (0-100)'}
    )

    # Distribution chart
    distribution_data = []
    for product in sentiment_results.keys():
        counts = sentiment_results[product]
        # Aggregate positive and negative sentiments
        aggregated_counts = {
            'Positive': counts.get('Very Positive', 0) + counts.get('Positive', 0),
            'Neutral': counts.get('Neutral', 0),
            'Negative': counts.get('Very Negative', 0) + counts.get('Negative', 0)
        }
        for sentiment, count in aggregated_counts.items():
            distribution_data.append({
                'Product': product,
                'Sentiment': sentiment,
                'Count': count
            })
    
    distribution_df = pd.DataFrame(distribution_data)
    distribution_fig = px.bar(
        distribution_df,
        x='Product',
        y='Count',
        color='Sentiment',
        title='Sentiment Distribution by Product',
        barmode='group',
        color_discrete_map={
            'Positive': '#2ECC71',  # Green
            'Neutral': '#F1C40F',   # Yellow
            'Negative': '#E74C3C'   # Red
        }
    )

    # Ratio chart (percentage stacked bar)
    ratio_fig = px.bar(
        distribution_df,
        x='Product',
        y='Count',
        color='Sentiment',
        title='Sentiment Distribution Ratio by Product',
        barmode='relative'
    )

    return score_comparison_fig, distribution_fig, ratio_fig, summary_df

def process_single_sheet(df, product_name, openai_client):
    """
    Process a single dataframe and return sentiment analysis results
    """
    if 'Reviews' not in df.columns:
        raise ValueError(f"'Reviews' column not found in sheet/file for {product_name}")

    reviews = df['Reviews'].fillna("")
    sentiments, scores = predict_sentiment_with_scores(reviews.tolist())

    df['Sentiment'] = sentiments
    df['Sentiment_Score'] = scores

    # Extract product likes and dislikes
    try:
        product_sentiment = get_product_sentiment(openai_client, reviews.tolist(), scores)
        
        # Initialize empty columns
        df['Likes'] = ""
        df['Dislikes'] = ""
        
        # Get the likes and dislikes lists
        likes_list = product_sentiment.product_likes
        dislikes_list = product_sentiment.product_dislikes
        
        # Only populate the first N rows where N is the length of the likes/dislikes lists
        for idx, (like, dislike) in enumerate(zip(likes_list, dislikes_list)):
            df.loc[idx, 'Likes'] = like
            df.loc[idx, 'Dislikes'] = dislike
            
    except Exception as e:
        print(f"Error extracting likes/dislikes for {product_name}: {str(e)}")
        df['Likes'] = ""
        df['Dislikes'] = ""

    # Calculate sentiment distribution
    sentiment_counts = pd.Series(sentiments).value_counts()
    avg_sentiment_score = round(sum(scores) / len(scores), 2)

    return df, sentiment_counts, avg_sentiment_score

def process_file(file_obj, api_key):
    """
    Process the input file and add sentiment analysis results
    """
    try:
        if not api_key:
            raise ValueError("OpenAI API key is required")
            
        client = instructor.patch(OpenAI(api_key=api_key))
        
        file_path = file_obj.name
        sentiment_results = defaultdict(pd.Series)
        avg_sentiment_scores = {}
        all_processed_dfs = {}

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            product_name = "Product"  # Default name for CSV
            processed_df, sentiment_counts, avg_score = process_single_sheet(df, product_name, client)
            all_processed_dfs[product_name] = processed_df
            sentiment_results[product_name] = sentiment_counts
            avg_sentiment_scores[product_name] = avg_score

        elif file_path.endswith(('.xlsx', '.xls')):
            excel_file = pd.ExcelFile(file_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                processed_df, sentiment_counts, avg_score = process_single_sheet(df, sheet_name, client)
                all_processed_dfs[sheet_name] = processed_df
                sentiment_results[sheet_name] = sentiment_counts
                avg_sentiment_scores[sheet_name] = avg_score
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

        # Create visualizations with new sentiment score chart
        score_comparison_fig, distribution_fig, ratio_fig, summary_df = create_comparison_charts(
            sentiment_results, avg_sentiment_scores
        )

        # Save results
        output_path = "sentiment_analysis_results.xlsx"
        with pd.ExcelWriter(output_path) as writer:
            for sheet_name, df in all_processed_dfs.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            if isinstance(summary_df, pd.DataFrame):  # Safety check
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

        return score_comparison_fig, distribution_fig, summary_df, output_path

    except Exception as e:
        raise gr.Error(str(e))


# Update the Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# Product Review Sentiment Analysis")

    gr.Markdown("""
    ### Quick Guide
    1. **Excel File (Multiple Products)**:
       - Create separate sheets for each product
       - Name sheets with product/company names
       - Include "Reviews" column in each sheet

    2. **CSV File (Single Product)**:
       - Include "Reviews" column

    Upload your file and click Analyze to get started.
    """)

    with gr.Row():
        api_key_input = gr.Textbox(
            label="OpenAI API Key",
            placeholder="Enter your OpenAI API key",
            type="password"
        )

    with gr.Row():
        file_input = gr.File(
            label="Upload File (CSV or Excel)",
            file_types=[".csv", ".xlsx", ".xls"]
        )

    with gr.Row():
        analyze_btn = gr.Button("Analyze Sentiments")

    with gr.Row():
        sentiment_score_plot = gr.Plot(label="Weighted Sentiment Scores")

    with gr.Row():
        distribution_plot = gr.Plot(label="Sentiment Distribution")

    with gr.Row():
        summary_table = gr.Dataframe(label="Summary Metrics")

    with gr.Row():
        output_file = gr.File(label="Download Full Report")

    analyze_btn.click(
        fn=process_file,
        inputs=[file_input, api_key_input],
        outputs=[sentiment_score_plot, distribution_plot, summary_table, output_file]
    )

# Launch interface
interface.launch()