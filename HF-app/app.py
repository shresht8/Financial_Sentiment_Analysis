import gradio as gr
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

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


def process_single_sheet(df, product_name):
    """
    Process a single dataframe and return sentiment analysis results
    """
    if 'Reviews' not in df.columns:
        raise ValueError(f"'Reviews' column not found in sheet/file for {product_name}")

    reviews = df['Reviews'].fillna("")
    sentiments, scores = predict_sentiment_with_scores(reviews.tolist())

    df['Sentiment'] = sentiments
    df['Sentiment_Score'] = scores

    # Calculate sentiment distribution
    sentiment_counts = pd.Series(sentiments).value_counts()
    avg_sentiment_score = round(sum(scores) / len(scores), 2)

    return df, sentiment_counts, avg_sentiment_score


def create_comparison_charts(sentiment_results, avg_scores):
    """
    Create investment-focused comparison charts including the new sentiment score visualization
    """
    # Prepare data for plotting
    plot_data = []
    for product, sentiment_counts in sentiment_results.items():
        sentiment_dict = sentiment_counts.to_dict()
        total = sum(sentiment_dict.values())

        row = {
            'Product': product,
            'Total Reviews': total
        }
        # Calculate percentages for each sentiment
        for sentiment, count in sentiment_dict.items():
            row[sentiment] = (count / total) * 100
        plot_data.append(row)

    df = pd.DataFrame(plot_data)

    # Ensure all sentiment columns exist in the correct order
    sentiments = ['Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative']
    for sentiment in sentiments:
        if sentiment not in df.columns:
            df[sentiment] = 0

    # Calculate weighted sentiment score (0 to 100)
    sentiment_weights = {
        'Very Negative': 0,
        'Negative': 25,
        'Neutral': 50,
        'Positive': 75,
        'Very Positive': 100
    }

    # Create stacked bar chart for sentiment distribution
    distribution_fig = go.Figure()
    sentiments = ['Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative']
    colors = ['rgb(39, 174, 96)', 'rgb(46, 204, 113)',
              'rgb(241, 196, 15)', 'rgb(231, 76, 60)',
              'rgb(192, 57, 43)']

    for sentiment, color in zip(sentiments, colors):
        distribution_fig.add_trace(go.Bar(
            name=sentiment,
            x=df['Product'],
            y=df[sentiment],
            marker_color=color
        ))

    distribution_fig.update_layout(
        barmode='stack',
        title='Sentiment Distribution by Product',
        yaxis_title='Percentage (%)',
        showlegend=True
    )

    # Calculate Positive-Negative Ratios
    df['Positive Ratio'] = df[['Positive', 'Very Positive']].sum(axis=1)
    df['Negative Ratio'] = df[['Negative', 'Very Negative']].sum(axis=1)

    # Create Positive-Negative ratio chart
    ratio_fig = go.Figure()
    ratio_fig.add_trace(go.Bar(
        name='Positive',
        x=df['Product'],
        y=df['Positive Ratio'],
        marker_color='rgb(50, 205, 50)'
    ))
    ratio_fig.add_trace(go.Bar(
        name='Negative',
        x=df['Product'],
        y=df['Negative Ratio'],
        marker_color='rgb(220, 20, 60)'
    ))
    ratio_fig.update_layout(
        barmode='group',
        title='Positive vs Negative Sentiment Ratio by Product',
        yaxis_title='Percentage (%)'
    )

    # Create summary DataFrame
    summary_data = {
        'Product': df['Product'].tolist(),
        'Total Reviews': df['Total Reviews'].tolist(),
        'Positive Ratio (%)': df['Positive Ratio'].round(2).tolist(),
        'Negative Ratio (%)': df['Negative Ratio'].round(2).tolist(),
        'Neutral Ratio (%)': df['Neutral'].round(2).tolist(),
        'Weighted Sentiment Score': [avg_scores[prod] for prod in df['Product']]
    }
    summary_df = pd.DataFrame(summary_data)

    # Create sentiment score chart
    score_comparison_fig = go.Figure()
    score_comparison_fig.add_trace(go.Bar(
        x=summary_df['Product'],
        y=summary_df['Weighted Sentiment Score'],
        text=[f"{score:.1f}" for score in summary_df['Weighted Sentiment Score']],
        textposition='auto',
        marker_color='rgb(65, 105, 225)',
        name='Sentiment Score'
    ))
    score_comparison_fig.update_layout(
        title='Weighted Sentiment Scores by Product (0-100)',
        yaxis_title='Sentiment Score',
        yaxis_range=[0, 100],
        showlegend=False,
        bargap=0.3,
        plot_bgcolor='white'
    )

    return score_comparison_fig, distribution_fig, ratio_fig, summary_df

    products = list(avg_scores.keys())
    scores = list(avg_scores.values())

    # Add bars for sentiment scores
    score_comparison_fig.add_trace(go.Bar(
        x=products,
        y=scores,
        text=[f"{score:.1f}" for score in scores],
        textposition='auto',
        marker_color='rgb(65, 105, 225)',
        name='Sentiment Score'
    ))

    # Update layout with appropriate styling
    score_comparison_fig.update_layout(
        title='Weighted Sentiment Scores by Product (0-100)',
        yaxis_title='Sentiment Score',
        yaxis_range=[0, 100],
        showlegend=False,
        bargap=0.3,
        plot_bgcolor='white'
    )

    # Add score to summary DataFrame
    summary_df['Weighted Sentiment Score'] = [avg_scores[prod] for prod in summary_df['Product']]

    # Create sentiment distribution stacked bar chart
    distribution_fig = go.Figure()
    colors = ['rgb(39, 174, 96)', 'rgb(46, 204, 113)',
              'rgb(241, 196, 15)', 'rgb(231, 76, 60)',
              'rgb(192, 57, 43)']

    # Add traces for each sentiment in order
    for sentiment, color in zip(sentiments, colors):
        distribution_fig.add_trace(go.Bar(
            name=sentiment,
            x=df['Product'],
            y=df[sentiment],
            marker_color=color
        ))

    distribution_fig.update_layout(
        barmode='stack',
        title='Sentiment Distribution by Product',
        yaxis_title='Percentage (%)',
        showlegend=True
    )

    return score_comparison_fig, distribution_fig, summary_df, output_path


def process_file(file_obj):
    """
    Process the input file and add sentiment analysis results
    """
    try:
        file_path = file_obj.name
        sentiment_results = defaultdict(pd.Series)
        avg_sentiment_scores = {}
        all_processed_dfs = {}

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            product_name = "Product"  # Default name for CSV
            processed_df, sentiment_counts, avg_score = process_single_sheet(df, product_name)
            all_processed_dfs[product_name] = processed_df
            sentiment_results[product_name] = sentiment_counts
            avg_sentiment_scores[product_name] = avg_score

        elif file_path.endswith(('.xlsx', '.xls')):
            excel_file = pd.ExcelFile(file_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                processed_df, sentiment_counts, avg_score = process_single_sheet(df, sheet_name)
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

        # Save results
        output_path = "sentiment_analysis_results.xlsx"
        with pd.ExcelWriter(output_path) as writer:
            # Save individual sheet data
            for sheet_name, df in all_processed_dfs.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Save summary data
            if isinstance(summary_df, pd.DataFrame):  # Ensure it's a DataFrame before saving
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
        inputs=[file_input],
        outputs=[sentiment_score_plot, distribution_plot, summary_table, output_file]
    )

# Launch interface
interface.launch()