"""
Evaluator Module - Analytics and Visualization Class
=====================================================

This module handles all model evaluation tasks:
- Confusion matrix generation
- Performance metrics calculation
- Results visualization
- Model comparison charts

The Evaluator class keeps evaluation logic separate from
training logic (Single Responsibility Principle).

Author: Muhammad Ahmad
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    auc
)
from typing import Dict, List, Tuple, Any
import io


class Evaluator:
    """
    Evaluation and Analytics class for ML models.
    
    This class encapsulates all evaluation logic, separating it from
    the model training and data preprocessing concerns. This follows
    the Single Responsibility Principle - each class has one job.
    
    Features:
        - Confusion matrix generation with visualization
        - Comprehensive metrics calculation
        - Interactive Plotly charts
        - Model comparison functionality
    """
    
    def __init__(self):
        """Initialize the Evaluator with empty metrics history."""
        self.__metrics_history = []  # Private: stores all evaluation results
        self.__labels = ['negative', 'positive']  # Default sentiment labels
    
    def set_labels(self, labels: list) -> None:
        """
        Set the class labels for the evaluation.
        
        Args:
            labels: List of class label names
        """
        self.__labels = labels
    
    def generate_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        normalize: bool = False
    ) -> np.ndarray:
        """
        Generate a confusion matrix from predictions.
        
        A confusion matrix shows the counts of:
        - True Positives (correctly predicted positive)
        - True Negatives (correctly predicted negative)
        - False Positives (incorrectly predicted positive)
        - False Negatives (incorrectly predicted negative)
        
        Args:
            y_true: Actual labels
            y_pred: Predicted labels
            normalize: Whether to normalize values (show percentages)
            
        Returns:
            2D numpy array containing the confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        return cm
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix"
    ) -> go.Figure:
        """
        Create an interactive Plotly confusion matrix heatmap.
        
        Args:
            y_true: Actual labels
            y_pred: Predicted labels
            title: Title for the plot
            
        Returns:
            Plotly Figure object
        """
        cm = self.generate_confusion_matrix(y_true, y_pred)
        
        # Create heatmap using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=self.__labels,
            y=self.__labels,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20},
            hoverongaps=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color='white'),
                x=0.5
            ),
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            width=500,
            height=450
        )
        
        return fig
    
    def plot_confusion_matrix_matplotlib(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix"
    ) -> plt.Figure:
        """
        Create a Matplotlib confusion matrix (alternative to Plotly).
        
        Args:
            y_true: Actual labels
            y_pred: Predicted labels
            title: Title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        cm = self.generate_confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.__labels,
            yticklabels=self.__labels,
            ax=ax,
            annot_kws={"size": 16}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('Actual Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Metrics calculated:
        - Accuracy: Overall correct predictions / total predictions
        - Precision: True positives / (True positives + False positives)
        - Recall: True positives / (True positives + False negatives)
        - F1 Score: Harmonic mean of precision and recall
        
        Args:
            y_true: Actual labels
            y_pred: Predicted labels
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary containing all metrics
        """
        # Determine if binary or multi-class
        average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
        
        # Handle string labels for binary classification
        pos_label = None
        if len(np.unique(y_true)) == 2:
            unique_labels = np.unique(y_true)
            # Find the positive label (usually 'positive' or 1)
            for label in unique_labels:
                if str(label).lower() in ['positive', '1', 'pos']:
                    pos_label = label
                    break
            if pos_label is None:
                pos_label = unique_labels[1]  # Default to second label
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)
        }
        
        # Store in history
        self.__metrics_history.append(metrics.copy())
        
        return metrics
    
    def get_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        labels: list = None
    ) -> str:
        """
        Generate a detailed classification report.
        
        Args:
            y_true: Actual labels
            y_pred: Predicted labels
            labels: Optional list of labels to use
            
        Returns:
            String containing the classification report
        """
        report_labels = labels if labels is not None else self.__labels
        
        # Ensure labels exist in y_true or y_pred
        unique_in_data = np.unique(np.concatenate([y_true, y_pred]))
        
        # If dynamic labels are needed
        if len(unique_in_data) != len(report_labels):
            report_labels = [str(l) for l in unique_in_data]
            
        return classification_report(y_true, y_pred, target_names=report_labels)
    
    def plot_metrics_bar_chart(
        self,
        metrics: Dict[str, float],
        title: str = "Model Performance Metrics"
    ) -> go.Figure:
        """
        Create a bar chart showing all metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            title: Title for the chart
            
        Returns:
            Plotly Figure object
        """
        # Filter only numeric metrics
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [metrics.get(m, 0) * 100 for m in metric_names]
        display_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Color gradient for bars
        colors = ['#00d4ff', '#00ff88', '#ffaa00', '#ff6b6b']
        
        fig = go.Figure(data=[
            go.Bar(
                x=display_names,
                y=values,
                marker_color=colors,
                text=[f'{v:.1f}%' for v in values],
                textposition='outside',
                textfont=dict(size=14, color='white')
            )
        ])
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color='white'),
                x=0.5
            ),
            xaxis_title="Metric",
            yaxis_title="Score (%)",
            yaxis_range=[0, 110],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False,
            height=400
        )
        
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        
        return fig
    
    def plot_model_comparison(
        self,
        models_metrics: List[Dict[str, float]] = None
    ) -> go.Figure:
        """
        Create a comparison chart for multiple models.
        
        Args:
            models_metrics: List of metric dictionaries for each model
                          If None, uses the internal metrics history
            
        Returns:
            Plotly Figure object
        """
        if models_metrics is None:
            models_metrics = self.__metrics_history
        
        if not models_metrics:
            # Return empty figure if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No models to compare yet. Train multiple models first.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color='white')
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        # Prepare data
        model_names = [m.get('model_name', f'Model {i+1}') for i, m in enumerate(models_metrics)]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        display_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        colors = ['#00d4ff', '#00ff88', '#ffaa00', '#ff6b6b']
        
        fig = go.Figure()
        
        for i, (metric, display_name) in enumerate(zip(metrics, display_names)):
            values = [m.get(metric, 0) * 100 for m in models_metrics]
            fig.add_trace(go.Bar(
                name=display_name,
                x=model_names,
                y=values,
                marker_color=colors[i],
                text=[f'{v:.1f}%' for v in values],
                textposition='outside'
            ))
        
        fig.update_layout(
            title=dict(
                text="Model Comparison",
                font=dict(size=18, color='white'),
                x=0.5
            ),
            barmode='group',
            xaxis_title="Model",
            yaxis_title="Score (%)",
            yaxis_range=[0, 110],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            height=450
        )
        
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        
        return fig
    
    def plot_accuracy_gauge(
        self,
        accuracy: float,
        title: str = "Model Accuracy"
    ) -> go.Figure:
        """
        Create a gauge chart showing accuracy.
        
        Args:
            accuracy: Accuracy value (0-1)
            title: Title for the gauge
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=accuracy * 100,
            title={'text': title, 'font': {'size': 18, 'color': 'white'}},
            number={'suffix': '%', 'font': {'size': 40, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white'},
                'bar': {'color': '#00d4ff'},
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 2,
                'bordercolor': 'white',
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255,0,0,0.3)'},
                    {'range': [50, 75], 'color': 'rgba(255,165,0,0.3)'},
                    {'range': [75, 100], 'color': 'rgba(0,255,0,0.3)'}
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 4},
                    'thickness': 0.75,
                    'value': accuracy * 100
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=300
        )
        
        return fig
    
    def plot_sentiment_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray = None
    ) -> go.Figure:
        """
        Plot the distribution of sentiments in the data.
        
        Args:
            y_true: Actual labels (or training labels)
            y_pred: Predicted labels (optional for comparison)
            
        Returns:
            Plotly Figure object
        """
        unique, counts = np.unique(y_true, return_counts=True)
        
        fig = go.Figure()
        
        # Actual distribution
        colors = ['#ff6b6b' if str(u).lower() in ['negative', '0', 'neg'] 
                  else '#00ff88' for u in unique]
        
        fig.add_trace(go.Pie(
            labels=unique,
            values=counts,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent',
            textfont=dict(size=14, color='white'),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
        ))
        
        fig.update_layout(
            title=dict(
                text="Sentiment Distribution",
                font=dict(size=18, color='white'),
                x=0.5
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            ),
            height=350
        )
        
        return fig
    
    def plot_results(self) -> go.Figure:
        """
        Create a comprehensive results dashboard.
        
        Returns:
            Plotly Figure with subplots showing all metrics
        """
        if not self.__metrics_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No results available. Train a model first.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color='white')
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        return self.plot_model_comparison()
    
    def get_metrics_history(self) -> List[Dict[str, float]]:
        """Get the history of all calculated metrics."""
        return self.__metrics_history.copy()
    
    def clear_history(self) -> None:
        """Clear the metrics history."""
        self.__metrics_history = []
    
    def get_best_model(self, metric: str = 'f1_score') -> Dict[str, Any]:
        """
        Find the best performing model based on a specific metric.
        
        Args:
            metric: Metric to use for comparison (accuracy, precision, recall, f1_score)
            
        Returns:
            Dictionary with best model information
        """
        if not self.__metrics_history:
            return {}
        
        best = max(self.__metrics_history, key=lambda x: x.get(metric, 0))
        return best
    
    def format_metrics_table(self, metrics: Dict[str, float]) -> pd.DataFrame:
        """
        Format metrics as a displayable DataFrame.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Formatted DataFrame
        """
        data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [
                f"{metrics.get('accuracy', 0) * 100:.2f}%",
                f"{metrics.get('precision', 0) * 100:.2f}%",
                f"{metrics.get('recall', 0) * 100:.2f}%",
                f"{metrics.get('f1_score', 0) * 100:.2f}%"
            ]
        }
        return pd.DataFrame(data)


# Testing the Evaluator
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Evaluator Module Test")
    print("=" * 60)
    
    # Create sample predictions
    y_true = np.array(['positive', 'negative', 'positive', 'positive', 
                       'negative', 'positive', 'negative', 'negative'])
    y_pred = np.array(['positive', 'negative', 'positive', 'negative', 
                       'negative', 'positive', 'positive', 'negative'])
    
    # Test the Evaluator
    evaluator = Evaluator()
    evaluator.set_labels(['negative', 'positive'])
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred, "Test Model")
    
    print("\n📊 Metrics:")
    for key, value in metrics.items():
        if key != 'model_name':
            print(f"   {key}: {value:.4f}")
    
    print("\n📋 Classification Report:")
    print(evaluator.get_classification_report(y_true, y_pred))
