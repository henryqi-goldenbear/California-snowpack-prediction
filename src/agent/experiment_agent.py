"""
Experiment Agent Module

A Mistral-based agent that suggests experiments, features, and model configurations.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from .mistral_client import MistralClient, get_mistral_client
import warnings


@dataclass
class FeatureSuggestion:
    """Suggestion for a feature engineering step."""
    feature_name: str
    description: str
    implementation: str  # Python code or pseudocode
    importance: str  # 'high', 'medium', 'low'
    category: str  # 'temporal', 'climate', 'statistical', 'custom'


@dataclass
class ModelSuggestion:
    """Suggestion for a model configuration."""
    model_type: str  # 'xgboost', 'lstm', 'prophet', 'random_forest', etc.
    parameters: Dict[str, Any]
    rationale: str
    expected_performance: str
    complexity: str  # 'low', 'medium', 'high'


@dataclass
class ExperimentSuggestion:
    """Suggestion for a complete experiment."""
    experiment_name: str
    description: str
    feature_suggestions: List[FeatureSuggestion]
    model_suggestions: List[ModelSuggestion]
    evaluation_metrics: List[str]
    train_test_split: Dict[str, Any]
    expected_outcome: str
    priority: int  # 1-10, higher is more promising


@dataclass
class ExperimentAgentConfig:
    """Configuration for the experiment agent."""
    api_key: Optional[str] = None
    model: str = "mistral-small"
    temperature: float = 0.3  # Lower for more deterministic suggestions
    max_tokens: int = 2000
    num_suggestions: int = 3
    focus_areas: List[str] = field(default_factory=lambda: [
        "feature_engineering", 
        "model_selection", 
        "evaluation_metrics",
        "data_quality"
    ])


class ExperimentAgent:
    """
    A Mistral-based agent that suggests experiments for snowpack prediction.
    
    This agent analyzes dataset characteristics and suggests:
    - Feature engineering steps
    - Model architectures and parameters
    - Evaluation strategies
    - Experiment configurations
    """
    
    def __init__(self, config: Optional[ExperimentAgentConfig] = None):
        """
        Initialize the experiment agent.
        
        Args:
            config: Configuration for the agent
        """
        self.config = config or ExperimentAgentConfig()
        self.client = get_mistral_client(self.config.api_key, model=self.config.model)
        self._system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the agent."""
        return """You are an expert data scientist and hydrologist specializing in snowpack prediction.
Your role is to analyze California snowpack data and suggest optimal machine learning experiments.

You have access to the following data types:
- Snowpack measurements: SWE (Snow Water Equivalent), snow depth
- Climate data: temperature, precipitation, humidity, wind speed
- ENSO data: ONI, MEI, Nino3.4 indices
- Temporal data: dates, seasons, water years

Your suggestions should be:
1. Practical and implementable in Python
2. Based on hydrological domain knowledge
3. Considerate of time-series characteristics
4. Focused on improving forecast accuracy

For feature engineering, consider:
- Temporal features: lags, rolling statistics, cumulative sums
- Climate features: temperature anomalies, precipitation patterns
- ENSO features: phase indicators, strength metrics
- Seasonal features: water year, accumulation/melt periods
- Station-specific features: historical averages, elevation effects

For model selection, consider:
- XGBoost: Good for tabular data with feature importance
- LSTM: Good for capturing temporal dependencies
- Prophet: Good for seasonal patterns and holidays
- Random Forest: Robust to outliers, good interpretability
- Linear models: Good baseline, interpretable

Always provide:
1. Clear rationale for each suggestion
2. Implementation guidance (Python code snippets)
3. Expected impact on model performance
4. Computational complexity considerations

Be concise but thorough. Prioritize suggestions that are most likely to improve forecast accuracy."""
    
    def _create_data_summary_prompt(self, data_summary: Dict) -> str:
        """Create a prompt describing the data summary."""
        prompt = "Analyze the following dataset and suggest experiments for snowpack prediction:\n\n"
        
        prompt += "DATASET SUMMARY:\n"
        prompt += f"- Number of samples: {data_summary.get('num_samples', 'N/A')}\n"
        prompt += f"- Number of features: {data_summary.get('num_features', 'N/A')}\n"
        prompt += f"- Target variable: {data_summary.get('target_variable', 'N/A')}\n"
        prompt += f"- Date range: {data_summary.get('date_range', 'N/A')}\n"
        
        if 'variables' in data_summary:
            prompt += "\nVARIABLES:\n"
            for var, info in data_summary['variables'].items():
                prompt += f"  - {var}: {info.get('description', 'No description')}\n"
                prompt += f"    Type: {info.get('type', 'unknown')}, "
                prompt += f"Missing: {info.get('missing_percent', 0):.1f}%, "
                prompt += f"Range: [{info.get('min', 'N/A')}, {info.get('max', 'N/A')}]\n"
        
        if 'correlations' in data_summary:
            prompt += "\nTOP CORRELATIONS WITH TARGET:\n"
            for var, corr in data_summary['correlations'].items():
                prompt += f"  - {var}: {corr:.3f}\n"
        
        if 'missing_values' in data_summary:
            prompt += "\nMISSING VALUES:\n"
            for var, missing in data_summary['missing_values'].items():
                if missing > 0:
                    prompt += f"  - {var}: {missing:.1f}%\n"
        
        if 'station_info' in data_summary:
            prompt += "\nSTATION INFORMATION:\n"
            prompt += f"- Number of stations: {data_summary['station_info'].get('num_stations', 'N/A')}\n"
            if 'elevations' in data_summary['station_info']:
                elevs = data_summary['station_info']['elevations']
                prompt += f"- Elevation range: [{min(elevs)}, {max(elevs)}] meters\n"
        
        return prompt
    
    def _parse_suggestions(self, response_text: str) -> List[Dict]:
        """Parse suggestions from Mistral response."""
        # Try to parse as JSON first
        try:
            # Look for JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # If not JSON, parse as text
        suggestions = []
        
        # Split by suggestion sections
        sections = response_text.split('SUGGESTION')
        
        for i, section in enumerate(sections[1:], 1):
            suggestion = {
                'id': i,
                'text': section.strip()
            }
            suggestions.append(suggestion)
            
        return suggestions
    
    def _extract_feature_suggestions(self, response_text: str) -> List[FeatureSuggestion]:
        """Extract feature suggestions from response."""
        suggestions = []
        
        # Look for feature suggestions in the text
        lines = response_text.split('\n')
        current_suggestion = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Feature:') or line.startswith('FEATURE:'):
                if current_suggestion:
                    suggestions.append(current_suggestion)
                current_suggestion = FeatureSuggestion(
                    feature_name=line.split(':', 1)[1].strip(),
                    description="",
                    implementation="",
                    importance="medium",
                    category="custom"
                )
            elif line.startswith('Description:') or line.startswith('DESCRIPTION:'):
                if current_suggestion:
                    current_suggestion.description = line.split(':', 1)[1].strip()
            elif line.startswith('Implementation:') or line.startswith('IMPLEMENTATION:'):
                if current_suggestion:
                    current_suggestion.implementation = line.split(':', 1)[1].strip()
            elif line.startswith('Importance:') or line.startswith('IMPORTANCE:'):
                if current_suggestion:
                    current_suggestion.importance = line.split(':', 1)[1].strip().lower()
            elif line.startswith('Category:') or line.startswith('CATEGORY:'):
                if current_suggestion:
                    current_suggestion.category = line.split(':', 1)[1].strip().lower()
        
        if current_suggestion:
            suggestions.append(current_suggestion)
            
        return suggestions
    
    def _extract_model_suggestions(self, response_text: str) -> List[ModelSuggestion]:
        """Extract model suggestions from response."""
        suggestions = []
        
        # Look for model suggestions
        lines = response_text.split('\n')
        current_suggestion = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Model:') or line.startswith('MODEL:'):
                if current_suggestion:
                    suggestions.append(current_suggestion)
                model_type = line.split(':', 1)[1].strip()
                current_suggestion = ModelSuggestion(
                    model_type=model_type,
                    parameters={},
                    rationale="",
                    expected_performance="",
                    complexity="medium"
                )
            elif line.startswith('Parameters:') or line.startswith('PARAMETERS:'):
                if current_suggestion:
                    try:
                        param_str = line.split(':', 1)[1].strip()
                        current_suggestion.parameters = json.loads(param_str)
                    except json.JSONDecodeError:
                        # Parse as key=value pairs
                        params = {}
                        for part in param_str.split(','):
                            if '=' in part:
                                key, value = part.split('=', 1)
                                params[key.strip()] = value.strip()
                        current_suggestion.parameters = params
            elif line.startswith('Rationale:') or line.startswith('RATIONALE:'):
                if current_suggestion:
                    current_suggestion.rationale = line.split(':', 1)[1].strip()
            elif line.startswith('Performance:') or line.startswith('PERFORMANCE:'):
                if current_suggestion:
                    current_suggestion.expected_performance = line.split(':', 1)[1].strip()
            elif line.startswith('Complexity:') or line.startswith('COMPLEXITY:'):
                if current_suggestion:
                    current_suggestion.complexity = line.split(':', 1)[1].strip().lower()
        
        if current_suggestion:
            suggestions.append(current_suggestion)
            
        return suggestions
    
    def suggest_features(
        self,
        variables: List[str],
        target_variable: str = "swe",
        data_summary: Optional[Dict] = None
    ) -> List[FeatureSuggestion]:
        """
        Suggest feature engineering steps.
        
        Args:
            variables: List of available variables
            target_variable: The target variable to predict
            data_summary: Optional summary of the data
            
        Returns:
            List of feature suggestions
        """
        prompt = self._create_system_prompt()
        
        user_prompt = f"""Suggest feature engineering steps for predicting {target_variable}.

Available variables: {', '.join(variables)}

Focus on features that would be most useful for snowpack prediction.
For each feature suggestion, provide:
1. Feature name
2. Description of what it captures
3. Implementation (Python code or pseudocode)
4. Importance (high, medium, low)
5. Category (temporal, climate, statistical, custom)

Provide at least {self.config.num_suggestions} suggestions.

Format your response with clear sections for each suggestion."""
        
        if data_summary:
            user_prompt += f"\n\nAdditional context:\n{json.dumps(data_summary, indent=2)}"
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            response_text = response['choices'][0]['message']['content']
            suggestions = self._extract_feature_suggestions(response_text)
            
            return suggestions
            
        except Exception as e:
            print(f"Error getting feature suggestions: {e}")
            # Return default suggestions
            return self._get_default_feature_suggestions(variables, target_variable)
    
    def _get_default_feature_suggestions(
        self, 
        variables: List[str], 
        target_variable: str
    ) -> List[FeatureSuggestion]:
        """Get default feature suggestions if API fails."""
        suggestions = []
        
        # Temporal features
        if 'date' in variables or any('date' in v.lower() for v in variables):
            suggestions.append(FeatureSuggestion(
                feature_name="day_of_year",
                description="Day of year for seasonal patterns",
                implementation="df['day_of_year'] = df['date'].dt.dayofyear",
                importance="high",
                category="temporal"
            ))
            
            suggestions.append(FeatureSuggestion(
                feature_name="water_year",
                description="Water year (Oct 1 - Sep 30) for hydrological analysis",
                implementation="df['water_year'] = df['date'].apply(lambda x: x.year if x.month >= 10 else x.year - 1)",
                importance="high",
                category="temporal"
            ))
        
        # Lag features
        if target_variable in variables:
            suggestions.append(FeatureSuggestion(
                feature_name=f"{target_variable}_lag_7",
                description=f"7-day lag of {target_variable}",
                implementation=f"df['{target_variable}_lag_7'] = df.groupby('station_id')[target_variable].shift(7)",
                importance="high",
                category="temporal"
            ))
            
            suggestions.append(FeatureSuggestion(
                feature_name=f"{target_variable}_lag_30",
                description=f"30-day lag of {target_variable}",
                implementation=f"df['{target_variable}_lag_30'] = df.groupby('station_id')[target_variable].shift(30)",
                importance="medium",
                category="temporal"
            ))
        
        # Rolling features
        if 'precipitation' in variables:
            suggestions.append(FeatureSuggestion(
                feature_name="precipitation_roll_sum_30",
                description="30-day rolling sum of precipitation",
                implementation="df['precipitation_roll_sum_30'] = df.groupby('station_id')['precipitation'].transform(lambda x: x.rolling(30, min_periods=1).sum())",
                importance="high",
                category="climate"
            ))
        
        if 'temperature' in variables:
            suggestions.append(FeatureSuggestion(
                feature_name="temperature_anomaly",
                description="Temperature anomaly from monthly climatology",
                implementation="monthly_clim = df.groupby('month')['temperature'].transform('mean'); df['temperature_anomaly'] = df['temperature'] - monthly_clim",
                importance="high",
                category="climate"
            ))
        
        # ENSO features
        if any('enso' in v.lower() or 'oni' in v.lower() for v in variables):
            suggestions.append(FeatureSuggestion(
                feature_name="enso_phase",
                description="ENSO phase (El Niño, La Niña, Neutral)",
                implementation="df['enso_phase'] = pd.cut(df['oni'], bins=[-np.inf, -0.5, 0.5, np.inf], labels=['La Niña', 'Neutral', 'El Niño'])",
                importance="high",
                category="climate"
            ))
        
        return suggestions
    
    def suggest_models(
        self,
        data_shape: tuple,
        target_variable: str = "swe",
        available_models: Optional[List[str]] = None,
        data_summary: Optional[Dict] = None
    ) -> List[ModelSuggestion]:
        """
        Suggest model architectures and configurations.
        
        Args:
            data_shape: Shape of the feature matrix (n_samples, n_features)
            target_variable: The target variable to predict
            available_models: List of available model types
            data_summary: Optional summary of the data
            
        Returns:
            List of model suggestions
        """
        n_samples, n_features = data_shape
        
        prompt = self._create_system_prompt()
        
        user_prompt = f"""Suggest model architectures for predicting {target_variable}.

Dataset characteristics:
- Number of samples: {n_samples}
- Number of features: {n_features}
- Target variable: {target_variable}

Available model types: {', '.join(available_models or ['xgboost', 'lstm', 'random_forest', 'prophet', 'linear_regression'])}

For each model suggestion, provide:
1. Model type
2. Parameters (as JSON)
3. Rationale for choosing this model
4. Expected performance
5. Complexity (low, medium, high)

Consider:
- The time-series nature of the data
- Potential non-linear relationships
- Interpretability requirements
- Computational constraints

Provide at least {self.config.num_suggestions} suggestions.

Format your response with clear sections for each suggestion."""
        
        if data_summary:
            user_prompt += f"\n\nAdditional context:\n{json.dumps(data_summary, indent=2)}"
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            response_text = response['choices'][0]['message']['content']
            suggestions = self._extract_model_suggestions(response_text)
            
            return suggestions
            
        except Exception as e:
            print(f"Error getting model suggestions: {e}")
            return self._get_default_model_suggestions(data_shape, target_variable)
    
    def _get_default_model_suggestions(
        self, 
        data_shape: tuple, 
        target_variable: str
    ) -> List[ModelSuggestion]:
        """Get default model suggestions if API fails."""
        n_samples, n_features = data_shape
        
        suggestions = []
        
        # XGBoost - good default for tabular data
        suggestions.append(ModelSuggestion(
            model_type="xgboost",
            parameters={
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            },
            rationale=f"XGBoost is excellent for tabular data with {n_features} features. Handles non-linear relationships well and provides feature importance.",
            expected_performance="High accuracy with good interpretability",
            complexity="medium"
        ))
        
        # LSTM - good for time series
        suggestions.append(ModelSuggestion(
            model_type="lstm",
            parameters={
                "layers": [64, 32],
                "dropout": 0.2,
                "recurrent_dropout": 0.2,
                "batch_size": 32,
                "epochs": 100,
                "validation_split": 0.2
            },
            rationale="LSTM captures temporal dependencies in snowpack data, especially seasonal patterns and accumulation/melt cycles.",
            expected_performance="Good for capturing temporal patterns, but requires more data",
            complexity="high"
        ))
        
        # Random Forest - robust baseline
        suggestions.append(ModelSuggestion(
            model_type="random_forest",
            parameters={
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            },
            rationale="Random Forest is robust to outliers and provides good baseline performance with feature importance.",
            expected_performance="Good baseline performance, less prone to overfitting",
            complexity="medium"
        ))
        
        return suggestions
    
    def suggest_experiments(
        self,
        data_summary: Dict,
        num_suggestions: Optional[int] = None
    ) -> List[ExperimentSuggestion]:
        """
        Suggest complete experiments based on data summary.
        
        Args:
            data_summary: Summary of the dataset
            num_suggestions: Number of experiments to suggest
            
        Returns:
            List of experiment suggestions
        """
        num_suggestions = num_suggestions or self.config.num_suggestions
        
        prompt = self._create_system_prompt()
        
        user_prompt = f"""Suggest {num_suggestions} complete experiments for snowpack prediction.

Dataset summary:
{json.dumps(data_summary, indent=2)}

For each experiment, provide:
1. Experiment name
2. Description
3. Feature engineering steps
4. Model configuration
5. Evaluation metrics
6. Train/test split strategy
7. Expected outcome
8. Priority (1-10)

Consider different approaches:
- Feature engineering focus
- Model architecture focus
- Hyperparameter tuning
- Different time horizons
- Station-specific vs. global models

Format your response with clear sections for each experiment."""
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens * 2  # More tokens for complete experiments
            )
            
            response_text = response['choices'][0]['message']['content']
            
            # Parse experiments from response
            experiments = self._parse_experiments(response_text, data_summary)
            
            return experiments
            
        except Exception as e:
            print(f"Error getting experiment suggestions: {e}")
            return self._get_default_experiments(data_summary, num_suggestions)
    
    def _parse_experiments(self, response_text: str, data_summary: Dict) -> List[ExperimentSuggestion]:
        """Parse experiments from Mistral response."""
        experiments = []
        
        # Try to extract experiment sections
        sections = response_text.split('EXPERIMENT')
        
        for i, section in enumerate(sections[1:], 1):
            try:
                experiment = self._parse_single_experiment(section, i, data_summary)
                experiments.append(experiment)
            except Exception as e:
                print(f"Error parsing experiment {i}: {e}")
                continue
        
        return experiments
    
    def _parse_single_experiment(
        self, 
        section: str, 
        experiment_id: int, 
        data_summary: Dict
    ) -> ExperimentSuggestion:
        """Parse a single experiment from text section."""
        lines = section.split('\n')
        
        experiment = ExperimentSuggestion(
            experiment_name=f"Experiment {experiment_id}",
            description="",
            feature_suggestions=[],
            model_suggestions=[],
            evaluation_metrics=["rmse", "mae", "r2"],
            train_test_split={"test_size": 0.2, "temporal": True},
            expected_outcome="",
            priority=5
        )
        
        current_field = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Name:') or line.startswith('EXPERIMENT NAME:'):
                experiment.experiment_name = line.split(':', 1)[1].strip()
            elif line.startswith('Description:') or line.startswith('DESCRIPTION:'):
                experiment.description = line.split(':', 1)[1].strip()
            elif line.startswith('Features:') or line.startswith('FEATURES:'):
                current_field = 'features'
            elif line.startswith('Models:') or line.startswith('MODELS:'):
                current_field = 'models'
            elif line.startswith('Metrics:') or line.startswith('METRICS:'):
                current_field = 'metrics'
            elif line.startswith('Split:') or line.startswith('SPLIT:'):
                current_field = 'split'
            elif line.startswith('Outcome:') or line.startswith('OUTCOME:'):
                experiment.expected_outcome = line.split(':', 1)[1].strip()
            elif line.startswith('Priority:') or line.startswith('PRIORITY:'):
                try:
                    experiment.priority = int(line.split(':', 1)[1].strip())
                except ValueError:
                    experiment.priority = 5
            elif current_field == 'features' and line and not line.startswith(':'):
                # Parse feature suggestion
                feature = self._parse_feature_line(line)
                if feature:
                    experiment.feature_suggestions.append(feature)
            elif current_field == 'models' and line and not line.startswith(':'):
                # Parse model suggestion
                model = self._parse_model_line(line)
                if model:
                    experiment.model_suggestions.append(model)
            elif current_field == 'metrics' and line and not line.startswith(':'):
                experiment.evaluation_metrics.extend([m.strip() for m in line.split(',')])
            elif current_field == 'split' and line and not line.startswith(':'):
                # Parse split configuration
                split_info = line.split(',')
                for item in split_info:
                    if '=' in item:
                        key, value = item.split('=', 1)
                        experiment.train_test_split[key.strip()] = value.strip()
        
        return experiment
    
    def _parse_feature_line(self, line: str) -> Optional[FeatureSuggestion]:
        """Parse a feature suggestion from a line of text."""
        # Simple parsing - could be enhanced
        parts = line.split('|')
        if len(parts) >= 3:
            return FeatureSuggestion(
                feature_name=parts[0].strip(),
                description=parts[1].strip(),
                implementation=parts[2].strip() if len(parts) > 2 else "",
                importance="medium",
                category="custom"
            )
        return None
    
    def _parse_model_line(self, line: str) -> Optional[ModelSuggestion]:
        """Parse a model suggestion from a line of text."""
        parts = line.split('|')
        if len(parts) >= 2:
            return ModelSuggestion(
                model_type=parts[0].strip(),
                parameters={} if len(parts) < 2 else json.loads(parts[1].strip()),
                rationale="",
                expected_performance="",
                complexity="medium"
            )
        return None
    
    def _get_default_experiments(
        self, 
        data_summary: Dict, 
        num_suggestions: int
    ) -> List[ExperimentSuggestion]:
        """Get default experiments if API fails."""
        experiments = []
        
        # Experiment 1: Baseline with temporal features
        experiments.append(ExperimentSuggestion(
            experiment_name="Baseline Temporal Model",
            description="Baseline experiment using temporal features and XGBoost",
            feature_suggestions=[
                FeatureSuggestion(
                    feature_name="day_of_year",
                    description="Day of year for seasonal patterns",
                    implementation="df['day_of_year'] = df['date'].dt.dayofyear",
                    importance="high",
                    category="temporal"
                ),
                FeatureSuggestion(
                    feature_name="swe_lag_7",
                    description="7-day lag of SWE",
                    implementation="df['swe_lag_7'] = df.groupby('station_id')['swe'].shift(7)",
                    importance="high",
                    category="temporal"
                )
            ],
            model_suggestions=[
                ModelSuggestion(
                    model_type="xgboost",
                    parameters={"n_estimators": 100, "max_depth": 6},
                    rationale="Good baseline for tabular data",
                    expected_performance="Solid baseline performance",
                    complexity="medium"
                )
            ],
            evaluation_metrics=["rmse", "mae", "r2"],
            train_test_split={"test_size": 0.2, "temporal": True},
            expected_outcome="Establish baseline performance for comparison",
            priority=8
        ))
        
        # Experiment 2: Climate-focused with ENSO
        experiments.append(ExperimentSuggestion(
            experiment_name="Climate + ENSO Model",
            description="Experiment focusing on climate variables and ENSO indices",
            feature_suggestions=[
                FeatureSuggestion(
                    feature_name="temperature_anomaly",
                    description="Temperature anomaly from climatology",
                    implementation="monthly_clim = df.groupby('month')['temperature'].transform('mean'); df['temp_anomaly'] = df['temperature'] - monthly_clim",
                    importance="high",
                    category="climate"
                ),
                FeatureSuggestion(
                    feature_name="precipitation_roll_sum_30",
                    description="30-day rolling sum of precipitation",
                    implementation="df['precip_roll_30'] = df.groupby('station_id')['precipitation'].transform(lambda x: x.rolling(30).sum())",
                    importance="high",
                    category="climate"
                ),
                FeatureSuggestion(
                    feature_name="enso_phase",
                    description="ENSO phase indicator",
                    implementation="df['enso_phase'] = pd.cut(df['oni'], bins=[-np.inf, -0.5, 0.5, np.inf], labels=['La Niña', 'Neutral', 'El Niño'])",
                    importance="high",
                    category="climate"
                )
            ],
            model_suggestions=[
                ModelSuggestion(
                    model_type="random_forest",
                    parameters={"n_estimators": 200, "max_depth": 8},
                    rationale="Handles non-linear climate relationships well",
                    expected_performance="Good for capturing climate impacts",
                    complexity="medium"
                )
            ],
            evaluation_metrics=["rmse", "mae", "r2"],
            train_test_split={"test_size": 0.2, "temporal": True},
            expected_outcome="Improve understanding of climate drivers",
            priority=9
        ))
        
        # Experiment 3: LSTM for temporal patterns
        experiments.append(ExperimentSuggestion(
            experiment_name="LSTM Temporal Model",
            description="Experiment using LSTM to capture temporal dependencies",
            feature_suggestions=[
                FeatureSuggestion(
                    feature_name="swe_lag_30",
                    description="30-day lag of SWE",
                    implementation="df['swe_lag_30'] = df.groupby('station_id')['swe'].shift(30)",
                    importance="high",
                    category="temporal"
                ),
                FeatureSuggestion(
                    feature_name="swe_roll_mean_14",
                    description="14-day rolling mean of SWE",
                    implementation="df['swe_roll_14'] = df.groupby('station_id')['swe'].transform(lambda x: x.rolling(14).mean())",
                    importance="medium",
                    category="temporal"
                )
            ],
            model_suggestions=[
                ModelSuggestion(
                    model_type="lstm",
                    parameters={"layers": [64, 32], "dropout": 0.2, "epochs": 100},
                    rationale="Captures complex temporal patterns in snowpack",
                    expected_performance="Excellent for temporal dependencies",
                    complexity="high"
                )
            ],
            evaluation_metrics=["rmse", "mae", "r2"],
            train_test_split={"test_size": 0.2, "temporal": True},
            expected_outcome="Capture complex temporal patterns in snowpack data",
            priority=7
        ))
        
        return experiments[:num_suggestions]
    
    def analyze_data_quality(
        self, 
        data_summary: Dict
    ) -> Dict:
        """
        Analyze data quality and suggest improvements.
        
        Args:
            data_summary: Summary of the dataset
            
        Returns:
            Dictionary with data quality analysis and suggestions
        """
        prompt = self._create_system_prompt()
        
        user_prompt = f"""Analyze the data quality of the following dataset and suggest improvements:

Dataset summary:
{json.dumps(data_summary, indent=2)}

Focus on:
1. Missing data patterns and imputation strategies
2. Outlier detection and handling
3. Data consistency issues
4. Feature distributions and anomalies
5. Temporal consistency

For each issue found, provide:
- Description of the issue
- Severity (low, medium, high)
- Recommended action
- Implementation guidance

Format your response with clear sections."""
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            response_text = response['choices'][0]['message']['content']
            
            # Parse analysis from response
            analysis = self._parse_data_quality_analysis(response_text)
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing data quality: {e}")
            return self._get_default_data_quality_analysis(data_summary)
    
    def _parse_data_quality_analysis(self, response_text: str) -> Dict:
        """Parse data quality analysis from response."""
        analysis = {
            'issues': [],
            'suggestions': [],
            'overall_quality': 'medium'
        }
        
        # Simple parsing - could be enhanced
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Issue:') or line.startswith('ISSUE:'):
                current_section = 'issue'
                analysis['issues'].append({})
            elif line.startswith('Suggestion:') or line.startswith('SUGGESTION:'):
                current_section = 'suggestion'
                analysis['suggestions'].append({})
            elif line.startswith('Severity:') or line.startswith('SEVERITY:'):
                if current_section == 'issue' and analysis['issues']:
                    analysis['issues'][-1]['severity'] = line.split(':', 1)[1].strip()
            elif line.startswith('Description:') or line.startswith('DESCRIPTION:'):
                if current_section == 'issue' and analysis['issues']:
                    analysis['issues'][-1]['description'] = line.split(':', 1)[1].strip()
            elif line.startswith('Action:') or line.startswith('ACTION:'):
                if current_section == 'suggestion' and analysis['suggestions']:
                    analysis['suggestions'][-1]['action'] = line.split(':', 1)[1].strip()
            elif line.startswith('Overall Quality:') or line.startswith('OVERALL QUALITY:'):
                analysis['overall_quality'] = line.split(':', 1)[1].strip().lower()
        
        return analysis
    
    def _get_default_data_quality_analysis(self, data_summary: Dict) -> Dict:
        """Get default data quality analysis."""
        analysis = {
            'issues': [],
            'suggestions': [],
            'overall_quality': 'medium'
        }
        
        # Check for missing values
        if 'missing_values' in data_summary:
            for var, missing in data_summary['missing_values'].items():
                if missing > 10:  # More than 10% missing
                    analysis['issues'].append({
                        'description': f"{var} has {missing:.1f}% missing values",
                        'severity': 'high' if missing > 30 else 'medium'
                    })
                    analysis['suggestions'].append({
                        'action': f"Impute or interpolate missing values in {var}"
                    })
        
        # Check for outliers
        if 'outliers' in data_summary:
            for var, outlier_info in data_summary['outliers'].items():
                if outlier_info.get('count', 0) > 0:
                    analysis['issues'].append({
                        'description': f"{var} has {outlier_info['count']} outliers",
                        'severity': 'medium'
                    })
                    analysis['suggestions'].append({
                        'action': f"Cap or remove outliers in {var}"
                    })
        
        # Check data range
        if 'date_range' in data_summary:
            date_range = data_summary['date_range']
            if date_range.get('years', 0) < 5:
                analysis['issues'].append({
                    'description': f"Dataset only covers {date_range['years']} years",
                    'severity': 'high'
                })
                analysis['suggestions'].append({
                    'action': "Collect more historical data for better model training"
                })
        
        return analysis
    
    def close(self):
        """Close the Mistral client."""
        self.client.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Module-level functions
def get_experiment_agent(api_key: Optional[str] = None, **kwargs) -> ExperimentAgent:
    """
    Create an experiment agent with optional configuration.
    
    Args:
        api_key: Mistral API key
        **kwargs: Additional configuration parameters
        
    Returns:
        ExperimentAgent instance
    """
    config = ExperimentAgentConfig(api_key=api_key, **kwargs)
    return ExperimentAgent(config)


def suggest_features(
    variables: List[str],
    target_variable: str = "swe",
    api_key: Optional[str] = None,
    **kwargs
) -> List[FeatureSuggestion]:
    """
    Suggest feature engineering steps.
    
    Args:
        variables: List of available variables
        target_variable: Target variable to predict
        api_key: Mistral API key
        **kwargs: Additional configuration
        
    Returns:
        List of feature suggestions
    """
    agent = get_experiment_agent(api_key, **kwargs)
    try:
        return agent.suggest_features(variables, target_variable)
    finally:
        agent.close()


def suggest_models(
    data_shape: tuple,
    target_variable: str = "swe",
    api_key: Optional[str] = None,
    **kwargs
) -> List[ModelSuggestion]:
    """
    Suggest model architectures.
    
    Args:
        data_shape: Shape of feature matrix
        target_variable: Target variable to predict
        api_key: Mistral API key
        **kwargs: Additional configuration
        
    Returns:
        List of model suggestions
    """
    agent = get_experiment_agent(api_key, **kwargs)
    try:
        return agent.suggest_models(data_shape, target_variable)
    finally:
        agent.close()
