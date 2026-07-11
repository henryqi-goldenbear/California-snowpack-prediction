"""
Forecast Explainer Module

Generates natural language explanations for forecast results.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from .mistral_client import MistralClient, get_mistral_client
from datetime import datetime, timedelta
import warnings


@dataclass
class ForecastExplanation:
    """Explanation for a forecast result."""
    forecast_horizon: str
    predicted_value: float
    baseline_value: Optional[float] = None
    change_percentage: Optional[float] = None
    key_drivers: List[Dict[str, Any]] = field(default_factory=list)
    confidence: str = "medium"  # 'low', 'medium', 'high'
    risk_level: str = "normal"  # 'low', 'normal', 'high', 'extreme'
    stakeholder_impact: Dict[str, str] = field(default_factory=dict)
    natural_language_summary: str = ""
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ForecastExplainerConfig:
    """Configuration for the forecast explainer."""
    api_key: Optional[str] = None
    model: str = "mistral-small"
    temperature: float = 0.5
    max_tokens: int = 1500
    stakeholder_types: List[str] = field(default_factory=lambda: [
        "water_managers",
        "ski_resorts", 
        "agriculture",
        "flood_control",
        "general_public"
    ])


class ForecastExplainer:
    """
    A Mistral-based explainer that generates natural language explanations
    for snowpack forecast results.
    """
    
    def __init__(self, config: Optional[ForecastExplainerConfig] = None):
        """
        Initialize the forecast explainer.
        
        Args:
            config: Configuration for the explainer
        """
        self.config = config or ForecastExplainerConfig()
        self.client = get_mistral_client(self.config.api_key, model=self.config.model)
        self._system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the explainer."""
        return """You are an expert hydrologist and communicator specializing in California snowpack.
Your role is to explain snowpack forecast results in clear, actionable language for different stakeholders.

Key concepts to understand:
- SWE (Snow Water Equivalent): The amount of water contained in snowpack, critical for water supply
- Snow depth: Physical depth of snow, important for ski operations
- ENSO: El Niño Southern Oscillation, major climate driver affecting precipitation and temperature
- Water year: Hydrological year from October 1 to September 30
- Accumulation season: Typically November-April
- Melt season: Typically May-July

For each forecast explanation, consider:
1. The magnitude of the forecast relative to historical averages
2. Key climate drivers (temperature, precipitation, ENSO phase)
3. Seasonal context (accumulation vs. melt season)
4. Regional variations (Northern vs. Central vs. Southern Sierra)
5. Implications for different stakeholders

Stakeholder priorities:
- Water managers: Water supply reliability, reservoir operations
- Ski resorts: Snow depth and quality, season length
- Agriculture: Water availability for irrigation, timing of melt
- Flood control: Rapid melt risks, peak flow timing
- General public: Recreation opportunities, water conservation

Always provide:
1. Clear, non-technical summary
2. Key drivers of the forecast
3. Confidence level and uncertainty
4. Stakeholder-specific impacts
5. Actionable recommendations

Be concise but comprehensive. Use analogies and comparisons to make technical concepts accessible."""
    
    def explain_forecast(
        self,
        predictions: Union[pd.DataFrame, pd.Series, np.ndarray, List[float]],
        feature_importance: Optional[Dict[str, float]] = None,
        historical_context: Optional[pd.DataFrame] = None,
        model_info: Optional[Dict] = None,
        forecast_horizon: Optional[str] = None,
        target_variable: str = "swe"
    ) -> ForecastExplanation:
        """
        Generate a comprehensive explanation for forecast results.
        
        Args:
            predictions: Forecast predictions
            feature_importance: Feature importance from the model
            historical_context: Historical data for comparison
            model_info: Information about the model used
            forecast_horizon: Time horizon of the forecast
            target_variable: Target variable being predicted
            
        Returns:
            ForecastExplanation with all details
        """
        # Convert predictions to consistent format
        if isinstance(predictions, (pd.DataFrame, pd.Series)):
            pred_values = predictions.values if hasattr(predictions, 'values') else predictions
        elif isinstance(predictions, np.ndarray):
            pred_values = predictions
        else:
            pred_values = np.array(predictions)
        
        # Get predicted value (use mean for multiple predictions)
        predicted_value = float(np.mean(pred_values))
        
        # Create context for the prompt
        context = self._create_forecast_context(
            predicted_value, 
            feature_importance, 
            historical_context, 
            model_info, 
            forecast_horizon, 
            target_variable
        )
        
        prompt = self._create_system_prompt()
        user_prompt = f"""Explain the following snowpack forecast:

{context}

Provide a comprehensive explanation including:
1. Natural language summary of the forecast
2. Key drivers influencing the prediction
3. Confidence level and uncertainty
4. Comparison to historical averages
5. Risk level assessment
6. Stakeholder-specific impacts
7. Actionable recommendations

Format your response with clear sections for each component."""
        
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
            
            # Parse explanation from response
            explanation = self._parse_forecast_explanation(
                response_text, 
                predicted_value, 
                feature_importance, 
                historical_context, 
                forecast_horizon, 
                target_variable
            )
            
            return explanation
            
        except Exception as e:
            print(f"Error generating forecast explanation: {e}")
            return self._get_default_explanation(
                predicted_value, 
                feature_importance, 
                historical_context, 
                forecast_horizon, 
                target_variable
            )
    
    def _create_forecast_context(
        self,
        predicted_value: float,
        feature_importance: Optional[Dict[str, float]],
        historical_context: Optional[pd.DataFrame],
        model_info: Optional[Dict],
        forecast_horizon: Optional[str],
        target_variable: str
    ) -> str:
        """Create context string for the forecast."""
        context = f"Forecast for {target_variable}: {predicted_value:.2f}\n"
        
        if forecast_horizon:
            context += f"Forecast horizon: {forecast_horizon}\n"
        
        if model_info:
            context += f"Model: {model_info.get('model_type', 'unknown')}\n"
            if 'parameters' in model_info:
                context += f"Model parameters: {json.dumps(model_info['parameters'], indent=2)}\n"
        
        if feature_importance:
            context += "\nFeature importance:\n"
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:10]:  # Top 10 features
                context += f"  - {feature}: {importance:.3f}\n"
        
        if historical_context is not None and not historical_context.empty:
            context += "\nHistorical context:\n"
            if target_variable in historical_context.columns:
                historical_mean = historical_context[target_variable].mean()
                historical_std = historical_context[target_variable].std()
                context += f"  - Historical mean: {historical_mean:.2f}\n"
                context += f"  - Historical std: {historical_std:.2f}\n"
                context += f"  - Current vs mean: {((predicted_value - historical_mean) / historical_mean * 100):.1f}%\n"
            
            if 'date' in historical_context.columns:
                context += f"  - Date range: {historical_context['date'].min()} to {historical_context['date'].max()}\n"
        
        return context
    
    def _parse_forecast_explanation(
        self,
        response_text: str,
        predicted_value: float,
        feature_importance: Optional[Dict[str, float]],
        historical_context: Optional[pd.DataFrame],
        forecast_horizon: Optional[str],
        target_variable: str
    ) -> ForecastExplanation:
        """Parse forecast explanation from Mistral response."""
        explanation = ForecastExplanation(
            forecast_horizon=forecast_horizon or "unknown",
            predicted_value=predicted_value,
            confidence="medium",
            risk_level="normal"
        )
        
        # Calculate baseline and change if historical context available
        if historical_context is not None and target_variable in historical_context.columns:
            baseline = historical_context[target_variable].mean()
            explanation.baseline_value = baseline
            if baseline != 0:
                explanation.change_percentage = ((predicted_value - baseline) / baseline) * 100
        
        # Parse sections from response
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Summary:') or line.startswith('SUMMARY:'):
                explanation.natural_language_summary = line.split(':', 1)[1].strip()
            elif line.startswith('Drivers:') or line.startswith('DRIVERS:'):
                current_section = 'drivers'
            elif line.startswith('Confidence:') or line.startswith('CONFIDENCE:'):
                explanation.confidence = line.split(':', 1)[1].strip().lower()
            elif line.startswith('Risk:') or line.startswith('RISK:'):
                explanation.risk_level = line.split(':', 1)[1].strip().lower()
            elif line.startswith('Water Managers:') or line.startswith('WATER MANAGERS:'):
                explanation.stakeholder_impact['water_managers'] = line.split(':', 1)[1].strip()
            elif line.startswith('Ski Resorts:') or line.startswith('SKI RESORTS:'):
                explanation.stakeholder_impact['ski_resorts'] = line.split(':', 1)[1].strip()
            elif line.startswith('Agriculture:') or line.startswith('AGRICULTURE:'):
                explanation.stakeholder_impact['agriculture'] = line.split(':', 1)[1].strip()
            elif line.startswith('Flood Control:') or line.startswith('FLOOD CONTROL:'):
                explanation.stakeholder_impact['flood_control'] = line.split(':', 1)[1].strip()
            elif line.startswith('Recommendations:') or line.startswith('RECOMMENDATIONS:'):
                current_section = 'recommendations'
            elif line.startswith('General Public:') or line.startswith('GENERAL PUBLIC:'):
                explanation.stakeholder_impact['general_public'] = line.split(':', 1)[1].strip()
            elif current_section == 'drivers' and line and not line.startswith(':'):
                # Parse driver information
                driver_info = self._parse_driver_line(line)
                if driver_info:
                    explanation.key_drivers.append(driver_info)
            elif current_section == 'recommendations' and line and not line.startswith(':'):
                explanation.recommendations.append(line.strip())
        
        # If no drivers parsed, add from feature importance
        if not explanation.key_drivers and feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:
                explanation.key_drivers.append({
                    'feature': feature,
                    'importance': importance,
                    'direction': 'positive' if importance > 0 else 'negative',
                    'description': f"{feature} with importance {importance:.3f}"
                })
        
        return explanation
    
    def _parse_driver_line(self, line: str) -> Optional[Dict]:
        """Parse a driver line from the response."""
        # Simple parsing - could be enhanced
        parts = line.split('|')
        if len(parts) >= 3:
            return {
                'feature': parts[0].strip(),
                'importance': float(parts[1].strip()) if parts[1].strip().replace('.', '').isdigit() else 0,
                'direction': parts[2].strip() if len(parts) > 2 else 'unknown',
                'description': parts[3].strip() if len(parts) > 3 else ''
            }
        return None
    
    def _get_default_explanation(
        self,
        predicted_value: float,
        feature_importance: Optional[Dict[str, float]],
        historical_context: Optional[pd.DataFrame],
        forecast_horizon: Optional[str],
        target_variable: str
    ) -> ForecastExplanation:
        """Get default explanation if API fails."""
        explanation = ForecastExplanation(
            forecast_horizon=forecast_horizon or "30 days",
            predicted_value=predicted_value,
            confidence="medium",
            risk_level="normal"
        )
        
        # Calculate baseline and change
        if historical_context is not None and target_variable in historical_context.columns:
            baseline = historical_context[target_variable].mean()
            explanation.baseline_value = baseline
            if baseline != 0:
                explanation.change_percentage = ((predicted_value - baseline) / baseline) * 100
        
        # Generate natural language summary
        if target_variable == "swe":
            variable_name = "Snow Water Equivalent"
            unit = "inches"
        elif target_variable == "snow_depth":
            variable_name = "Snow Depth"
            unit = "inches"
        else:
            variable_name = target_variable.upper()
            unit = "units"
        
        if explanation.baseline_value:
            change_pct = explanation.change_percentage or 0
            if change_pct > 10:
                trend = "significantly above average"
            elif change_pct > 0:
                trend = "above average"
            elif change_pct < -10:
                trend = "significantly below average"
            elif change_pct < 0:
                trend = "below average"
            else:
                trend = "near average"
            
            explanation.natural_language_summary = (
                f"The forecast predicts {variable_name} of {predicted_value:.1f} {unit} "
                f"for the {forecast_horizon} period, which is {trend} "
                f"({change_pct:+.1f}% from the historical average of {explanation.baseline_value:.1f} {unit})."
            )
        else:
            explanation.natural_language_summary = (
                f"The forecast predicts {variable_name} of {predicted_value:.1f} {unit} "
                f"for the {forecast_horizon} period."
            )
        
        # Add key drivers from feature importance
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:3]:
                explanation.key_drivers.append({
                    'feature': feature,
                    'importance': importance,
                    'direction': 'positive',
                    'description': f"{feature} is a key driver with importance {importance:.3f}"
                })
        
        # Add stakeholder impacts
        if change_pct:
            if change_pct > 10:
                explanation.stakeholder_impact = {
                    'water_managers': "Above average snowpack suggests good water supply. Plan for normal to high reservoir releases.",
                    'ski_resorts': "Excellent snow conditions expected. Extended ski season likely.",
                    'agriculture': "Adequate water supply for irrigation. Normal planting schedules recommended.",
                    'flood_control': "Monitor melt rates closely. Potential for above-normal runoff.",
                    'general_public': "Great conditions for winter recreation. Water conservation still recommended."
                }
                explanation.risk_level = "low"
            elif change_pct < -10:
                explanation.stakeholder_impact = {
                    'water_managers': "Below average snowpack raises water supply concerns. Implement conservation measures.",
                    'ski_resorts': "Poor snow conditions expected. Consider early season closure or snowmaking investments.",
                    'agriculture': "Water supply may be limited. Plan for reduced irrigation allocations.",
                    'flood_control': "Low flood risk from snowmelt. Focus on drought preparedness.",
                    'general_public': "Limited winter recreation opportunities. Heightened water conservation awareness."
                }
                explanation.risk_level = "high"
            else:
                explanation.stakeholder_impact = {
                    'water_managers': "Near-average snowpack. Maintain normal water management operations.",
                    'ski_resorts': "Average snow conditions. Normal ski season operations expected.",
                    'agriculture': "Normal water supply expected. Standard irrigation planning.",
                    'flood_control': "Normal flood risk. Standard monitoring procedures.",
                    'general_public': "Typical winter conditions. Normal recreation and water use."
                }
                explanation.risk_level = "normal"
        
        # Add recommendations
        if change_pct:
            if change_pct > 10:
                explanation.recommendations = [
                    "Monitor snowpack accumulation rates to confirm trend",
                    "Prepare for potential above-normal runoff in spring",
                    "Consider water storage optimization strategies",
                    "Communicate water supply outlook to stakeholders"
                ]
            elif change_pct < -10:
                explanation.recommendations = [
                    "Implement water conservation measures immediately",
                    "Explore alternative water sources",
                    "Communicate drought preparedness to stakeholders",
                    "Monitor for potential early snowmelt"
                ]
            else:
                explanation.recommendations = [
                    "Continue normal monitoring and operations",
                    "Maintain contingency plans for both high and low scenarios",
                    "Regular stakeholder communication"
                ]
        
        return explanation
    
    def explain_feature_importance(
        self,
        feature_importance: Dict[str, float],
        target_variable: str = "swe"
    ) -> Dict:
        """
        Explain feature importance in natural language.
        
        Args:
            feature_importance: Dictionary of feature names and importance scores
            target_variable: Target variable being predicted
            
        Returns:
            Dictionary with explanations for each feature
        """
        prompt = self._create_system_prompt()
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        context = f"Feature importance for {target_variable} prediction:\n"
        for feature, importance in sorted_features:
            context += f"  - {feature}: {importance:.4f}\n"
        
        user_prompt = f"""Explain the following feature importance results for {target_variable} prediction:

{context}

For each of the top 5 features, provide:
1. What the feature represents
2. Why it's important for snowpack prediction
3. How it influences {target_variable}
4. Practical implications

Format your response with clear sections for each feature."""
        
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
            
            # Parse explanations
            explanations = self._parse_feature_explanations(response_text, sorted_features)
            
            return explanations
            
        except Exception as e:
            print(f"Error explaining feature importance: {e}")
            return self._get_default_feature_explanations(sorted_features, target_variable)
    
    def _parse_feature_explanations(
        self, 
        response_text: str, 
        sorted_features: List[tuple]
    ) -> Dict:
        """Parse feature explanations from response."""
        explanations = {}
        
        lines = response_text.split('\n')
        current_feature = None
        
        for line in lines:
            line = line.strip()
            
            # Check if line starts with a feature name
            for feature, importance in sorted_features[:5]:
                if line.startswith(f"{feature}:") or line.startswith(f"Feature: {feature}"):
                    current_feature = feature
                    explanations[current_feature] = {
                        'importance': importance,
                        'explanation': "",
                        'influence': "",
                        'implications': ""
                    }
                    break
            
            if current_feature:
                if line.startswith('Explanation:') or line.startswith('EXPLANATION:'):
                    explanations[current_feature]['explanation'] = line.split(':', 1)[1].strip()
                elif line.startswith('Influence:') or line.startswith('INFLUENCE:'):
                    explanations[current_feature]['influence'] = line.split(':', 1)[1].strip()
                elif line.startswith('Implications:') or line.startswith('IMPLICATIONS:'):
                    explanations[current_feature]['implications'] = line.split(':', 1)[1].strip()
        
        return explanations
    
    def _get_default_feature_explanations(
        self, 
        sorted_features: List[tuple], 
        target_variable: str
    ) -> Dict:
        """Get default feature explanations."""
        explanations = {}
        
        # Common feature explanations for snowpack prediction
        common_explanations = {
            'day_of_year': {
                'explanation': "Day of year captures seasonal patterns in snowpack accumulation and melt.",
                'influence': "Strong seasonal cycle with accumulation in winter and melt in spring.",
                'implications': "Essential for capturing the annual snowpack cycle."
            },
            'temperature': {
                'explanation': "Temperature affects snow accumulation (cold) and melt (warm).",
                'influence': "Higher temperatures generally reduce snowpack through melt.",
                'implications': "Critical for understanding melt dynamics and timing."
            },
            'precipitation': {
                'explanation': "Precipitation provides the water input for snowpack accumulation.",
                'influence': "Higher precipitation increases snowpack, especially as snow.",
                'implications': "Primary driver of snowpack accumulation."
            },
            'swe_lag_7': {
                'explanation': "7-day lag of SWE captures recent snowpack conditions.",
                'influence': "Recent snowpack conditions strongly influence current conditions.",
                'implications': "Helps model understand recent accumulation or melt trends."
            },
            'temperature_anomaly': {
                'explanation': "Temperature anomaly shows deviation from normal conditions.",
                'influence': "Positive anomalies (warmer) reduce snowpack, negative anomalies (colder) increase it.",
                'implications': "Important for understanding climate variability impacts."
            },
            'precipitation_roll_sum_30': {
                'explanation': "30-day rolling sum of precipitation captures recent moisture input.",
                'influence': "Higher recent precipitation leads to higher snowpack.",
                'implications': "Helps model understand recent accumulation patterns."
            },
            'oni': {
                'explanation': "Oceanic Niño Index measures ENSO phase and strength.",
                'influence': "El Niño typically brings wetter conditions, La Niña drier to California.",
                'implications': "Important for seasonal forecasting and climate context."
            },
            'enso_phase': {
                'explanation': "ENSO phase (El Niño, La Niña, Neutral) categorizes climate state.",
                'influence': "Different phases have characteristic impacts on precipitation and temperature.",
                'implications': "Provides categorical climate context for interpretation."
            }
        }
        
        for feature, importance in sorted_features[:5]:
            if feature in common_explanations:
                explanations[feature] = {
                    'importance': importance,
                    **common_explanations[feature]
                }
            else:
                explanations[feature] = {
                    'importance': importance,
                    'explanation': f"{feature} is an important predictor of {target_variable}.",
                    'influence': f"Higher {feature} values tend to be associated with higher {target_variable}.",
                    'implications': f"This feature provides valuable information for {target_variable} prediction."
                }
        
        return explanations
    
    def generate_stakeholder_report(
        self,
        forecast_explanation: ForecastExplanation,
        stakeholder_type: str = "water_managers"
    ) -> str:
        """
        Generate a stakeholder-specific report.
        
        Args:
            forecast_explanation: The forecast explanation to use
            stakeholder_type: Type of stakeholder
            
        Returns:
            Stakeholder-specific report as string
        """
        prompt = self._create_system_prompt()
        
        # Create context from explanation
        context = f"""Forecast Summary:
- Target: {forecast_explanation.forecast_horizon}
- Predicted value: {forecast_explanation.predicted_value:.2f}
- Baseline: {forecast_explanation.baseline_value or 'N/A'}
- Change: {forecast_explanation.change_percentage or 0:+.1f}%
- Confidence: {forecast_explanation.confidence}
- Risk level: {forecast_explanation.risk_level}

Key drivers:
"""
        
        for driver in forecast_explanation.key_drivers:
            context += f"  - {driver.get('feature', 'Unknown')}: {driver.get('description', '')}\n"
        
        context += f"\nStakeholder impact (general):\n"
        for stakeholder, impact in forecast_explanation.stakeholder_impact.items():
            context += f"  - {stakeholder}: {impact}\n"
        
        user_prompt = f"""Generate a detailed report for {stakeholder_type} based on the following forecast:

{context}

Tailor the report specifically for {stakeholder_type} with:
1. Executive summary
2. Key findings relevant to their operations
3. Specific impacts on their activities
4. Recommended actions
5. Timeline for implementation
6. Contacts for follow-up questions

Use professional language appropriate for the stakeholder type.
Format the report with clear sections and bullet points for readability."""
        
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
            
            return response['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"Error generating stakeholder report: {e}")
            return self._get_default_stakeholder_report(forecast_explanation, stakeholder_type)
    
    def _get_default_stakeholder_report(
        self, 
        forecast_explanation: ForecastExplanation, 
        stakeholder_type: str
    ) -> str:
        """Get default stakeholder report."""
        change_pct = forecast_explanation.change_percentage or 0
        predicted = forecast_explanation.predicted_value
        baseline = forecast_explanation.baseline_value or predicted
        
        # Stakeholder-specific templates
        templates = {
            'water_managers': f"""WATER SUPPLY FORECAST REPORT

Executive Summary:
The latest snowpack forecast predicts {predicted:.1f} inches of SWE, which is {change_pct:+.1f}% from the historical average of {baseline:.1f} inches. This represents a {""" + (
                "above-average" if change_pct > 0 else "below-average"
            ) + f""" snowpack year.

Key Findings:
- Current snowpack: {predicted:.1f} inches
- Historical average: {baseline:.1f} inches
- Deviation: {change_pct:+.1f}%
- Confidence: {forecast_explanation.confidence}
- Risk level: {forecast_explanation.risk_level}

Operational Impacts:
- Water supply reliability: {""" + (
                "High - Above average snowpack suggests good water supply" if change_pct > 10 else
                "Moderate - Near average conditions expected" if abs(change_pct) <= 10 else
                "Low - Below average snowpack raises supply concerns"
            ) + """}
- Reservoir operations: Plan for """ + (
                "normal to high releases" if change_pct > 10 else
                "normal operations" if abs(change_pct) <= 10 else
                "conservation measures and reduced releases"
            ) + """
- Drought preparedness: """ + (
                "Standard monitoring" if change_pct > 0 else
                "Enhanced conservation measures recommended"
            ) + """

Recommended Actions:
1. """ + (
                "Optimize reservoir storage for above-normal inflow" if change_pct > 10 else
                "Maintain standard reservoir operations" if abs(change_pct) <= 10 else
                "Implement water conservation measures immediately"
            ) + """
2. """ + (
                "Communicate water supply outlook to stakeholders" if change_pct > 10 else
                "Continue normal stakeholder communication" if abs(change_pct) <= 10 else
                "Develop drought contingency plans"
            ) + """
3. Monitor snowpack accumulation rates weekly
4. Coordinate with regional water agencies

Timeline:
- Immediate: Review and adjust operations based on forecast
- 1 week: Implement recommended actions
- 1 month: Evaluate forecast accuracy and adjust as needed

For questions, contact: Water Operations Center
""",
            
            'ski_resorts': f"""SKI SEASON FORECAST REPORT

Executive Summary:
The snowpack forecast for the upcoming season predicts {predicted:.1f} inches of SWE, which is {change_pct:+.1f}% from average. This indicates a """ + (
                "strong snow year with excellent conditions" if change_pct > 10 else
                "typical snow year with average conditions" if abs(change_pct) <= 10 else
                "weak snow year with challenging conditions"
            ) + f""" expected.

Key Findings:
- Predicted SWE: {predicted:.1f} inches
- Average SWE: {baseline:.1f} inches
- Deviation: {change_pct:+.1f}%
- Snow depth implication: """ + (
                "Above average snow depth expected" if change_pct > 10 else
                "Average snow depth expected" if abs(change_pct) <= 10 else
                "Below average snow depth expected"
            ) + """

Operational Impacts:
- Season length: """ + (
                "Extended season likely" if change_pct > 10 else
                "Normal season length expected" if abs(change_pct) <= 10 else
                "Shorter season likely, consider early closure"
            ) + """
- Snow quality: """ + (
                "Excellent powder conditions" if change_pct > 10 else
                "Good to excellent conditions" if abs(change_pct) <= 10 else
                "Variable conditions, may require snowmaking"
            ) + """
- Visitor experience: """ + (
                "Exceptional guest satisfaction expected" if change_pct > 10 else
                "Typical guest experience" if abs(change_pct) <= 10 else
                "Potential for reduced visitor satisfaction"
            ) + """

Recommended Actions:
1. """ + (
                "Invest in marketing to capitalize on excellent conditions" if change_pct > 10 else
                "Maintain standard marketing and operations" if abs(change_pct) <= 10 else
                "Consider snowmaking investments and promotional pricing"
            ) + """
2. """ + (
                "Plan for extended season operations" if change_pct > 10 else
                "Continue normal season planning" if abs(change_pct) <= 10 else
                "Develop contingency plans for early closure"
            ) + """
3. Monitor weather conditions weekly
4. Coordinate with regional tourism partners

Timeline:
- Immediate: Adjust marketing and operations plans
- 2 weeks: Implement seasonal hiring decisions
- 1 month: Finalize season pass pricing and promotions

For questions, contact: Resort Operations Management
""",
            
            'agriculture': f"""AGRICULTURAL WATER SUPPLY REPORT

Executive Summary:
The snowpack forecast indicates {predicted:.1f} inches of SWE, which is {change_pct:+.1f}% from the historical average. This suggests """ + (
                "above-average water availability" if change_pct > 10 else
                "average water availability" if abs(change_pct) <= 10 else
                "below-average water availability"
            ) + f""" for the upcoming growing season.

Key Findings:
- Predicted water supply: {predicted:.1f} inches SWE
- Historical average: {baseline:.1f} inches SWE
- Deviation: {change_pct:+.1f}%
- Irrigation outlook: """ + (
                "Adequate to abundant" if change_pct > 10 else
                "Adequate" if abs(change_pct) <= 10 else
                "Limited"
            ) + """

Operational Impacts:
- Crop planning: """ + (
                "Consider water-intensive crops" if change_pct > 10 else
                "Normal crop planning" if abs(change_pct) <= 10 else
                "Focus on drought-resistant crops"
            ) + """
- Irrigation scheduling: """ + (
                "Normal to expanded irrigation schedules" if change_pct > 10 else
                "Standard irrigation schedules" if abs(change_pct) <= 10 else
                "Reduced irrigation allocations expected"
            ) + """
- Water costs: """ + (
                "Stable to lower water costs" if change_pct > 10 else
                "Stable water costs" if abs(change_pct) <= 10 else
                "Potential water cost increases"
            ) + """

Recommended Actions:
1. """ + (
                "Plan for full irrigation capacity utilization" if change_pct > 10 else
                "Continue standard irrigation planning" if abs(change_pct) <= 10 else
                "Develop drought contingency irrigation plans"
            ) + """
2. """ + (
                "Consider expanding acreage for water-intensive crops" if change_pct > 10 else
                "Maintain current crop mix" if abs(change_pct) <= 10 else
                "Shift to drought-tolerant crop varieties"
            ) + """
3. Monitor soil moisture conditions regularly
4. Coordinate with water districts for allocation updates

Timeline:
- Immediate: Review crop and irrigation plans
- 1 month: Finalize planting decisions
- 3 months: Implement irrigation strategies

For questions, contact: Agricultural Water Management
"""
        }
        
        # Return template for specified stakeholder or default
        return templates.get(stakeholder_type, templates['water_managers'])
    
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
def get_forecast_explainer(api_key: Optional[str] = None, **kwargs) -> ForecastExplainer:
    """
    Create a forecast explainer with optional configuration.
    
    Args:
        api_key: Mistral API key
        **kwargs: Additional configuration parameters
        
    Returns:
        ForecastExplainer instance
    """
    config = ForecastExplainerConfig(api_key=api_key, **kwargs)
    return ForecastExplainer(config)


def explain_forecast(
    predictions: Union[pd.DataFrame, pd.Series, np.ndarray, List[float]],
    feature_importance: Optional[Dict[str, float]] = None,
    historical_context: Optional[pd.DataFrame] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> ForecastExplanation:
    """
    Generate a forecast explanation.
    
    Args:
        predictions: Forecast predictions
        feature_importance: Feature importance from model
        historical_context: Historical data for comparison
        api_key: Mistral API key
        **kwargs: Additional configuration
        
    Returns:
        ForecastExplanation object
    """
    explainer = get_forecast_explainer(api_key, **kwargs)
    try:
        return explainer.explain_forecast(
            predictions, 
            feature_importance, 
            historical_context
        )
    finally:
        explainer.close()
