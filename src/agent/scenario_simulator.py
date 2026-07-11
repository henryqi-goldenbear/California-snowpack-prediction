"""
Scenario Simulator Module

Allows users to specify scenarios in natural language and simulate their impact on snowpack forecasts.
"""

import json
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from .mistral_client import MistralClient, get_mistral_client
from .experiment_agent import ExperimentAgent
from datetime import datetime, timedelta
import warnings


@dataclass
class Scenario:
    """Represents a scenario with parameter adjustments."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    natural_language_description: str = ""
    category: str = "climate"  # 'climate', 'weather', 'enso', 'custom'


@dataclass
class ScenarioResult:
    """Results from running a scenario simulation."""
    scenario: Scenario
    original_predictions: Union[pd.Series, np.ndarray]
    scenario_predictions: Union[pd.Series, np.ndarray]
    parameter_changes: Dict[str, Tuple[Any, Any]]  # (original, new)
    impact_summary: Dict[str, Any]
    comparison_metrics: Dict[str, float]
    natural_language_summary: str = ""


@dataclass
class ScenarioSimulatorConfig:
    """Configuration for the scenario simulator."""
    api_key: Optional[str] = None
    model: str = "mistral-small"
    temperature: float = 0.4
    max_tokens: int = 1500
    default_scenarios: List[Dict] = field(default_factory=lambda: [
        {
            'name': 'Strong El Niño',
            'description': 'Strong El Niño conditions with increased precipitation and warmer temperatures',
            'parameters': {
                'oni': 2.0,
                'enso_phase': 'El Niño',
                'precipitation_multiplier': 1.2,
                'temperature_adjustment': 1.5
            },
            'category': 'enso'
        },
        {
            'name': 'Strong La Niña',
            'description': 'Strong La Niña conditions with decreased precipitation and cooler temperatures',
            'parameters': {
                'oni': -2.0,
                'enso_phase': 'La Niña',
                'precipitation_multiplier': 0.8,
                'temperature_adjustment': -1.0
            },
            'category': 'enso'
        },
        {
            'name': 'Warm Winter',
            'description': 'Warmer than average winter temperatures',
            'parameters': {
                'temperature_adjustment': 2.0,
                'snow_line_elevation': 100  # Higher snow line
            },
            'category': 'climate'
        },
        {
            'name': 'Cold Winter',
            'description': 'Colder than average winter temperatures',
            'parameters': {
                'temperature_adjustment': -2.0,
                'snow_line_elevation': -100  # Lower snow line
            },
            'category': 'climate'
        },
        {
            'name': 'Wet Winter',
            'description': 'Wetter than average winter with increased precipitation',
            'parameters': {
                'precipitation_multiplier': 1.5,
                'storm_frequency': 1.2
            },
            'category': 'weather'
        },
        {
            'name': 'Dry Winter',
            'description': 'Drier than average winter with decreased precipitation',
            'parameters': {
                'precipitation_multiplier': 0.6,
                'storm_frequency': 0.8
            },
            'category': 'weather'
        }
    ])


class ScenarioSimulator:
    """
    A Mistral-based simulator that allows users to specify scenarios in natural language
    and simulate their impact on snowpack forecasts.
    """
    
    def __init__(self, config: Optional[ScenarioSimulatorConfig] = None):
        """
        Initialize the scenario simulator.
        
        Args:
            config: Configuration for the simulator
        """
        self.config = config or ScenarioSimulatorConfig()
        self.client = get_mistral_client(self.config.api_key, model=self.config.model)
        self._system_prompt = self._create_system_prompt()
        self._default_scenarios = self._load_default_scenarios()
        
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the simulator."""
        return """You are an expert hydrologist and climate scientist specializing in California snowpack.
Your role is to parse natural language scenario descriptions and translate them into 
specific parameter adjustments for snowpack modeling.

Key concepts to understand:
- ENSO: El Niño Southern Oscillation - major climate driver
  - El Niño: Warmer Pacific, typically wetter for California
  - La Niña: Cooler Pacific, typically drier for California
- Temperature: Affects snow accumulation (cold) and melt (warm)
- Precipitation: Primary input for snowpack accumulation
- Snow line: Elevation above which precipitation falls as snow
- Storm frequency: How often precipitation events occur
- Snow density: Ratio of SWE to snow depth

Common scenario types:
1. ENSO scenarios: "El Niño", "La Niña", "strong El Niño", "moderate La Niña"
2. Temperature scenarios: "warmer winter", "colder than average", "2°C warmer"
3. Precipitation scenarios: "wetter winter", "drier conditions", "10% more precipitation"
4. Combined scenarios: "El Niño with 2°C warmer temperatures and 15% more precipitation"

Parameter adjustments to consider:
- ENSO indices: oni, mei, nino34, soi
- Temperature: adjustments in °C or °F, snow line elevation
- Precipitation: multipliers (1.1 = 10% more), storm frequency
- Snow properties: snow density, albedo, melt factors
- Temporal: accumulation rates, melt timing

For each scenario, provide:
1. Scenario name
2. Category (enso, climate, weather, custom)
3. Specific parameter adjustments with values
4. Expected impact on snowpack
5. Rationale for the adjustments

Be precise with numerical values. Use realistic ranges for climate variables.
Consider interactions between parameters (e.g., warmer temperatures might reduce snow accumulation even with more precipitation).

Format your response as JSON with clear parameter specifications."""
    
    def _load_default_scenarios(self) -> List[Scenario]:
        """Load default scenarios from configuration."""
        scenarios = []
        for scenario_dict in self.config.default_scenarios:
            scenario = Scenario(
                name=scenario_dict['name'],
                description=scenario_dict['description'],
                parameters=scenario_dict['parameters'],
                category=scenario_dict.get('category', 'custom')
            )
            scenarios.append(scenario)
        return scenarios
    
    def parse_scenario(
        self, 
        scenario_description: str
    ) -> Scenario:
        """
        Parse a natural language scenario description into a Scenario object.
        
        Args:
            scenario_description: Natural language description of the scenario
            
        Returns:
            Scenario object with parsed parameters
        """
        prompt = self._create_system_prompt()
        
        user_prompt = f"""Parse the following natural language scenario and convert it to specific parameter adjustments:

Scenario: "{scenario_description}"

Provide your response as JSON with the following structure:
{{
    "name": "short descriptive name",
    "description": "detailed description",
    "category": "enso|climate|weather|custom",
    "parameters": {{
        "parameter_name": value,
        ...
    }},
    "expected_impact": "description of expected impact on snowpack",
    "rationale": "explanation of why these parameters were chosen"
}}

Consider:
- ENSO phase and strength
- Temperature adjustments
- Precipitation adjustments
- Snow line changes
- Any other relevant climate variables

Be specific with numerical values. Use realistic ranges for California climate."""
        
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
            
            # Parse JSON from response
            scenario_dict = self._extract_json_from_response(response_text)
            
            if scenario_dict:
                scenario = Scenario(
                    name=scenario_dict.get('name', scenario_description[:50]),
                    description=scenario_dict.get('description', scenario_description),
                    parameters=scenario_dict.get('parameters', {}),
                    natural_language_description=scenario_description,
                    category=scenario_dict.get('category', 'custom')
                )
                return scenario
            else:
                # Fallback to simple parsing
                return self._simple_parse_scenario(scenario_description)
                
        except Exception as e:
            print(f"Error parsing scenario: {e}")
            return self._simple_parse_scenario(scenario_description)
    
    def _extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """Extract JSON from Mistral response."""
        # Look for JSON in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _simple_parse_scenario(self, scenario_description: str) -> Scenario:
        """Simple parsing of scenario description without API."""
        scenario_description = scenario_description.lower()
        
        # Initialize parameters
        parameters = {}
        category = 'custom'
        
        # Check for ENSO terms
        if 'el niño' in scenario_description:
            parameters['oni'] = 1.5  # Moderate El Niño
            parameters['enso_phase'] = 'El Niño'
            category = 'enso'
        elif 'la niña' in scenario_description:
            parameters['oni'] = -1.5  # Moderate La Niña
            parameters['enso_phase'] = 'La Niña'
            category = 'enso'
            
        # Check for temperature adjustments
        temp_match = re.search(r'([+-]?\d+(\.\d+)?)\s*°?[CF]\s*(warmer|colder|warmer than|colder than)', scenario_description)
        if temp_match:
            temp_change = float(temp_match.group(1))
            if 'colder' in temp_match.group(3):
                temp_change = -temp_change
            parameters['temperature_adjustment'] = temp_change
            category = 'climate'
            
        # Check for precipitation adjustments
        precip_match = re.search(r'(\d+(\.\d+)?)%\s*(more|less|increase|decrease)\s*(precipitation|rain|snow)', scenario_description)
        if precip_match:
            precip_change = float(precip_match.group(1))
            if 'less' in precip_match.group(3) or 'decrease' in precip_match.group(3):
                precip_change = -precip_change
            # Convert percentage to multiplier
            parameters['precipitation_multiplier'] = 1 + (precip_change / 100)
            category = 'weather'
            
        # Check for simple terms
        if 'warm' in scenario_description and 'temperature' not in parameters:
            parameters['temperature_adjustment'] = 1.0
            category = 'climate'
        elif 'cold' in scenario_description and 'temperature' not in parameters:
            parameters['temperature_adjustment'] = -1.0
            category = 'climate'
            
        if 'wet' in scenario_description and 'precipitation_multiplier' not in parameters:
            parameters['precipitation_multiplier'] = 1.2
            category = 'weather'
        elif 'dry' in scenario_description and 'precipitation_multiplier' not in parameters:
            parameters['precipitation_multiplier'] = 0.8
            category = 'weather'
        
        # Generate name
        if parameters:
            name_parts = []
            if 'enso_phase' in parameters:
                name_parts.append(parameters['enso_phase'])
            if 'temperature_adjustment' in parameters:
                temp = parameters['temperature_adjustment']
                name_parts.append(f"{abs(temp)}°{'C' if abs(temp) < 10 else 'F'} {'warmer' if temp > 0 else 'colder'}")
            if 'precipitation_multiplier' in parameters:
                precip = parameters['precipitation_multiplier']
                pct = (precip - 1) * 100
                name_parts.append(f"{abs(pct):.0f}% {'more' if pct > 0 else 'less'} precipitation")
            
            name = " ".join(name_parts) if name_parts else "Custom Scenario"
        else:
            name = "Custom Scenario"
        
        return Scenario(
            name=name,
            description=scenario_description,
            parameters=parameters,
            natural_language_description=scenario_description,
            category=category
        )
    
    def run_scenario(
        self,
        base_data: pd.DataFrame,
        scenario: Union[Scenario, str],
        model: Optional[Any] = None,
        target_column: str = "swe"
    ) -> ScenarioResult:
        """
        Run a scenario simulation on the base data.
        
        Args:
            base_data: Original DataFrame with snowpack data
            scenario: Scenario object or natural language description
            model: Optional model to use for prediction
            target_column: Target column to predict
            
        Returns:
            ScenarioResult with original and scenario predictions
        """
        # Parse scenario if it's a string
        if isinstance(scenario, str):
            scenario = self.parse_scenario(scenario)
        
        # Apply scenario to base data
        scenario_data = self._apply_scenario_to_data(base_data, scenario)
        
        # Get original predictions (if model provided)
        if model is not None:
            # Assuming model has predict method
            original_predictions = model.predict(base_data)
            scenario_predictions = model.predict(scenario_data)
        else:
            # Use simple heuristic if no model provided
            original_predictions = self._simple_predict(base_data, target_column)
            scenario_predictions = self._simple_predict(scenario_data, target_column)
        
        # Calculate impact metrics
        impact_summary, comparison_metrics = self._calculate_impact(
            original_predictions, 
            scenario_predictions, 
            base_data, 
            scenario_data, 
            target_column
        )
        
        # Generate natural language summary
        natural_language_summary = self._generate_scenario_summary(
            scenario, 
            impact_summary, 
            comparison_metrics, 
            target_column
        )
        
        # Track parameter changes
        parameter_changes = self._track_parameter_changes(base_data, scenario_data)
        
        return ScenarioResult(
            scenario=scenario,
            original_predictions=original_predictions,
            scenario_predictions=scenario_predictions,
            parameter_changes=parameter_changes,
            impact_summary=impact_summary,
            comparison_metrics=comparison_metrics,
            natural_language_summary=natural_language_summary
        )
    
    def _apply_scenario_to_data(
        self, 
        base_data: pd.DataFrame, 
        scenario: Scenario
    ) -> pd.DataFrame:
        """Apply scenario parameters to base data."""
        scenario_data = base_data.copy()
        
        # Apply each parameter adjustment
        for param, value in scenario.parameters.items():
            if param in scenario_data.columns:
                # Direct column adjustment
                if isinstance(value, (int, float)):
                    if param in ['temperature', 'temperature_adjustment']:
                        scenario_data[param] = scenario_data[param] + value
                    elif param == 'precipitation_multiplier':
                        scenario_data['precipitation'] = scenario_data['precipitation'] * value
                    else:
                        scenario_data[param] = value
                else:
                    scenario_data[param] = value
            elif param == 'temperature_adjustment':
                # Adjust all temperature columns
                temp_cols = [col for col in scenario_data.columns if 'temperature' in col.lower()]
                for col in temp_cols:
                    scenario_data[col] = scenario_data[col] + value
            elif param == 'precipitation_multiplier':
                # Adjust all precipitation columns
                precip_cols = [col for col in scenario_data.columns if 'precipitation' in col.lower()]
                for col in precip_cols:
                    scenario_data[col] = scenario_data[col] * value
            elif param == 'oni':
                # Set ENSO index
                if 'oni' in scenario_data.columns:
                    scenario_data['oni'] = value
                elif 'nino34' in scenario_data.columns:
                    scenario_data['nino34'] = value * 0.8  # Approximate relationship
            elif param == 'enso_phase':
                # Set ENSO phase
                if 'enso_phase' in scenario_data.columns:
                    scenario_data['enso_phase'] = value
            elif param == 'snow_line_elevation':
                # Adjust snow line (simplified - affect precipitation as snow vs rain)
                if 'snow_line' in scenario_data.columns:
                    scenario_data['snow_line'] = scenario_data['snow_line'] + value
                # For simplicity, adjust snow accumulation based on elevation
                if 'swe' in scenario_data.columns and value > 0:
                    # Higher snow line means less snow accumulation at lower elevations
                    # Simple approximation: reduce SWE by 1% per 100m elevation increase
                    elevation_factor = 1 - (value / 10000)
                    scenario_data['swe'] = scenario_data['swe'] * max(0.5, elevation_factor)
        
        return scenario_data
    
    def _simple_predict(
        self, 
        data: pd.DataFrame, 
        target_column: str
    ) -> np.ndarray:
        """Simple prediction heuristic if no model provided."""
        if target_column in data.columns:
            # Use historical average as simple prediction
            return np.full(len(data), data[target_column].mean())
        else:
            return np.zeros(len(data))
    
    def _calculate_impact(
        self,
        original_predictions: Union[pd.Series, np.ndarray],
        scenario_predictions: Union[pd.Series, np.ndarray],
        base_data: pd.DataFrame,
        scenario_data: pd.DataFrame,
        target_column: str
    ) -> Tuple[Dict, Dict]:
        """Calculate impact metrics between original and scenario predictions."""
        # Convert to numpy arrays
        if isinstance(original_predictions, pd.Series):
            orig_pred = original_predictions.values
        else:
            orig_pred = original_predictions
            
        if isinstance(scenario_predictions, pd.Series):
            scen_pred = scenario_predictions.values
        else:
            scen_pred = scenario_predictions
        
        # Calculate metrics
        impact_summary = {
            'mean_original': float(np.mean(orig_pred)),
            'mean_scenario': float(np.mean(scen_pred)),
            'absolute_change': float(np.mean(scen_pred) - np.mean(orig_pred)),
            'percentage_change': float(((np.mean(scen_pred) - np.mean(orig_pred)) / np.mean(orig_pred) * 100) if np.mean(orig_pred) != 0 else 0),
            'min_original': float(np.min(orig_pred)),
            'min_scenario': float(np.min(scen_pred)),
            'max_original': float(np.max(orig_pred)),
            'max_scenario': float(np.max(scen_pred)),
            'std_original': float(np.std(orig_pred)),
            'std_scenario': float(np.std(scen_pred))
        }
        
        # Comparison metrics
        comparison_metrics = {
            'rmse': float(np.sqrt(np.mean((scen_pred - orig_pred) ** 2))),
            'mae': float(np.mean(np.abs(scen_pred - orig_pred))),
            'r_squared': float(1 - (np.sum((scen_pred - orig_pred) ** 2) / np.sum((orig_pred - np.mean(orig_pred)) ** 2))) if np.std(orig_pred) != 0 else 0
        }
        
        return impact_summary, comparison_metrics
    
    def _track_parameter_changes(
        self, 
        base_data: pd.DataFrame, 
        scenario_data: pd.DataFrame
    ) -> Dict[str, Tuple[Any, Any]]:
        """Track which parameters were changed and how."""
        parameter_changes = {}
        
        # Check numeric columns for changes
        numeric_cols = base_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in scenario_data.columns:
                base_mean = base_data[col].mean()
                scenario_mean = scenario_data[col].mean()
                
                if base_mean != scenario_mean:
                    parameter_changes[col] = (base_mean, scenario_mean)
        
        # Check categorical columns
        categorical_cols = base_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in scenario_data.columns:
                base_unique = set(base_data[col].unique())
                scenario_unique = set(scenario_data[col].unique())
                
                if base_unique != scenario_unique:
                    parameter_changes[col] = (list(base_unique), list(scenario_unique))
        
        return parameter_changes
    
    def _generate_scenario_summary(
        self,
        scenario: Scenario,
        impact_summary: Dict,
        comparison_metrics: Dict,
        target_column: str
    ) -> str:
        """Generate a natural language summary of the scenario results."""
        prompt = self._create_system_prompt()
        
        # Create context
        context = f"""Scenario: {scenario.name}
Description: {scenario.description}

Parameters adjusted:
"""
        
        for param, value in scenario.parameters.items():
            context += f"  - {param}: {value}\n"
        
        context += f"""
Impact Summary:
- Original mean {target_column}: {impact_summary['mean_original']:.2f}
- Scenario mean {target_column}: {impact_summary['mean_scenario']:.2f}
- Absolute change: {impact_summary['absolute_change']:+.2f}
- Percentage change: {impact_summary['percentage_change']:+.1f}%
- RMSE: {comparison_metrics['rmse']:.2f}
- MAE: {comparison_metrics['mae']:.2f}

Generate a natural language summary explaining:
1. What the scenario represents
2. How it affects snowpack predictions
3. The magnitude and direction of changes
4. Practical implications
5. Key takeaways

Format as a clear, concise paragraph suitable for stakeholders."""
        
        user_prompt = f"""Summarize the following scenario simulation results in natural language:

{context}"""
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens // 2
            )
            
            return response['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"Error generating scenario summary: {e}")
            return self._get_default_scenario_summary(
                scenario, 
                impact_summary, 
                comparison_metrics, 
                target_column
            )
    
    def _get_default_scenario_summary(
        self,
        scenario: Scenario,
        impact_summary: Dict,
        comparison_metrics: Dict,
        target_column: str
    ) -> str:
        """Get default scenario summary."""
        change_pct = impact_summary['percentage_change']
        
        # Determine direction and magnitude
        if change_pct > 10:
            direction = "significantly increases"
            magnitude = "substantial increase"
        elif change_pct > 0:
            direction = "increases"
            magnitude = "moderate increase"
        elif change_pct < -10:
            direction = "significantly decreases"
            magnitude = "substantial decrease"
        elif change_pct < 0:
            direction = "decreases"
            magnitude = "moderate decrease"
        else:
            direction = "has little effect on"
            magnitude = "minimal change in"
        
        # Format target column name
        if target_column == "swe":
            target_name = "Snow Water Equivalent"
        elif target_column == "snow_depth":
            target_name = "snow depth"
        else:
            target_name = target_column
        
        summary = (
            f"The '{scenario.name}' scenario {direction} {target_name} predictions, "
            f"resulting in a {magnitude} of {change_pct:+.1f}% from the baseline. "
            f"The scenario adjusts parameters including {', '.join(list(scenario.parameters.keys())[:3])}. "
            f"This demonstrates how {scenario.category} variations can significantly impact snowpack forecasts, "
            f"highlighting the importance of considering these factors in water resource planning."
        )
        
        return summary
    
    def list_scenarios(self) -> List[Scenario]:
        """List available default scenarios."""
        return self._default_scenarios.copy()
    
    def compare_scenarios(
        self,
        base_data: pd.DataFrame,
        scenarios: List[Union[Scenario, str]],
        model: Optional[Any] = None,
        target_column: str = "swe"
    ) -> Dict[str, ScenarioResult]:
        """
        Compare multiple scenarios against the base data.
        
        Args:
            base_data: Original DataFrame
            scenarios: List of scenarios to compare
            model: Optional model for predictions
            target_column: Target column to predict
            
        Returns:
            Dictionary mapping scenario names to ScenarioResult objects
        """
        results = {}
        
        for scenario in scenarios:
            if isinstance(scenario, str):
                scenario = self.parse_scenario(scenario)
            
            result = self.run_scenario(base_data, scenario, model, target_column)
            results[scenario.name] = result
        
        return results
    
    def suggest_scenarios(
        self,
        data_summary: Dict,
        num_suggestions: int = 3
    ) -> List[Scenario]:
        """
        Suggest relevant scenarios based on data characteristics.
        
        Args:
            data_summary: Summary of the dataset
            num_suggestions: Number of scenarios to suggest
            
        Returns:
            List of suggested scenarios
        """
        prompt = self._create_system_prompt()
        
        user_prompt = f"""Suggest {num_suggestions} relevant scenarios for snowpack prediction based on the following data:

Dataset summary:
{json.dumps(data_summary, indent=2)}

Consider:
- Current climate conditions (ENSO phase, temperature trends)
- Historical patterns in the data
- Key drivers of snowpack variability
- Stakeholder interests

For each scenario, provide:
1. Scenario name
2. Description
3. Category
4. Parameter adjustments
5. Rationale

Format your response as JSON array of scenario objects."""
        
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
            
            # Parse scenarios from response
            scenarios_json = self._extract_json_from_response(response_text)
            
            if scenarios_json and isinstance(scenarios_json, list):
                scenarios = []
                for scenario_dict in scenarios_json[:num_suggestions]:
                    scenario = Scenario(
                        name=scenario_dict.get('name', 'Suggested Scenario'),
                        description=scenario_dict.get('description', ''),
                        parameters=scenario_dict.get('parameters', {}),
                        category=scenario_dict.get('category', 'custom')
                    )
                    scenarios.append(scenario)
                return scenarios
            else:
                return self._default_scenarios[:num_suggestions]
                
        except Exception as e:
            print(f"Error suggesting scenarios: {e}")
            return self._default_scenarios[:num_suggestions]
    
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
def get_scenario_simulator(api_key: Optional[str] = None, **kwargs) -> ScenarioSimulator:
    """
    Create a scenario simulator with optional configuration.
    
    Args:
        api_key: Mistral API key
        **kwargs: Additional configuration parameters
        
    Returns:
        ScenarioSimulator instance
    """
    config = ScenarioSimulatorConfig(api_key=api_key, **kwargs)
    return ScenarioSimulator(config)


def run_scenario(
    base_data: pd.DataFrame,
    scenario: Union[Scenario, str],
    model: Optional[Any] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> ScenarioResult:
    """
    Run a scenario simulation.
    
    Args:
        base_data: Original DataFrame
        scenario: Scenario or natural language description
        model: Optional model for predictions
        api_key: Mistral API key
        **kwargs: Additional configuration
        
    Returns:
        ScenarioResult object
    """
    simulator = get_scenario_simulator(api_key, **kwargs)
    try:
        return simulator.run_scenario(base_data, scenario, model)
    finally:
        simulator.close()


def parse_scenario(
    scenario_description: str,
    api_key: Optional[str] = None,
    **kwargs
) -> Scenario:
    """
    Parse a natural language scenario description.
    
    Args:
        scenario_description: Natural language description
        api_key: Mistral API key
        **kwargs: Additional configuration
        
    Returns:
        Scenario object
    """
    simulator = get_scenario_simulator(api_key, **kwargs)
    try:
        return simulator.parse_scenario(scenario_description)
    finally:
        simulator.close()
