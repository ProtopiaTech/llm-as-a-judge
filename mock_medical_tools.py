"""
Mock Medical Tools for KEYTRUDA Agent Evaluation
Provides realistic medical tool responses without real API calls
"""

import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ToolResponse:
    """Standard tool response format"""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None


class MockMedicalTools:
    """Mock implementation of medical tools for agent testing"""

    def __init__(self):
        self.call_log = []  # Track all tool calls for analysis

    def drug_interactions_checker(self, drug1: str, drug2: str, dose1: str = None, dose2: str = None) -> ToolResponse:
        """Check drug interactions between medications"""
        self.call_log.append({
            "tool": "drug_interactions_checker",
            "inputs": {"drug1": drug1, "drug2": drug2, "dose1": dose1, "dose2": dose2}
        })

        # Normalize drug names
        drug1_norm = drug1.lower().strip()
        drug2_norm = drug2.lower().strip()

        # Common interaction patterns
        interactions = {
            ("ibuprofen", "pembrolizumab"): {
                "interaction_level": "moderate",
                "risk_category": "monitor_closely",
                "mechanism": "increased_bleeding_risk",
                "recommendation": "use_lowest_effective_dose",
                "frequency": "monitor_for_bleeding",
                "clinical_significance": "moderate"
            },
            ("warfarin", "pembrolizumab"): {
                "interaction_level": "major",
                "risk_category": "caution_advised",
                "mechanism": "immune_mediated_bleeding",
                "recommendation": "monitor_inr_closely",
                "frequency": "weekly_monitoring",
                "clinical_significance": "high"
            },
            ("metformin", "pembrolizumab"): {
                "interaction_level": "minor",
                "risk_category": "no_significant_interaction",
                "mechanism": "none_identified",
                "recommendation": "continue_as_prescribed",
                "frequency": "routine_monitoring",
                "clinical_significance": "low"
            },
            ("lisinopril", "pembrolizumab"): {
                "interaction_level": "minor",
                "risk_category": "no_significant_interaction",
                "mechanism": "none_identified",
                "recommendation": "continue_as_prescribed",
                "frequency": "routine_monitoring",
                "clinical_significance": "low"
            }
        }

        # Check for interaction
        key = (drug1_norm, drug2_norm)
        reverse_key = (drug2_norm, drug1_norm)

        if key in interactions:
            data = interactions[key]
        elif reverse_key in interactions:
            data = interactions[reverse_key]
        else:
            # Default response for unknown interactions
            data = {
                "interaction_level": "unknown",
                "risk_category": "consult_pharmacist",
                "mechanism": "insufficient_data",
                "recommendation": "discuss_with_healthcare_provider",
                "frequency": "as_directed",
                "clinical_significance": "unknown"
            }

        return ToolResponse(
            success=True,
            data={
                "drug1": drug1,
                "drug2": drug2,
                "dose1": dose1,
                "dose2": dose2,
                **data
            }
        )

    def side_effects_database(self, symptom: str, drug: str) -> ToolResponse:
        """Look up side effect information"""
        self.call_log.append({
            "tool": "side_effects_database",
            "inputs": {"symptom": symptom, "drug": drug}
        })

        symptom_norm = symptom.lower().strip()

        side_effects = {
            "fatigue": {
                "frequency": "very_common_>30%",
                "severity": "mild_to_moderate",
                "onset": "1-4_weeks",
                "duration": "throughout_treatment",
                "management": "rest_energy_conservation",
                "when_to_call_doctor": "if_severe_or_worsening",
                "monitoring": "routine"
            },
            "rash": {
                "frequency": "common_10-30%",
                "severity": "mild_to_severe",
                "onset": "2-8_weeks",
                "duration": "variable",
                "management": "topical_steroids_moisturizers",
                "when_to_call_doctor": "if_severe_or_spreading",
                "monitoring": "weekly_assessment"
            },
            "diarrhea": {
                "frequency": "common_10-30%",
                "severity": "mild_to_severe",
                "onset": "1-6_weeks",
                "duration": "variable",
                "management": "hydration_antidiarrheals",
                "when_to_call_doctor": "if_severe_or_persistent",
                "monitoring": "daily_assessment"
            },
            "nausea": {
                "frequency": "common_10-30%",
                "severity": "mild_to_moderate",
                "onset": "1-2_weeks",
                "duration": "intermittent",
                "management": "antiemetics_small_frequent_meals",
                "when_to_call_doctor": "if_preventing_eating",
                "monitoring": "routine"
            }
        }

        if symptom_norm in side_effects:
            data = side_effects[symptom_norm]
        else:
            data = {
                "frequency": "unknown",
                "severity": "variable",
                "onset": "variable",
                "duration": "variable",
                "management": "consult_healthcare_provider",
                "when_to_call_doctor": "if_concerning",
                "monitoring": "as_directed"
            }

        return ToolResponse(
            success=True,
            data={
                "symptom": symptom,
                "drug": drug,
                **data
            }
        )

    def dosage_calculator(self, weight: float, indication: str, drug: str) -> ToolResponse:
        """Calculate appropriate medication dosing"""
        self.call_log.append({
            "tool": "dosage_calculator",
            "inputs": {"weight": weight, "indication": indication, "drug": drug}
        })

        # Standard KEYTRUDA dosing is typically weight-independent
        standard_dosing = {
            "melanoma": {
                "dose": "200mg",
                "frequency": "every_3_weeks",
                "route": "intravenous",
                "duration": "30_minutes"
            },
            "lung_cancer": {
                "dose": "200mg",
                "frequency": "every_3_weeks",
                "route": "intravenous",
                "duration": "30_minutes"
            },
            "breast_cancer": {
                "dose": "200mg",
                "frequency": "every_3_weeks",
                "route": "intravenous",
                "duration": "30_minutes"
            },
            "kidney_cancer": {
                "dose": "200mg",
                "frequency": "every_3_weeks",
                "route": "intravenous",
                "duration": "30_minutes"
            }
        }

        indication_norm = indication.lower().replace("_", " ").replace("-", " ")

        # Find matching indication
        dose_info = None
        for key, value in standard_dosing.items():
            if key.replace("_", " ") in indication_norm or indication_norm in key.replace("_", " "):
                dose_info = value
                break

        if not dose_info:
            dose_info = {
                "dose": "consult_oncologist",
                "frequency": "as_prescribed",
                "route": "intravenous",
                "duration": "variable"
            }

        return ToolResponse(
            success=True,
            data={
                "weight": weight,
                "indication": indication,
                "drug": drug,
                **dose_info,
                "weight_based": False,
                "bsa_based": False,
                "fixed_dose": True
            }
        )

    def insurance_coverage_checker(self, drug: str, plan: str, indication: str = None) -> ToolResponse:
        """Check insurance coverage for medications"""
        self.call_log.append({
            "tool": "insurance_coverage_checker",
            "inputs": {"drug": drug, "plan": plan, "indication": indication}
        })

        plan_norm = plan.lower().strip()

        coverage_data = {
            "medicare": {
                "covered": True,
                "tier": "specialty_tier",
                "prior_authorization": True,
                "copay": "20%_coinsurance",
                "annual_limit": None,
                "step_therapy": False
            },
            "medicaid": {
                "covered": True,
                "tier": "specialty_tier",
                "prior_authorization": True,
                "copay": "varies_by_state",
                "annual_limit": None,
                "step_therapy": True
            },
            "blue_cross_blue_shield": {
                "covered": True,
                "tier": "tier_5_specialty",
                "prior_authorization": True,
                "copay": "$100-500_per_infusion",
                "annual_limit": None,
                "step_therapy": False
            },
            "aetna": {
                "covered": True,
                "tier": "specialty_tier",
                "prior_authorization": True,
                "copay": "20%_coinsurance",
                "annual_limit": None,
                "step_therapy": False
            }
        }

        if plan_norm in coverage_data:
            data = coverage_data[plan_norm]
        else:
            data = {
                "covered": "unknown",
                "tier": "check_with_plan",
                "prior_authorization": "likely_required",
                "copay": "contact_insurance",
                "annual_limit": "unknown",
                "step_therapy": "possible"
            }

        return ToolResponse(
            success=True,
            data={
                "drug": drug,
                "plan": plan,
                "indication": indication,
                **data
            }
        )

    def clinical_trials_finder(self, cancer_type: str, location: str = None) -> ToolResponse:
        """Find relevant clinical trials"""
        self.call_log.append({
            "tool": "clinical_trials_finder",
            "inputs": {"cancer_type": cancer_type, "location": location}
        })

        # Mock trial data
        trials = [
            {
                "nct_id": "NCT05123456",
                "title": "KEYTRUDA Combination Study in Advanced Cancer",
                "phase": "Phase_3",
                "status": "recruiting",
                "sponsor": "Merck_Sharp_Dohme",
                "location": location or "Multiple_US_Sites"
            },
            {
                "nct_id": "NCT05234567",
                "title": "Pembrolizumab Plus Chemotherapy in Metastatic Cancer",
                "phase": "Phase_2",
                "status": "recruiting",
                "sponsor": "Academic_Medical_Center",
                "location": location or "Multiple_US_Sites"
            }
        ]

        return ToolResponse(
            success=True,
            data={
                "cancer_type": cancer_type,
                "location": location,
                "trials_found": len(trials),
                "trials": trials
            }
        )

    def lab_results_interpreter(self, test_name: str, values: Dict[str, Any]) -> ToolResponse:
        """Interpret lab results in context of treatment"""
        self.call_log.append({
            "tool": "lab_results_interpreter",
            "inputs": {"test_name": test_name, "values": values}
        })

        interpretations = {
            "liver_function": {
                "normal_ranges": {"ALT": "7-35 U/L", "AST": "8-35 U/L"},
                "monitoring_frequency": "every_6_weeks",
                "action_thresholds": {"ALT": ">100 U/L", "AST": ">100 U/L"},
                "clinical_significance": "hepatitis_monitoring"
            },
            "thyroid_function": {
                "normal_ranges": {"TSH": "0.4-4.0 mIU/L", "T4": "4.5-11.2 mcg/dL"},
                "monitoring_frequency": "every_12_weeks",
                "action_thresholds": {"TSH": "<0.1 or >10", "T4": "<4 or >12"},
                "clinical_significance": "thyroiditis_monitoring"
            },
            "baseline_monitoring": {
                "normal_ranges": "varies_by_test",
                "monitoring_frequency": "before_each_cycle",
                "action_thresholds": "clinical_judgment",
                "clinical_significance": "treatment_safety"
            }
        }

        test_norm = test_name.lower().strip()
        if test_norm in interpretations:
            data = interpretations[test_norm]
        else:
            data = {
                "normal_ranges": "consult_lab_reference",
                "monitoring_frequency": "as_directed",
                "action_thresholds": "clinical_judgment",
                "clinical_significance": "treatment_monitoring"
            }

        return ToolResponse(
            success=True,
            data={
                "test_name": test_name,
                "values": values,
                **data
            }
        )

    def appointment_scheduler(self, urgency: str, specialty: str) -> ToolResponse:
        """Provide guidance on scheduling appointments"""
        self.call_log.append({
            "tool": "appointment_scheduler",
            "inputs": {"urgency": urgency, "specialty": specialty}
        })

        scheduling_guidance = {
            ("routine", "oncology"): {
                "timeframe": "2-4_weeks",
                "preparation": "bring_medication_list",
                "tests_needed": "recent_labs_imaging",
                "questions_to_ask": "side_effects_efficacy_next_steps"
            },
            ("urgent", "oncology"): {
                "timeframe": "within_1_week",
                "preparation": "document_symptoms",
                "tests_needed": "stat_labs_if_indicated",
                "questions_to_ask": "symptom_management_treatment_changes"
            },
            ("emergency", "oncology"): {
                "timeframe": "same_day_or_er",
                "preparation": "bring_all_medications",
                "tests_needed": "comprehensive_workup",
                "questions_to_ask": "immediate_management_hospitalization"
            }
        }

        key = (urgency.lower(), specialty.lower())
        if key in scheduling_guidance:
            data = scheduling_guidance[key]
        else:
            data = {
                "timeframe": "contact_office",
                "preparation": "bring_questions",
                "tests_needed": "as_directed",
                "questions_to_ask": "clarify_concerns"
            }

        return ToolResponse(
            success=True,
            data={
                "urgency": urgency,
                "specialty": specialty,
                **data
            }
        )

    def get_call_log(self) -> List[Dict[str, Any]]:
        """Return log of all tool calls for analysis"""
        return self.call_log.copy()

    def clear_call_log(self):
        """Clear the tool call log"""
        self.call_log.clear()


# Global instance for testing
mock_tools = MockMedicalTools()