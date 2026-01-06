import requests
import json


class OllamaEvaluator:
    def __init__(self, model="llama2"):
        self.model = model
    
    def describe_chart(self, input_data, df_base):
        prompt = f"""
        Analyze this process mining data and provide insights:
        
        Top process data:
        {input_data.to_string()}
        
        Total cases: {df_base['case_id'].nunique()}
        Total events: {len(df_base)}
        
        Please provide a short analysis of the process patterns.
        """
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False  # Request non-streaming response
            }
        )
        
        if response.status_code == 200:
            try:
                result = response.json()
                return result.get("response", "No response received")
            except json.JSONDecodeError:
                # If still getting streaming response, handle it manually
                lines = response.text.strip().split('\n')
                full_response = ""
                for line in lines:
                    if line.strip():
                        try:
                            json_obj = json.loads(line)
                            if "response" in json_obj:
                                full_response += json_obj["response"]
                        except json.JSONDecodeError:
                            continue
                return full_response if full_response else "Failed to parse response"
        else:
            return f"Error: HTTP {response.status_code} - {response.text}"

    def analyze_outliers(self, outlier_summary, outlier_data, full_df=None, outlier_pattern=None):
        """
        Analyze high-confidence outliers and provide insights including detailed activity analysis.

        Args:
            outlier_summary: Summary dict from get_outlier_summary()
            outlier_data: DataFrame containing the high-confidence outlier events
            full_df: Complete DataFrame for context analysis
            outlier_pattern: OutlierDetectionPattern object for accessing detection details
        """
        # Extract key information about the outliers
        stats = outlier_summary.get('statistics', {})
        outlier_details = outlier_summary.get('outlier_details', {})
        top_cases = outlier_summary.get('top_outlier_cases', [])[:5]
        outlier_activities = outlier_summary.get('outlier_activities', [])[:5]

        # Analyze activity patterns in detail
        activity_analysis = ""
        if not outlier_data.empty and 'activity' in outlier_data.columns:
            # Get activity frequency in outliers vs normal events
            outlier_activity_counts = outlier_data['activity'].value_counts()

            if full_df is not None and 'activity' in full_df.columns:
                normal_activity_counts = full_df['activity'].value_counts()
                total_events = len(full_df)

                activity_analysis = "\n\n        DETAILED ACTIVITY ANALYSIS WITH DETECTION REASONS:"
                for activity, outlier_count in outlier_activity_counts.head(10).items():
                    normal_count = normal_activity_counts.get(activity, 0)
                    normal_percentage = (
                        normal_count / total_events * 100) if total_events > 0 else 0
                    outlier_percentage = (
                        outlier_count / len(outlier_data) * 100) if len(outlier_data) > 0 else 0

                    activity_analysis += f"\n        - '{activity}': {outlier_count} outliers ({outlier_percentage:.1f}% of outliers), {normal_count} total occurrences ({normal_percentage:.1f}% of all events)"

                    # Add breakdown of WHY this activity appears as outlier
                    if outlier_pattern and hasattr(outlier_pattern, 'outlier_types'):
                        activity_outliers = outlier_data[outlier_data['activity'] == activity]
                        method_counts = {}

                        for idx in activity_outliers.index:
                            if idx in outlier_pattern.outlier_types:
                                for method in outlier_pattern.outlier_types[idx]:
                                    method_counts[method] = method_counts.get(
                                        method, 0) + 1

                        if method_counts:
                            method_breakdown = []
                            for method, count in method_counts.items():
                                if method == 'time':
                                    method_breakdown.append(
                                        f"{count} due to timing issues")
                                elif method == 'case_duration':
                                    method_breakdown.append(
                                        f"{count} due to case duration")
                                elif method == 'activity_frequency':
                                    method_breakdown.append(
                                        f"{count} due to rarity")
                                elif method == 'resource':
                                    method_breakdown.append(
                                        f"{count} due to resource issues")
                                elif method == 'sequence':
                                    method_breakdown.append(
                                        f"{count} due to workflow violations")
                                elif method == 'case_complexity':
                                    method_breakdown.append(
                                        f"{count} due to case complexity")
                                else:
                                    method_breakdown.append(
                                        f"{count} by {method}")

                            activity_analysis += f"\n          Reasons: {'; '.join(method_breakdown)}"

        # Get detailed outlier examples with WHY they are outliers
        detailed_examples = ""
        if not outlier_data.empty and len(outlier_data) > 0:
            detailed_examples = f"\n\n        DETAILED OUTLIER EXAMPLES WITH DETECTION REASONS (first 5):"
            for i, (idx, row) in enumerate(outlier_data.head(5).iterrows()):
                case_id = row.get('case_id', 'N/A')
                activity = row.get('activity', 'N/A')
                time = row.get('actual_time', 'N/A')
                resource = row.get('resource', 'N/A')

                # Add timing context if available
                timing_info = ""
                if 'relative_time' in row:
                    timing_info = f", Relative Time: {row.get('relative_time', 'N/A')}"
                if 'duration' in row:
                    timing_info += f", Duration: {row.get('duration', 'N/A')}"

                detailed_examples += f"\n        Event {i+1}: Case='{case_id}', Activity='{activity}', Time='{time}', Resource='{resource}'{timing_info}"

                # Add WHY this is an outlier (which detection methods flagged it)
                if outlier_pattern and hasattr(outlier_pattern, 'outlier_types') and idx in outlier_pattern.outlier_types:
                    detection_reasons = outlier_pattern.outlier_types[idx]
                    reason_explanations = []

                    for reason in detection_reasons:
                        if reason == 'time':
                            reason_explanations.append(
                                "unusual timing (off-hours/weekend)")
                        elif reason == 'case_duration':
                            reason_explanations.append(
                                "case took unusually long/short time")
                        elif reason == 'activity_frequency':
                            reason_explanations.append(
                                "rare activity (< 1% frequency)")
                        elif reason == 'resource':
                            reason_explanations.append(
                                "unusual resource workload pattern")
                        elif reason == 'sequence':
                            reason_explanations.append(
                                "rare activity transition/workflow")
                        elif reason == 'case_complexity':
                            reason_explanations.append(
                                "case has unusual complexity (events/activities/duration)")
                        else:
                            reason_explanations.append(
                                f"detected by {reason} method")

                    detailed_examples += f"\n          → OUTLIER REASONS: {'; '.join(reason_explanations)}"

                # Add case context if available
                if full_df is not None and case_id != 'N/A':
                    case_events = full_df[full_df['case_id'] == case_id]
                    if len(case_events) > 1:
                        case_activities = case_events['activity'].tolist()
                        event_position = case_events.index.get_loc(
                            idx) if idx in case_events.index else -1
                        detailed_examples += f"\n          → Case context: {len(case_events)} events total: {' → '.join(case_activities[:5])}{'...' if len(case_activities) > 5 else ''}"
                        if event_position >= 0:
                            detailed_examples += f"\n          → Position in case: event #{event_position + 1}/{len(case_events)}"

        # Build a comprehensive prompt
        prompt = f"""
                You are analyzing HIGH-CONFIDENCE OUTLIERS detected in a **gynecology hospital's business process event log**.

                OUTLIER STATISTICS:
                - Total outliers: {stats.get('total_outliers', 0)} / {stats.get('total_events', 0)} events ({stats.get('outlier_percentage', 0):.1f}%)
                - Cases affected: {stats.get('cases_with_outliers', 0)} / {stats.get('total_cases', 0)} cases
                - Detection methods used: {stats.get('detection_methods_used', 0)}/6
                - Max outlier confidence score: {stats.get('max_outlier_score', 0)}

                OUTLIER BREAKDOWN BY TYPE:
                """

        for outlier_type, details in outlier_details.items():
            prompt += f"\n- {outlier_type.replace('_', ' ').title()}: {details['count']} events ({details['percentage']:.1f}%)"

        if top_cases:
            prompt += f"\n\nTOP CASES WITH MOST OUTLIERS:"
            for case in top_cases:
                prompt += f"\n- Case {case['case_id']}: {case['outlier_events']} outlier events"

        if outlier_activities:
            prompt += f"\n\nMOST COMMON OUTLIER ACTIVITIES:"
            for activity in outlier_activities:
                prompt += f"\n- '{activity['activity']}': {activity['outlier_count']} occurrences"

        # Add detailed context and examples
        prompt += activity_analysis
        prompt += detailed_examples

        prompt += f"""

        Please provide a concise, business-focused analysis using the SPECIFIC DETECTION REASONS provided above.

        ### 1. Root Cause Analysis by Detection Method
        For each detection reason found:
        - **Timing Issues**: Explain why activities happen at off-hours (emergency protocols, staffing, patient needs)
        - **Case Duration Problems**: Why cases take too long/short (complications, efficiency, missing steps)
        - **Rare Activities**: Why certain procedures are uncommon (specialized care, complications, errors)
        - **Resource Issues**: Staff workload problems (understaffing, skill gaps, scheduling)
        - **Workflow Violations**: Why activities happen out of sequence (urgency, protocols, training)
        - **Case Complexity**: Why some cases are too simple/complex (data quality, care variations)

        ### 2. Activity-Specific Business Impact
        For each major outlier activity, based on its detection reasons:
        - **Patient Safety**: How these specific anomalies could affect care quality
        - **Operational Risk**: Impact on efficiency, compliance, resource utilization
        - **Financial Impact**: Cost implications of these process deviations

        ### 3. Prioritized Recommendations
        Based on detection reasons and frequency:
        - **Immediate Actions**: Which outliers need urgent investigation (safety/compliance)
        - **Process Improvements**: Targeted fixes for each detection method
        - **Monitoring Strategy**: KPIs to track improvement for each outlier type

        Focus on the SPECIFIC reasons provided rather than generic possibilities. Use the detection method breakdowns to give targeted, actionable insights.

        Keep the response structured by activity name. Use concise, professional language suitable for a process improvement report.
        """

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )

            if response.status_code == 200:
                try:
                    result = response.json()
                    return result.get("response", "No response received")
                except json.JSONDecodeError:
                    # Handle streaming response manually
                    lines = response.text.strip().split('\n')
                    full_response = ""
                    for line in lines:
                        if line.strip():
                            try:
                                json_obj = json.loads(line)
                                if "response" in json_obj:
                                    full_response += json_obj["response"]
                            except json.JSONDecodeError:
                                continue
                    return full_response if full_response else "Failed to parse response"
            else:
                return f"Error: HTTP {response.status_code} - {response.text}"

        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}. Make sure Ollama is running on localhost:11434"
