import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from datetime import datetime
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import altair as alt

# --- Enhanced Helper Functions ---
def parse_headers_enhanced(headers):
    """Enhanced header parsing to handle multiple CSV formats"""
    mapping = {
        'suctionPressures': [],
        'dischargePressures': [],
        'suctionTemps': [],
        'supplyAirTemps': [],
        'dischargeTemps': [],
        'outdoorAirTemps': [],
        'outdoorRH': [],  # Added for outdoor relative humidity
        'coolingSetpoints': [],
        'heatingSetpoints': [],
        'date': None,
        'time': None
    }
    
    for i, h in enumerate(headers):
        h_clean = str(h).strip()
        lower = h_clean.lower()
        
        # Date and Time detection
        if any(keyword in lower for keyword in ['date']) and mapping['date'] is None:
            mapping['date'] = i
        elif any(keyword in lower for keyword in ['time']) and mapping['time'] is None:
            mapping['time'] = i
        
        # Enhanced pressure detection
        elif any(keyword in lower for keyword in ['sucpr', 'suc pr', 'suction pr', 'suction_pr']) or \
             (('suc' in lower or 'suction' in lower) and ('pr' in lower or 'pressure' in lower)):
            mapping['suctionPressures'].append(i)
        
        elif any(keyword in lower for keyword in ['dischg', 'dis chg', 'discharge pr', 'head pr', 'headpr']) or \
             (('discharge' in lower or 'head' in lower) and ('pr' in lower or 'pressure' in lower)):
            mapping['dischargePressures'].append(i)
        
        # Enhanced temperature detection
        elif any(keyword in lower for keyword in ['suctmp', 'suc tmp', 'suction tmp', 'suction_tmp', 'suction temp']):
            mapping['suctionTemps'].append(i)
        
        elif any(keyword in lower for keyword in ['sat ', 'supply air', 'supply_air', 'supply air temp']):
            mapping['supplyAirTemps'].append(i)
        
        elif any(keyword in lower for keyword in ['dischg', 'dis chg', 'discharge']) and 'temp' in lower:
            mapping['dischargeTemps'].append(i)
        
        elif any(keyword in lower for keyword in ['oat', 'outdoor', 'outside']) and ('temp' in lower or 'air' in lower):
            mapping['outdoorAirTemps'].append(i)
        
        # Relative Humidity detection
        elif any(keyword in lower for keyword in ['oa rh', 'outdoor rh', 'outside rh', 'outdoor humidity']):
            mapping['outdoorRH'].append(i)
        
        # Setpoint detection
        elif any(keyword in lower for keyword in ['csp', 'cool', 'cooling']) and ('sp' in lower or 'setpoint' in lower):
            mapping['coolingSetpoints'].append(i)
        
        elif any(keyword in lower for keyword in ['hsp', 'heat', 'heating']) and ('sp' in lower or 'setpoint' in lower):
            mapping['heatingSetpoints'].append(i)
    
    return mapping

def combine_dataframes(dataframes_with_files):
    """Combine multiple dataframes and standardize column names"""
    combined_data = []
    all_mappings = []
    
    for df, filename in dataframes_with_files:
        headers = df.columns.tolist()
        mapping = parse_headers_enhanced(headers)
        all_mappings.append(mapping)
        
        # Create standardized columns for this dataframe
        standardized_df = pd.DataFrame()
        
        # Add source file info
        standardized_df['source_file'] = filename
        
        # Standardize datetime
        try:
            if mapping['date'] is not None and mapping['time'] is not None:
                standardized_df['datetime'] = pd.to_datetime(df.iloc[:, mapping['date']].astype(str) + ' ' + df.iloc[:, mapping['time']].astype(str))
            elif mapping['date'] is not None:
                standardized_df['datetime'] = pd.to_datetime(df.iloc[:, mapping['date']])
            elif mapping['time'] is not None:
                standardized_df['datetime'] = pd.to_datetime(df.iloc[:, mapping['time']])
            else:
                standardized_df['datetime'] = pd.to_datetime('today')  # Fallback
        except:
            standardized_df['datetime'] = pd.NaT
        
        # Standardize pressure columns
        for idx_list, new_name in [(mapping['suctionPressures'], 'suction_pressure'),
                                   (mapping['dischargePressures'], 'discharge_pressure')]:
            if idx_list:
                # If multiple columns, take the average or first one
                if len(idx_list) == 1:
                    standardized_df[new_name] = pd.to_numeric(df.iloc[:, idx_list[0]], errors='coerce')
                else:
                    # Average multiple columns
                    temp_cols = [pd.to_numeric(df.iloc[:, idx], errors='coerce') for idx in idx_list]
                    standardized_df[new_name] = pd.concat(temp_cols, axis=1).mean(axis=1)
        
        # Standardize temperature columns
        for idx_list, new_name in [(mapping['suctionTemps'], 'suction_temp'),
                                   (mapping['supplyAirTemps'], 'supply_air_temp'),
                                   (mapping['dischargeTemps'], 'discharge_temp'),
                                   (mapping['outdoorAirTemps'], 'outdoor_air_temp')]:
            if idx_list:
                if len(idx_list) == 1:
                    standardized_df[new_name] = pd.to_numeric(df.iloc[:, idx_list[0]], errors='coerce')
                else:
                    temp_cols = [pd.to_numeric(df.iloc[:, idx], errors='coerce') for idx in idx_list]
                    standardized_df[new_name] = pd.concat(temp_cols, axis=1).mean(axis=1)
        
        # Standardize humidity columns
        if mapping['outdoorRH']:
            if len(mapping['outdoorRH']) == 1:
                standardized_df['outdoor_rh'] = pd.to_numeric(df.iloc[:, mapping['outdoorRH'][0]], errors='coerce')
            else:
                temp_cols = [pd.to_numeric(df.iloc[:, idx], errors='coerce') for idx in mapping['outdoorRH']]
                standardized_df['outdoor_rh'] = pd.concat(temp_cols, axis=1).mean(axis=1)
        
        # Standardize setpoint columns
        for idx_list, new_name in [(mapping['coolingSetpoints'], 'cooling_setpoint'),
                                   (mapping['heatingSetpoints'], 'heating_setpoint')]:
            if idx_list:
                if len(idx_list) == 1:
                    standardized_df[new_name] = pd.to_numeric(df.iloc[:, idx_list[0]], errors='coerce')
                else:
                    temp_cols = [pd.to_numeric(df.iloc[:, idx], errors='coerce') for idx in idx_list]
                    standardized_df[new_name] = pd.concat(temp_cols, axis=1).mean(axis=1)
        
        combined_data.append(standardized_df)
    
    # Combine all standardized dataframes
    final_combined = pd.concat(combined_data, ignore_index=True, sort=False)
    
    return final_combined, all_mappings

def analyze_combined_hvac_data(combined_df):
    """Analyze the combined HVAC data and return issues"""
    issues = []
    
    # Analyze each standardized column
    for col in combined_df.columns:
        if col in ['source_file', 'datetime']:
            continue
            
        col_data = pd.to_numeric(combined_df[col], errors='coerce').dropna()
        if len(col_data) == 0:
            continue
        
        # Suction Pressure Analysis
        if col == 'suction_pressure':
            avg_pressure = col_data.mean()
            if avg_pressure < 60:
                issues.append({
                    "severity": "high",
                    "message": f"Low suction pressure across all systems (Avg: {avg_pressure:.1f} PSI)",
                    "explanation": "Low suction pressure typically indicates refrigerant undercharge, restriction in liquid line, or evaporator issues.",
                    "suggestions": ["Check for refrigerant leaks across all units", "Verify proper refrigerant charge", "Inspect liquid lines for restrictions", "Check evaporator coil condition"],
                    "issue_type": "refrigerant_system",
                    "priority": 1
                })
            elif avg_pressure > 90:
                issues.append({
                    "severity": "medium",
                    "message": f"High suction pressure detected across systems (Avg: {avg_pressure:.1f} PSI)",
                    "explanation": "High suction pressure may indicate overcharge, compressor issues, or excessive heat load.",
                    "suggestions": ["Check refrigerant charge levels", "Inspect compressor operation", "Verify cooling load calculations", "Check for non-condensables"],
                    "issue_type": "refrigerant_system",
                    "priority": 3
                })
        
        # Discharge Pressure Analysis
        elif col == 'discharge_pressure':
            avg_pressure = col_data.mean()
            if avg_pressure > 400:
                issues.append({
                    "severity": "high",
                    "message": f"High discharge pressure across systems (Avg: {avg_pressure:.1f} PSI)",
                    "explanation": "High discharge pressure indicates condenser problems, overcharge, or airflow restrictions.",
                    "suggestions": ["Clean condenser coils on all units", "Check condenser fan operation", "Verify proper airflow", "Check for system overcharge"],
                    "issue_type": "condenser_system",
                    "priority": 1
                })
            elif avg_pressure < 150:
                issues.append({
                    "severity": "medium",
                    "message": f"Low discharge pressure across systems (Avg: {avg_pressure:.1f} PSI)",
                    "explanation": "Low discharge pressure may indicate undercharge, compressor wear, or valve problems.",
                    "suggestions": ["Check refrigerant charge on all units", "Test compressor valves", "Inspect for internal leaks", "Verify compressor operation"],
                    "issue_type": "compressor_system",
                    "priority": 2
                })
        
        # Supply Air Temperature Analysis
        elif col == 'supply_air_temp':
            avg_temp = col_data.mean()
            temp_range = col_data.max() - col_data.min()
            
            if avg_temp > 70:  # High supply air temp
                issues.append({
                    "severity": "high",
                    "message": f"High supply air temperature across systems (Avg: {avg_temp:.1f}°F)",
                    "explanation": "High supply air temperature indicates poor cooling performance, dirty filters, or refrigerant issues.",
                    "suggestions": ["Replace air filters on all units", "Check refrigerant levels", "Inspect evaporator coils", "Verify proper airflow"],
                    "issue_type": "cooling_performance",
                    "priority": 1
                })
            elif avg_temp < 45:  # Very low supply air temp
                issues.append({
                    "severity": "medium",
                    "message": f"Very low supply air temperature detected (Avg: {avg_temp:.1f}°F)",
                    "explanation": "Extremely low supply air temperature may indicate overcooling or control issues.",
                    "suggestions": ["Check thermostat settings", "Verify cooling load calculations", "Inspect damper operation", "Review control sequences"],
                    "issue_type": "control_system",
                    "priority": 4
                })
            
            if temp_range > 20:
                issues.append({
                    "severity": "medium",
                    "message": f"High supply air temperature variation (Range: {temp_range:.1f}°F)",
                    "explanation": "Large temperature swings indicate cycling issues, control problems, or system instability.",
                    "suggestions": ["Check thermostat operation", "Verify control settings", "Inspect for short cycling", "Review system capacity"],
                    "issue_type": "control_system",
                    "priority": 3
                })
        
        # Outdoor Air Temperature Analysis (for context)
        elif col == 'outdoor_air_temp':
            avg_temp = col_data.mean()
            if avg_temp > 95:
                issues.append({
                    "severity": "low",
                    "message": f"High outdoor air temperature conditions (Avg: {avg_temp:.1f}°F)",
                    "explanation": "High outdoor temperatures increase system load and may affect performance.",
                    "suggestions": ["Monitor system performance during peak hours", "Consider scheduling maintenance during cooler periods", "Check condenser performance in high ambient conditions"],
                    "issue_type": "environmental",
                    "priority": 5
                })
        
        # Outdoor Relative Humidity Analysis
        elif col == 'outdoor_rh':
            avg_rh = col_data.mean()
            if avg_rh > 80:
                issues.append({
                    "severity": "low",
                    "message": f"High outdoor relative humidity (Avg: {avg_rh:.1f}%)",
                    "explanation": "High humidity increases latent cooling load and may affect dehumidification performance.",
                    "suggestions": ["Monitor indoor humidity levels", "Check dehumidification performance", "Verify proper drainage", "Consider humidity control adjustments"],
                    "issue_type": "environmental",
                    "priority": 5
                })
        
        # General outlier detection
        if len(col_data) > 10:
            q1, q3 = np.percentile(col_data, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                outliers = col_data[(col_data < q1 - 1.5*iqr) | (col_data > q3 + 1.5*iqr)]
                if len(outliers) > len(col_data) * 0.15:
                    issues.append({
                        "severity": "medium",
                        "message": f"Frequent unusual readings in {col.replace('_', ' ').title()}",
                        "explanation": "Multiple abnormal readings suggest equipment malfunction, sensor drift, or operating condition changes.",
                        "suggestions": ["Calibrate sensors", "Check equipment operation during abnormal reading periods", "Review maintenance logs", "Monitor for recurring patterns"],
                        "outlier_count": len(outliers),
                        "issue_type": "sensor_system",
                        "priority": 4
                    })
    
    return issues

def generate_service_recommendations(issues):
    """Generate prioritized service recommendations for technicians"""
    # Sort issues by priority (lower number = higher priority)
    sorted_issues = sorted(issues, key=lambda x: x.get('priority', 999))
    
    recommendations = {
        'immediate_actions': [],
        'this_visit_priorities': [],
        'next_visit_items': [],
        'monitoring_items': []
    }
    
    for issue in sorted_issues:
        priority = issue.get('priority', 5)
        
        if priority == 1:  # Immediate
            recommendations['immediate_actions'].extend(issue['suggestions'])
        elif priority in [2, 3]:  # This visit
            recommendations['this_visit_priorities'].extend(issue['suggestions'])
        elif priority == 4:  # Next visit
            recommendations['next_visit_items'].extend(issue['suggestions'])
        else:  # Monitor
            recommendations['monitoring_items'].extend(issue['suggestions'])
    
    # Remove duplicates while preserving order
    for category in recommendations:
        seen = set()
        unique_items = []
        for item in recommendations[category]:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        recommendations[category] = unique_items
    
    return recommendations

def create_unified_plot(combined_df):
    """Create a unified plot showing all system parameters"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Define color scheme for different files
    files = combined_df['source_file'].unique()
    colors_list = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    file_colors = {file: colors_list[i % len(colors_list)] for i, file in enumerate(files)}
    
    plot_idx = 0
    
    # Plot 1: System Pressures
    if 'suction_pressure' in combined_df.columns or 'discharge_pressure' in combined_df.columns:
        for file in files:
            file_data = combined_df[combined_df['source_file'] == file]
            if 'suction_pressure' in file_data.columns:
                axes[plot_idx].plot(file_data['datetime'], file_data['suction_pressure'], 
                                  label=f'Suction - {file}', color=file_colors[file], alpha=0.7, marker='o', markersize=2)
            if 'discharge_pressure' in file_data.columns:
                axes[plot_idx].plot(file_data['datetime'], file_data['discharge_pressure'], 
                                  label=f'Discharge - {file}', color=file_colors[file], alpha=0.7, linestyle='--', marker='s', markersize=2)
        
        axes[plot_idx].set_title("System Pressures - All Units", fontweight='bold')
        axes[plot_idx].set_ylabel("Pressure (PSI)")
        axes[plot_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].tick_params(axis='x', rotation=45)
        plot_idx += 1
    
    # Plot 2: Supply Air Temperatures
    if 'supply_air_temp' in combined_df.columns:
        for file in files:
            file_data = combined_df[combined_df['source_file'] == file]
            if 'supply_air_temp' in file_data.columns:
                axes[plot_idx].plot(file_data['datetime'], file_data['supply_air_temp'], 
                                  label=f'Supply Air - {file}', color=file_colors[file], alpha=0.8, marker='o', markersize=2)
        
        axes[plot_idx].set_title("Supply Air Temperature - All Units", fontweight='bold')
        axes[plot_idx].set_ylabel("Temperature (°F)")
        axes[plot_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].tick_params(axis='x', rotation=45)
        plot_idx += 1
    
    # Plot 3: System Temperatures
    if any(col in combined_df.columns for col in ['suction_temp', 'discharge_temp']):
        for file in files:
            file_data = combined_df[combined_df['source_file'] == file]
            if 'suction_temp' in file_data.columns:
                axes[plot_idx].plot(file_data['datetime'], file_data['suction_temp'], 
                                  label=f'Suction Temp - {file}', color=file_colors[file], alpha=0.7, marker='o', markersize=2)
            if 'discharge_temp' in file_data.columns:
                axes[plot_idx].plot(file_data['datetime'], file_data['discharge_temp'], 
                                  label=f'Discharge Temp - {file}', color=file_colors[file], alpha=0.7, linestyle='--', marker='s', markersize=2)
        
        axes[plot_idx].set_title("System Temperatures - All Units", fontweight='bold')
        axes[plot_idx].set_ylabel("Temperature (°F)")
        axes[plot_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].tick_params(axis='x', rotation=45)
        plot_idx += 1
    
    # Plot 4: Outdoor Conditions
    if 'outdoor_air_temp' in combined_df.columns:
        for file in files:
            file_data = combined_df[combined_df['source_file'] == file]
            if 'outdoor_air_temp' in file_data.columns:
                axes[plot_idx].plot(file_data['datetime'], file_data['outdoor_air_temp'], 
                                  label=f'OA Temp - {file}', color=file_colors[file], alpha=0.8, marker='d', markersize=2)
        
        axes[plot_idx].set_title("Outdoor Air Temperature", fontweight='bold')
        axes[plot_idx].set_ylabel("Temperature (°F)")
        axes[plot_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].tick_params(axis='x', rotation=45)
        plot_idx += 1
    
    # Plot 5: Outdoor Relative Humidity
    if 'outdoor_rh' in combined_df.columns:
        for file in files:
            file_data = combined_df[combined_df['source_file'] == file]
            if 'outdoor_rh' in file_data.columns:
                axes[plot_idx].plot(file_data['datetime'], file_data['outdoor_rh'], 
                                  label=f'OA RH - {file}', color=file_colors[file], alpha=0.8, marker='v', markersize=2)
        
        axes[plot_idx].set_title("Outdoor Relative Humidity", fontweight='bold')
        axes[plot_idx].set_ylabel("Relative Humidity (%)")
        axes[plot_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].tick_params(axis='x', rotation=45)
        plot_idx += 1
    
    # Plot 6: Setpoints (if available)
    if any(col in combined_df.columns for col in ['cooling_setpoint', 'heating_setpoint']):
        for file in files:
            file_data = combined_df[combined_df['source_file'] == file]
            if 'cooling_setpoint' in file_data.columns:
                axes[plot_idx].plot(file_data['datetime'], file_data['cooling_setpoint'], 
                                  label=f'Cooling SP - {file}', color=file_colors[file], alpha=0.7, marker='^', markersize=2)
            if 'heating_setpoint' in file_data.columns:
                axes[plot_idx].plot(file_data['datetime'], file_data['heating_setpoint'], 
                                  label=f'Heating SP - {file}', color=file_colors[file], alpha=0.7, linestyle=':', marker='v', markersize=2)
        
        axes[plot_idx].set_title("Temperature Setpoints", fontweight='bold')
        axes[plot_idx].set_ylabel("Temperature (°F)")
        axes[plot_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].tick_params(axis='x', rotation=45)
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle("Multi-System HVAC Performance Analysis", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def generate_comprehensive_pdf_report(project_title, logo_file, combined_df, issues, recommendations):
    """Generate comprehensive PDF report with service recommendations"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=12,
        textColor=colors.darkblue
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=8,
        textColor=colors.darkred
    )
    
    normal_style = styles['Normal']
    normal_style.alignment = TA_JUSTIFY
    
    story = []
    
    # Add logo if provided
    if logo_file:
        try:
            logo_file.seek(0)
            logo = Image(logo_file, width=2*inch, height=1*inch)
            logo.hAlign = 'CENTER'
            story.append(logo)
            story.append(Spacer(1, 12))
        except:
            pass
    
    # Title
    story.append(Paragraph(project_title, title_style))
    story.append(Paragraph("Multi-System HVAC Diagnostic Analysis Report", heading_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 20))
    
    # System Overview
    story.append(Paragraph("System Overview", heading_style))
    files_analyzed = combined_df['source_file'].unique()
    story.append(Paragraph(f"<b>Files Analyzed:</b> {len(files_analyzed)}", normal_style))
    for file in files_analyzed:
        story.append(Paragraph(f"• {file}", normal_style))
    story.append(Paragraph(f"<b>Total Data Points:</b> {len(combined_df)}", normal_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    if issues:
        high_count = len([i for i in issues if i['severity'] == 'high'])
        medium_count = len([i for i in issues if i['severity'] == 'medium'])
        low_count = len([i for i in issues if i['severity'] == 'low'])
        
        summary_text = f"""
        This comprehensive analysis of multiple HVAC systems identifies {len(issues)} total issues requiring attention:
        <br/>• {high_count} High Priority Issues (require immediate attention)
        <br/>• {medium_count} Medium Priority Issues (should be addressed during this visit)
        <br/>• {low_count} Low Priority Issues (monitor and plan for next visit)
        """
        story.append(Paragraph(summary_text, normal_style))
    else:
        story.append(Paragraph("Multi-system analysis shows no immediate issues detected. All parameters appear to be within normal operating ranges across all analyzed units.", normal_style))
    
    story.append(Spacer(1, 20))
    
    # SERVICE RECOMMENDATIONS SECTION
    story.append(Paragraph("🔧 SERVICE TECHNICIAN RECOMMENDATIONS", heading_style))
    story.append(Paragraph("Prioritized action items for optimal service efficiency", subheading_style))
    
    if recommendations['immediate_actions']:
        story.append(Paragraph("🚨 IMMEDIATE ACTIONS REQUIRED", subheading_style))
        story.append(Paragraph("Address these items first before proceeding with other work:", normal_style))
        for i, action in enumerate(recommendations['immediate_actions'], 1):
            story.append(Paragraph(f"{i}. {action}", normal_style))
        story.append(Spacer(1, 12))
    
    if recommendations['this_visit_priorities']:
        story.append(Paragraph("🔴 THIS VISIT PRIORITIES", subheading_style))
        story.append(Paragraph("Complete these items during current service visit:", normal_style))
        for i, action in enumerate(recommendations['this_visit_priorities'], 1):
            story.append(Paragraph(f"{i}. {action}", normal_style))
        story.append(Spacer(1, 12))
    
    if recommendations['next_visit_items']:
        story.append(Paragraph("🟡 NEXT VISIT PLANNING", subheading_style))
        story.append(Paragraph("Schedule these items for the next service visit:", normal_style))
        for i, action in enumerate(recommendations['next_visit_items'], 1):
            story.append(Paragraph(f"{i}. {action}", normal_style))
        story.append(Spacer(1, 12))
    
    if recommendations['monitoring_items']:
        story.append(Paragraph("📊 ONGOING MONITORING", subheading_style))
        story.append(Paragraph("Continue to monitor these conditions:", normal_style))
        for i, action in enumerate(recommendations['monitoring_items'], 1):
            story.append(Paragraph(f"{i}. {action}", normal_style))
        story.append(Spacer(1, 12))
    
    # Detailed Findings
    story.append(PageBreak())
    story.append(Paragraph("Detailed Technical Findings", heading_style))
    
    if issues:
        high_issues = [i for i in issues if i['severity'] == 'high']
        medium_issues = [i for i in issues if i['severity'] == 'medium']
        low_issues = [i for i in issues if i['severity'] == 'low']
        
        for severity, issue_list, title in [('high', high_issues, "🔴 HIGH PRIORITY ISSUES"),
                                           ('medium', medium_issues, "🟡 MEDIUM PRIORITY ISSUES"),
                                           ('low', low_issues, "🔵 LOW PRIORITY ISSUES")
