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
        'outdoorHumidity': [],  # Added OA RH
        'coolingSetpoints': [],
        'heatingSetpoints': [],
        'date': None,
        'time': None
    }
    
    for i, h in enumerate(headers):
        h_clean = str(h).strip()
        lower = h_clean.lower()
        
        # Date and Time detection
        if any(keyword in lower for keyword in ['date', 'datetime']) and mapping['date'] is None:
            mapping['date'] = i
        elif any(keyword in lower for keyword in ['time']) and mapping['time'] is None:
            mapping['time'] = i
        
        # Enhanced pressure detection
        elif any(keyword in lower for keyword in ['sucpr', 'suc pr', 'suction pr', 'suction_pr', 'suction pressure']) or \
             (('suc' in lower or 'suction' in lower) and ('pr' in lower or 'pressure' in lower)):
            mapping['suctionPressures'].append(i)
        
        elif any(keyword in lower for keyword in ['dischg', 'dis chg', 'discharge pr', 'head pr', 'headpr', 'discharge pressure']) or \
             (('discharge' in lower or 'head' in lower) and ('pr' in lower or 'pressure' in lower)):
            mapping['dischargePressures'].append(i)
        
        # Enhanced temperature detection
        elif any(keyword in lower for keyword in ['suctmp', 'suc tmp', 'suction tmp', 'suction_tmp', 'suction temp']):
            mapping['suctionTemps'].append(i)
        
        elif any(keyword in lower for keyword in ['sat ', 'supply air', 'supply_air', 'supply air temp', 'sat temp']):
            mapping['supplyAirTemps'].append(i)
        
        elif any(keyword in lower for keyword in ['discharge temp', 'dis temp', 'dischg temp']):
            mapping['dischargeTemps'].append(i)
        
        elif any(keyword in lower for keyword in ['oat', 'outdoor', 'outside', 'outdoor air']) and ('temp' in lower or 'air' in lower):
            mapping['outdoorAirTemps'].append(i)
        
        # Humidity detection (NEW)
        elif any(keyword in lower for keyword in ['oa rh', 'outdoor rh', 'outdoor humidity', 'outside humidity', 'humidity']):
            mapping['outdoorHumidity'].append(i)
        
        # Setpoint detection
        elif any(keyword in lower for keyword in ['csp', 'cool', 'cooling']) and ('sp' in lower or 'setpoint' in lower):
            mapping['coolingSetpoints'].append(i)
        
        elif any(keyword in lower for keyword in ['hsp', 'heat', 'heating']) and ('sp' in lower or 'setpoint' in lower):
            mapping['heatingSetpoints'].append(i)
    
    return mapping

def format_date_enhanced(date_str, time_str=None):
    """Enhanced date formatting that can handle date and time separately"""
    try:
        if time_str is not None:
            # Combine date and time
            combined = f"{date_str} {time_str}"
            return pd.to_datetime(combined)
        else:
            return pd.to_datetime(date_str)
    except:
        return pd.NaT

def analyze_hvac_data_enhanced(data, headers, mapping):
    """Enhanced HVAC analysis with improved detection logic"""
    issues = []
    
    # HVAC-specific analysis based on actual data patterns
    for colIdx, header in enumerate(headers):
        col_data = pd.to_numeric(data.iloc[:, colIdx], errors='coerce').dropna()
        if len(col_data) == 0:
            continue
            
        header_lower = str(header).lower()
        
        # Suction Pressure Analysis (Enhanced)
        if colIdx in mapping['suctionPressures']:
            avg_pressure = col_data.mean()
            if avg_pressure < 60:  # Low suction pressure
                issues.append({
                    "severity": "high",
                    "message": f"Low suction pressure detected in {header} (Avg: {avg_pressure:.1f} PSI)",
                    "explanation": "Low suction pressure typically indicates refrigerant undercharge, restriction in liquid line, or evaporator issues.",
                    "suggestions": ["Check for refrigerant leaks", "Verify proper refrigerant charge", "Inspect liquid line for restrictions", "Check evaporator coil condition"],
                    "issue_type": "refrigerant_system",
                    "priority": 2
                })
            elif avg_pressure > 90:  # High suction pressure
                issues.append({
                    "severity": "medium",
                    "message": f"High suction pressure detected in {header} (Avg: {avg_pressure:.1f} PSI)",
                    "explanation": "High suction pressure may indicate overcharge, compressor issues, or excessive heat load.",
                    "suggestions": ["Check refrigerant charge level", "Inspect compressor operation", "Verify cooling load calculations", "Check for non-condensables"],
                    "issue_type": "refrigerant_system",
                    "priority": 4
                })
        
        # Discharge Pressure Analysis (Enhanced)
        elif colIdx in mapping['dischargePressures']:
            avg_pressure = col_data.mean()
            if avg_pressure > 400:  # High discharge pressure
                issues.append({
                    "severity": "high", 
                    "message": f"High discharge pressure detected in {header} (Avg: {avg_pressure:.1f} PSI)",
                    "explanation": "High discharge pressure indicates condenser problems, overcharge, or airflow restrictions.",
                    "suggestions": ["Clean condenser coil", "Check condenser fan operation", "Verify proper airflow", "Check for overcharge"],
                    "issue_type": "condenser_system",
                    "priority": 1
                })
            elif avg_pressure < 150:  # Low discharge pressure
                issues.append({
                    "severity": "medium",
                    "message": f"Low discharge pressure detected in {header} (Avg: {avg_pressure:.1f} PSI)",
                    "explanation": "Low discharge pressure may indicate undercharge, compressor wear, or valve problems.",
                    "suggestions": ["Check refrigerant charge", "Test compressor valves", "Inspect for internal leaks", "Verify compressor operation"],
                    "issue_type": "compressor_system",
                    "priority": 3
                })
        
        # Enhanced Temperature Analysis
        elif colIdx in mapping['suctionTemps']:
            avg_temp = col_data.mean()
            if avg_temp > 65:  # High suction temp
                issues.append({
                    "severity": "medium",
                    "message": f"High suction temperature in {header} (Avg: {avg_temp:.1f}°F)",
                    "explanation": "High suction temperature indicates low refrigerant charge or expansion valve problems.",
                    "suggestions": ["Check superheat settings", "Verify refrigerant charge", "Inspect expansion valve", "Check for restrictions"],
                    "issue_type": "refrigerant_system",
                    "priority": 5
                })
            elif avg_temp < 35:  # Risk of freezing
                issues.append({
                    "severity": "high",
                    "message": f"Low suction temperature risk in {header} (Avg: {avg_temp:.1f}°F)",
                    "explanation": "Very low suction temperature risks liquid refrigerant returning to compressor.",
                    "suggestions": ["Check superheat immediately", "Verify proper airflow", "Inspect expansion valve", "Check for flooding"],
                    "issue_type": "refrigerant_system",
                    "priority": 1
                })
        
        elif colIdx in mapping['supplyAirTemps'] or colIdx in mapping['dischargeTemps']:
            avg_temp = col_data.mean()
            if avg_temp > 120:  # High discharge temp
                issues.append({
                    "severity": "high",
                    "message": f"High discharge temperature in {header} (Avg: {avg_temp:.1f}°F)",
                    "explanation": "High discharge temperature indicates compressor stress, poor heat rejection, or overcharge.",
                    "suggestions": ["Check condenser operation", "Verify proper airflow", "Check refrigerant charge", "Inspect compressor condition"],
                    "issue_type": "compressor_system",
                    "priority": 1
                })
            elif avg_temp < 50:  # Very low supply air temp
                issues.append({
                    "severity": "medium",
                    "message": f"Very low supply air temperature in {header} (Avg: {avg_temp:.1f}°F)",
                    "explanation": "Extremely low supply air temperature may indicate overcooling or control issues.",
                    "suggestions": ["Check thermostat settings", "Verify cooling load", "Inspect damper operation", "Check for overcooling"],
                    "issue_type": "control_system",
                    "priority": 6
                })
        
        # Humidity Analysis (NEW)
        elif colIdx in mapping['outdoorHumidity']:
            avg_humidity = col_data.mean()
            if avg_humidity > 85:  # High humidity
                issues.append({
                    "severity": "medium",
                    "message": f"High outdoor humidity detected in {header} (Avg: {avg_humidity:.1f}%)",
                    "explanation": "High outdoor humidity increases cooling load and can affect system efficiency.",
                    "suggestions": ["Monitor system performance in high humidity", "Check dehumidification capability", "Verify proper drainage", "Consider humidity control measures"],
                    "issue_type": "environmental",
                    "priority": 7
                })
            elif avg_humidity < 20:  # Very low humidity
                issues.append({
                    "severity": "low",
                    "message": f"Very low outdoor humidity in {header} (Avg: {avg_humidity:.1f}%)",
                    "explanation": "Very low humidity may indicate measurement issues or unusual weather conditions.",
                    "suggestions": ["Verify sensor calibration", "Check for sensor damage", "Compare with weather data"],
                    "issue_type": "sensor_system",
                    "priority": 10
                })
        
        # Temperature stability analysis for all temperature readings
        if colIdx in (mapping['suctionTemps'] + mapping['supplyAirTemps'] + mapping['dischargeTemps'] + mapping['outdoorAirTemps']):
            temp_range = col_data.max() - col_data.min()
            if temp_range > 25:  # High temperature variation
                issues.append({
                    "severity": "medium", 
                    "message": f"High temperature variation in {header} (Range: {temp_range:.1f}°F)",
                    "explanation": "Large temperature swings indicate cycling issues, control problems, or system instability.",
                    "suggestions": ["Check thermostat operation", "Verify control settings", "Inspect for short cycling", "Check system sizing"],
                    "issue_type": "control_system",
                    "priority": 8
                })
        
        # General outlier detection with HVAC context
        if len(col_data) > 5:  # Only analyze if we have enough data points
            q1, q3 = np.percentile(col_data, [25, 75])
            iqr = q3 - q1
            if iqr > 0:  # Avoid division by zero
                outliers = col_data[(col_data < q1 - 1.5*iqr) | (col_data > q3 + 1.5*iqr)]
                if len(outliers) > len(col_data) * 0.15:  # More than 15% outliers
                    issues.append({
                        "severity": "medium",
                        "message": f"Frequent unusual readings in {header}",
                        "explanation": "Multiple abnormal readings suggest equipment malfunction, sensor drift, or operating condition changes.",
                        "suggestions": ["Calibrate sensors", "Check equipment operation during outlier periods", "Review maintenance logs", "Monitor for patterns"],
                        "outlier_count": len(outliers),
                        "issue_type": "sensor_system",
                        "priority": 9
                    })
    
    return issues

def generate_next_visit_actions(issues):
    """Generate prioritized actions for next technician visit"""
    # Sort issues by priority
    sorted_issues = sorted(issues, key=lambda x: x.get('priority', 99))
    
    actions = []
    action_id = 1
    
    for issue in sorted_issues:
        for suggestion in issue['suggestions']:
            actions.append({
                'id': action_id,
                'action': suggestion,
                'reason': issue['message'],
                'severity': issue['severity'],
                'issue_type': issue['issue_type']
            })
            action_id += 1
    
    # Remove duplicates while preserving order
    seen = set()
    unique_actions = []
    for action in actions:
        if action['action'] not in seen:
            seen.add(action['action'])
            unique_actions.append(action)
    
    return unique_actions[:15]  # Limit to top 15 actions

def combine_dataframes(dataframes, file_names):
    """Combine multiple dataframes with source tracking"""
    combined_data = []
    
    for df, filename in zip(dataframes, file_names):
        df_copy = df.copy()
        df_copy['source_file'] = filename
        combined_data.append(df_copy)
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True, sort=False)
        return combined_df
    return None

def create_unified_plots(combined_df, combined_mapping, project_title):
    """Create unified plots from combined data"""
    
    # Create datetime column if possible
    if combined_mapping['date'] is not None:
        try:
            if combined_mapping['time'] is not None:
                combined_df['datetime'] = pd.to_datetime(
                    combined_df.iloc[:, combined_mapping['date']].astype(str) + ' ' + 
                    combined_df.iloc[:, combined_mapping['time']].astype(str),
                    errors='coerce'
                )
            else:
                combined_df['datetime'] = pd.to_datetime(combined_df.iloc[:, combined_mapping['date']], errors='coerce')
        except:
            combined_df['datetime'] = pd.to_datetime('now')
    else:
        combined_df['datetime'] = pd.date_range(start='2024-01-01', periods=len(combined_df), freq='H')
    
    # Filter out rows with invalid datetime
    plot_df = combined_df[combined_df['datetime'].notna()].copy()
    
    if len(plot_df) == 0:
        st.warning("No valid datetime data found for plotting")
        return None
    
    # Create comprehensive multi-subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Unified HVAC System Analysis - {project_title}', fontsize=16, fontweight='bold')
    
    headers = combined_df.columns.tolist()
    
    # Plot 1: System Pressures
    ax1 = axes[0, 0]
    colors_pressure = ['blue', 'navy', 'lightblue', 'darkblue']
    color_idx = 0
    
    for idx in combined_mapping['suctionPressures']:
        if idx < len(headers):
            data = pd.to_numeric(plot_df.iloc[:, idx], errors='coerce')
            ax1.plot(plot_df['datetime'], data, label=f"Suction: {headers[idx]}", 
                    color=colors_pressure[color_idx % len(colors_pressure)], linewidth=2, marker='o', markersize=1)
            color_idx += 1
    
    for idx in combined_mapping['dischargePressures']:
        if idx < len(headers):
            data = pd.to_numeric(plot_df.iloc[:, idx], errors='coerce')
            ax1.plot(plot_df['datetime'], data, label=f"Discharge: {headers[idx]}", 
                    color=colors_pressure[color_idx % len(colors_pressure)], linewidth=2, marker='s', markersize=1)
            color_idx += 1
    
    ax1.set_title('System Pressures (All Files)', fontweight='bold')
    ax1.set_ylabel('Pressure (PSI)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: System Temperatures
    ax2 = axes[0, 1]
    colors_temp = ['red', 'orange', 'darkred', 'coral']
    color_idx = 0
    
    for idx in combined_mapping['suctionTemps']:
        if idx < len(headers):
            data = pd.to_numeric(plot_df.iloc[:, idx], errors='coerce')
            ax2.plot(plot_df['datetime'], data, label=f"Suction: {headers[idx]}", 
                    color=colors_temp[color_idx % len(colors_temp)], linewidth=2, marker='o', markersize=1)
            color_idx += 1
    
    for idx in combined_mapping['supplyAirTemps']:
        if idx < len(headers):
            data = pd.to_numeric(plot_df.iloc[:, idx], errors='coerce')
            ax2.plot(plot_df['datetime'], data, label=f"Supply Air: {headers[idx]}", 
                    color=colors_temp[color_idx % len(colors_temp)], linewidth=2, marker='s', markersize=1)
            color_idx += 1
    
    for idx in combined_mapping['dischargeTemps']:
        if idx < len(headers):
            data = pd.to_numeric(plot_df.iloc[:, idx], errors='coerce')
            ax2.plot(plot_df['datetime'], data, label=f"Discharge: {headers[idx]}", 
                    color=colors_temp[color_idx % len(colors_temp)], linewidth=2, marker='^', markersize=1)
            color_idx += 1
    
    ax2.set_title('System Temperatures (All Files)', fontweight='bold')
    ax2.set_ylabel('Temperature (°F)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Outdoor Conditions
    ax3 = axes[0, 2]
    
    for idx in combined_mapping['outdoorAirTemps']:
        if idx < len(headers):
            data = pd.to_numeric(plot_df.iloc[:, idx], errors='coerce')
            ax3.plot(plot_df['datetime'], data, label=f"OA Temp: {headers[idx]}", 
                    color='green', linewidth=2, marker='d', markersize=1)
    
    # Add humidity on secondary y-axis
    if combined_mapping['outdoorHumidity']:
        ax3_humid = ax3.twinx()
        for idx in combined_mapping['outdoorHumidity']:
            if idx < len(headers):
                data = pd.to_numeric(plot_df.iloc[:, idx], errors='coerce')
                ax3_humid.plot(plot_df['datetime'], data, label=f"OA RH: {headers[idx]}", 
                              color='purple', linewidth=2, linestyle='--', marker='x', markersize=1)
        ax3_humid.set_ylabel('Humidity (%)', color='purple')
        ax3_humid.tick_params(axis='y', labelcolor='purple')
    
    ax3.set_title('Outdoor Conditions (All Files)', fontweight='bold')
    ax3.set_ylabel('Temperature (°F)', color='green')
    ax3.tick_params(axis='y', labelcolor='green')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Setpoints
    ax4 = axes[1, 0]
    
    for idx in combined_mapping['coolingSetpoints']:
        if idx < len(headers):
            data = pd.to_numeric(plot_df.iloc[:, idx], errors='coerce')
            ax4.plot(plot_df['datetime'], data, label=f"Cooling SP: {headers[idx]}", 
                    color='cyan', linewidth=2, marker='v', markersize=1)
    
    for idx in combined_mapping['heatingSetpoints']:
        if idx < len(headers):
            data = pd.to_numeric(plot_df.iloc[:, idx], errors='coerce')
            ax4.plot(plot_df['datetime'], data, label=f"Heating SP: {headers[idx]}", 
                    color='magenta', linewidth=2, marker='*', markersize=2)
    
    ax4.set_title('Temperature Setpoints (All Files)', fontweight='bold')
    ax4.set_ylabel('Temperature (°F)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # Plot 5: Data Source Distribution
    ax5 = axes[1, 1]
    if 'source_file' in plot_df.columns:
        source_counts = plot_df['source_file'].value_counts()
        ax5.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
        ax5.set_title('Data Distribution by File')
    else:
        ax5.text(0.5, 0.5, 'No Source File Data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Data Distribution by File')
    
    # Plot 6: System Overview (Key Parameters)
    ax6 = axes[1, 2]
    
    # Plot key system parameters on same axis with different colors
    if combined_mapping['suctionPressures']:
        idx = combined_mapping['suctionPressures'][0]
        if idx < len(headers):
            data = pd.to_numeric(plot_df.iloc[:, idx], errors='coerce')
            ax6.plot(plot_df['datetime'], data, label=f"Suction Pressure", 
                    color='blue', linewidth=2, alpha=0.7)
    
    if combined_mapping['supplyAirTemps']:
        idx = combined_mapping['supplyAirTemps'][0]
        if idx < len(headers):
            # Normalize temperature to pressure scale for comparison
            data = pd.to_numeric(plot_df.iloc[:, idx], errors='coerce')
            normalized_data = (data - data.min()) / (data.max() - data.min()) * 100 + 50  # Scale to 50-150 range
            ax6.plot(plot_df['datetime'], normalized_data, label=f"Supply Air Temp (Normalized)", 
                    color='red', linewidth=2, alpha=0.7)
    
    ax6.set_title('System Overview', fontweight='bold')
    ax6.set_ylabel('Values (Mixed Scale)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def generate_unified_pdf_report(project_title, logo_file, combined_issues, next_visit_actions, df_summary=None):
    """Generate a comprehensive PDF report for combined analysis"""
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
    
    # Build the PDF content
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
    story.append(Paragraph("Unified HVAC Diagnostic Analysis Report", heading_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    if combined_issues:
        high_count = len([i for i in combined_issues if i['severity'] == 'high'])
        medium_count = len([i for i in combined_issues if i['severity'] == 'medium'])
        low_count = len([i for i in combined_issues if i['severity'] == 'low'])
        
        summary_text = f"""
        This unified report analyzes HVAC system performance data from multiple sources and identifies {len(combined_issues)} total issues requiring attention:
        <br/>• {high_count} High Priority Issues (require immediate attention)
        <br/>• {medium_count} Medium Priority Issues (should be addressed soon)
        <br/>• {low_count} Low Priority Issues (monitor and plan maintenance)
        <br/><br/>This analysis combines data from multiple CSV files to provide a comprehensive system overview.
        """
        story.append(Paragraph(summary_text, normal_style))
    else:
        story.append(Paragraph("System analysis shows no immediate issues detected across all data sources. All parameters appear to be within normal operating ranges.", normal_style))
    
    story.append(Spacer(1, 20))
    
    # Next Visit Action Plan
    story.append(Paragraph("🔧 Next Technician Visit - Action Plan", heading_style))
    story.append(Paragraph("Complete the following actions in order of priority:", normal_style))
    story.append(Spacer(1, 10))
    
    if next_visit_actions:
        # Create action table
        action_data = [['Priority', 'Action Required', 'Reason', 'Severity']]
        for i, action in enumerate(next_visit_actions, 1):
            severity_icon = "🔴" if action['severity'] == 'high' else "🟡" if action['severity'] == 'medium' else "🔵"
            action_data.append([
                str(i),
                action['action'],
                action['reason'][:60] + "..." if len(action['reason']) > 60 else action['reason'],
                severity_icon
            ])
        
        action_table = Table(action_data, colWidths=[0.5*inch, 3*inch, 2.5*inch, 0.5*inch])
        action_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(action_table)
    else:
        story.append(Paragraph("No specific actions required at this time. Continue with regular maintenance schedule.", normal_style))
    
    story.append(Spacer(1, 20))
    
    # Detailed Findings
    story.append(Paragraph("Detailed Findings", heading_style))
    
    if combined_issues:
        # Group issues by severity
        high_issues = [i for i in combined_issues if i['severity'] == 'high']
        medium_issues = [i for i in combined_issues if i['severity'] == 'medium']
        low_issues = [i for i in combined_issues if i['severity'] == 'low']
        
   # High Priority Issues
        if high_issues:
            story.append(Paragraph("🔴 HIGH PRIORITY ISSUES", subheading_style))
            for i, issue in enumerate(high_issues, 1):
                story.append(Paragraph(f"<b>{i}. {issue['message']}</b>", normal_style))
                story.append(Paragraph(f"<b>Explanation:</b> {issue['explanation']}", normal_style))
                
                recommendations = "<br/>".join([f"• {rec}" for rec in issue['suggestions']])
                story.append(Paragraph(f"<b>Recommended Actions:</b><br/>{recommendations}", normal_style))
                
                if "outlier_count" in issue:
                    story.append(Paragraph(f"<b>Affected Readings:</b> {issue['outlier_count']}", normal_style))
                story.append(Spacer(1, 12))
        
        # Medium Priority Issues
        if medium_issues:
            story.append(Paragraph("🟡 MEDIUM PRIORITY ISSUES", subheading_style))
            for i, issue in enumerate(medium_issues, 1):
                story.append(Paragraph(f"<b>{i}. {issue['message']}</b>", normal_style))
                story.append(Paragraph(f"<b>Explanation:</b> {issue['explanation']}", normal_style))
                
                recommendations = "<br/>".join([f"• {rec}" for rec in issue['suggestions']])
                story.append(Paragraph(f"<b>Recommended Actions:</b><br/>{recommendations}", normal_style))
                
                if "outlier_count" in issue:
                    story.append(Paragraph(f"<b>Affected Readings:</b> {issue['outlier_count']}", normal_style))
                story.append(Spacer(1, 12))
        
        # Low Priority Issues
        if low_issues:
            story.append(Paragraph("🔵 LOW PRIORITY ISSUES", subheading_style))
            for i, issue in enumerate(low_issues, 1):
                story.append(Paragraph(f"<b>{i}. {issue['message']}</b>", normal_style))
                story.append(Paragraph(f"<b>Explanation:</b> {issue['explanation']}", normal_style))
                
                recommendations = "<br/>".join([f"• {rec}" for rec in issue['suggestions']])
                story.append(Paragraph(f"<b>Recommended Actions:</b><br/>{recommendations}", normal_style))
                
                if "outlier_count" in issue:
                    story.append(Paragraph(f"<b>Affected Readings:</b> {issue['outlier_count']}", normal_style))
                story.append(Spacer(1, 12))
    
    # Add data summary if provided
    if df_summary is not None:
        story.append(Spacer(1, 20))
        story.append(Paragraph("Data Summary Statistics", heading_style))
        
        # Create a simple table with basic stats
        try:
            numeric_df = df_summary.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                stats_data = [['Parameter', 'Mean', 'Min', 'Max', 'Std Dev']]
                for col in numeric_df.columns[:10]:  # Limit to first 10 columns
                    stats_data.append([
                        col,
                        f"{numeric_df[col].mean():.2f}",
                        f"{numeric_df[col].min():.2f}",
                        f"{numeric_df[col].max():.2f}",
                        f"{numeric_df[col].std():.2f}"
                    ])
                
                table = Table(stats_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
        except:
            story.append(Paragraph("Data summary statistics could not be generated.", normal_style))
    
    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph("Report Notes", heading_style))
    story.append(Paragraph("""
    This automated diagnostic report is based on pattern analysis of Air Carolinas HVAC system data. 
    All recommendations should be verified by qualified HVAC technicians before implementation. 
    Regular maintenance and professional inspections are essential for optimal system performance.
    """, normal_style))
    
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated by {project_title} Analysis System", normal_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- Streamlit App ---
st.set_page_config(page_title="Enhanced HVAC Data Analysis", layout="wide")

# --- Sidebar Configuration ---
st.sidebar.title("Configuration")
logo_file = st.sidebar.file_uploader("Upload Logo", type=["png", "jpg", "jpeg"])

# --- Display Logo and Title ---
if logo_file:
    st.image(logo_file, width=200)

# Title and project input
project_title = st.text_input("Enter Project Title", "HVAC Diagnostic Report")
st.title(project_title)

# --- File Upload ---
uploaded_files = st.file_uploader(
    "Upload one or more CSV or Excel files", 
    type=['csv', 'xlsx', 'xls'],
    accept_multiple_files=True
)
all_dataframes = []
file_metadata = []

def read_csv_with_encoding(file_obj):
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
    for encoding in encodings_to_try:
        try:
            file_obj.seek(0)
            content = file_obj.read().decode(encoding)
            return pd.read_csv(StringIO(content)), content
        except Exception:
            continue
    file_obj.seek(0)
    content = file_obj.read().decode('utf-8', errors='replace')
    return pd.read_csv(StringIO(content)), content

if uploaded_files:
    for uploaded_file in uploaded_files:
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
        # Combine all issues into one list
        combined_issues = []
        for _, df in file_metadata:
            headers = df.columns.tolist()
            mapping = parse_headers_enhanced(headers)  # Your existing mapping logic
            try:
                if mapping['date'] is not None and mapping['time'] is not None:
                    date_col = headers[mapping['date']]
                    time_col = headers[mapping['time']]
                    df['datetime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col])
                elif mapping['date'] is not None:
                    df['datetime'] = pd.to_datetime(df[headers[mapping['date']]])
                elif mapping['time'] is not None:
                    df['datetime'] = pd.to_datetime(df[headers[mapping['time']]])
            except Exception as e:
                st.warning(f"⚠️ Could not parse datetime: {e}")
                
            issues = analyze_hvac_data_enhanced(df, headers, mapping)
            combined_issues.extend(issues)

            # Generate summary stats from combined_df
            df_summary = combined_df.describe()

            if 'suction_pressure' in combined_df.columns:
                st.subheader("📈 Suction Pressure Trends Across All Files")
                
                chart = alt.Chart(combined_df).mark_line().encode(
                    x=alt.X('datetime:T', title='Time', axis=alt.Axis(format='%I %p')),  # 👈 Format here
                    y='suction_pressure',
                    color='source_file'
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)
            st.subheader("📊 System-Wide Data Overview (All Files Combined)")
            st.dataframe(combined_df.head())
        try:
            file_extension = uploaded_file.name.lower().split('.')[-1]
            if file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(
                    uploaded_file,
                    engine='openpyxl' if file_extension == 'xlsx' else 'xlrd'
                )
                st.success(f"✅ Excel file '{uploaded_file.name}' successfully read")
                content = None
            else:
                df, content = read_csv_with_encoding(uploaded_file)
                st.success(f"✅ CSV file '{uploaded_file.name}' successfully read")

            df['source_file'] = uploaded_file.name  # Optional: tag source
            all_dataframes.append(df)
            
            headers = df.columns.tolist()
            mapping = parse_headers_enhanced(headers)
            
            # Show detected columns for each file
            st.subheader(f"🔍 Detected Columns in {uploaded_file.name}")
            if mapping['suctionPressures']:
                st.write(f"**Suction Pressures:** {[headers[i] for i in mapping['suctionPressures']]}")
            if mapping['dischargePressures']:
                st.write(f"**Discharge Pressures:** {[headers[i] for i in mapping['dischargePressures']]}")
            if mapping['suctionTemps']:
                st.write(f"**Suction Temps:** {[headers[i] for i in mapping['suctionTemps']]}")
            if mapping['supplyAirTemps']:
                st.write(f"**Supply Air Temps:** {[headers[i] for i in mapping['supplyAirTemps']]}")
            if mapping['outdoorAirTemps']:
                st.write(f"**Outdoor Air Temps:** {[headers[i] for i in mapping['outdoorAirTemps']]}")
            if mapping['date'] is not None:
                st.write(f"**Date Column:** {headers[mapping['date']]}")
            if mapping['time'] is not None:
                st.write(f"**Time Column:** {headers[mapping['time']]}")
            
            # Analyze and display issues for each file
            issues = analyze_hvac_data_enhanced(df, headers, mapping)
            if issues:
                st.write("Detected Issues:")
                for issue in issues:
                    st.write(f"- {issue['message']}")
            else:
                st.write("No major issues detected.")

            # --- Main App Logic ---
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10))
            
            # Show basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Data Points", len(df))
            with col2:
                st.metric("Date Range", f"{len(df.index)} readings")
            with col3:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                st.metric("Numeric Parameters", len(numeric_cols))

            st.subheader("🔧 HVAC Diagnostic Analysis")
            if issues:
                # Show summary counts
                high_count = len([i for i in issues if i['severity'] == 'high'])
                medium_count = len([i for i in issues if i['severity'] == 'medium'])
                low_count = len([i for i in issues if i['severity'] == 'low'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔴 High Priority", high_count)
                with col2:
                    st.metric("🟡 Medium Priority", medium_count)  
                with col3:
                    st.metric("🔵 Low Priority", low_count)
                
                st.markdown("---")
                
                # Display issues
                for issue in issues:
                    if issue['severity'] == 'high':
                        st.error(f"🔴 **{issue['message']}**")
                    elif issue['severity'] == 'medium':
                        st.warning(f"🟡 **{issue['message']}**")
                    else:
                        st.info(f"🔵 **{issue['message']}**")
                    
                    st.markdown(f"**Why this matters:** {issue['explanation']}")
                    st.markdown("**Recommended actions:**")
                    for s in issue['suggestions']:
                        st.markdown(f"• {s}")
                    if "outlier_count" in issue:
                        st.markdown(f"**Affected readings:** {issue['outlier_count']}")
                    st.markdown("---")
            else:
                st.success("✅ No immediate HVAC issues detected in the data analysis.")
    
          # Enhanced Time-series plot with multiple subplots
            if mapping['date'] is not None:
                # Create datetime column
                if mapping['time'] is not None:
                    df['__datetime__'] = df.apply(lambda row: format_date_enhanced(row.iloc[mapping['date']], row.iloc[mapping['time']]), axis=1)
                else:
                    df['__datetime__'] = df.iloc[:, mapping['date']].apply(lambda x: format_date_enhanced(x))
                
                df_plot = df[df['__datetime__'].notna()].copy()
                
                if len(df_plot) > 0:
                    st.subheader("📈 Time-Series Analysis")
                    
                    # Create multiple plots for different parameter types
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                    
                    # Plot 1: System Pressures
                    if mapping['suctionPressures'] or mapping['dischargePressures']:
                        for idx in mapping['suctionPressures']:
                            ax1.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'), 
                                    label=f"{headers[idx]}", color='blue', linewidth=2, marker='o', markersize=2)
                        for idx in mapping['dischargePressures']:
                            ax1.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'), 
                                    label=f"{headers[idx]}", color='navy', linewidth=2, marker='s', markersize=2)
                        ax1.set_title("System Pressures", fontweight='bold')
                        ax1.set_ylabel("Pressure (PSI)", fontweight='bold')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        ax1.tick_params(axis='x', rotation=45)
                    else:
                        ax1.text(0.5, 0.5, 'No Pressure Data Available', ha='center', va='center', transform=ax1.transAxes)
                        ax1.set_title("System Pressures", fontweight='bold')
                    
                    # Plot 2: System Temperatures
                    if mapping['suctionTemps'] or mapping['supplyAirTemps'] or mapping['dischargeTemps']:
                        for idx in mapping['suctionTemps']:
                            ax2.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'), 
                                    label=f"{headers[idx]}", color='red', linewidth=2, marker='o', markersize=2)
                        for idx in mapping['supplyAirTemps']:
                            ax2.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'), 
                                    label=f"{headers[idx]}", color='orange', linewidth=2, marker='s', markersize=2)
                        for idx in mapping['dischargeTemps']:
                            ax2.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'), 
                                    label=f"{headers[idx]}", color='darkred', linewidth=2, marker='^', markersize=2)
                        ax2.set_title("System Temperatures", fontweight='bold')
                        ax2.set_ylabel("Temperature (°F)", fontweight='bold')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        ax2.tick_params(axis='x', rotation=45)
                    else:
                        # Look for any temperature columns that weren't specifically categorized
                        temp_found = False
                        for idx, header in enumerate(headers):
                            if 'temp' in header.lower():
                                ax2.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'), 
                                        label=f"{header}", color='red', linewidth=2, marker='o', markersize=2)
                                temp_found = True
                        
                        if temp_found:
                            ax2.set_title("System Temperatures", fontweight='bold')
                            ax2.set_ylabel("Temperature (°F)", fontweight='bold')
                            ax2.legend()
                            ax2.grid(True, alpha=0.3)
                            ax2.tick_params(axis='x', rotation=45)
                        else:
                            ax2.text(0.5, 0.5, 'No Temperature Data Available', ha='center', va='center', transform=ax2.transAxes)
                            ax2.set_title("System Temperatures", fontweight='bold')
                    
                    # Plot 3: Outdoor Air Temperature
                    if mapping['outdoorAirTemps']:
                        for idx in mapping['outdoorAirTemps']:
                            ax3.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'), 
                                    label=f"{headers[idx]}", color='green', linewidth=2, marker='d', markersize=2)
                        ax3.set_title("Outdoor Air Temperature", fontweight='bold')
                        ax3.set_ylabel("Temperature (°F)", fontweight='bold')
                        ax3.legend()
                        ax3.grid(True, alpha=0.3)
                        ax3.tick_params(axis='x', rotation=45)
                    else:
                        ax3.text(0.5, 0.5, 'No Outdoor Air Temperature Data', ha='center', va='center', transform=ax3.transAxes)
                        ax3.set_title("Outdoor Air Temperature", fontweight='bold')
                    
                    # Plot 4: Setpoints
                    if mapping['coolingSetpoints'] or mapping['heatingSetpoints']:
                        for idx in mapping['coolingSetpoints']:
                            ax4.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'), 
                                    label=f"{headers[idx]}", color='cyan', linewidth=2, marker='v', markersize=2)
                        for idx in mapping['heatingSetpoints']:
                            ax4.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'), 
                                    label=f"{headers[idx]}", color='magenta', linewidth=2, marker='*', markersize=3)
                        ax4.set_title("Temperature Setpoints", fontweight='bold')
                        ax4.set_ylabel("Temperature (°F)", fontweight='bold')
                        ax4.legend()
                        ax4.grid(True, alpha=0.3)
                        ax4.tick_params(axis='x', rotation=45)
                    else:
                        ax4.text(0.5, 0.5, 'No Setpoint Data Available', ha='center', va='center', transform=ax4.transAxes)
                        ax4.set_title("Temperature Setpoints", fontweight='bold')
                    
                    # Format dates on all subplots
                    for ax in [ax1, ax2, ax3, ax4]:
                        ax.tick_params(axis='x', rotation=45)
                    
                    plt.suptitle(f"HVAC System Performance Analysis - {project_title}", fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Additional simplified overview plot
                    st.subheader("📊 System Overview (Dual-Axis)")
                    fig2, ax_temp = plt.subplots(figsize=(14, 8))
                    ax_press = ax_temp.twinx()
                    
                    # Plot key temperatures on primary axis
                    temp_plotted = False
                    for idx in mapping['suctionTemps']:
                        ax_temp.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'), 
                                    label=f"{headers[idx]}", color='red', linewidth=2, marker='o', markersize=3)
                        temp_plotted = True
                    for idx in mapping['supplyAirTemps']:
                        ax_temp.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'), 
                                    label=f"{headers[idx]}", color='orange', linewidth=2, marker='s', markersize=3)
                        temp_plotted = True
                    
                    # If no specific temperature data found, look for any temperature columns
                    if not temp_plotted:
                        for idx, header in enumerate(headers):
                            if 'temp' in header.lower():
                                ax_temp.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'), 
                                            label=f"{header}", color='red', linewidth=2, marker='o', markersize=3)
                                temp_plotted = True
                    
                    # Plot pressures on secondary axis
                    for idx in mapping['suctionPressures']:
                        ax_press.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'), 
                                     label=f"{headers[idx]}", color='blue', linewidth=2, linestyle='--')
                    for idx in mapping['dischargePressures']:
                        ax_press.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'), 
                                     label=f"{headers[idx]}", color='navy', linewidth=2, linestyle='--')
                    
                    # Formatting
                    ax_temp.set_ylabel("Temperature (°F)", color='red', fontsize=12, fontweight='bold')
                    ax_press.set_ylabel("Pressure (PSI)", color='blue', fontsize=12, fontweight='bold')
                    ax_temp.set_xlabel("Date/Time", fontsize=12, fontweight='bold')
                    ax_temp.tick_params(axis='y', labelcolor='red')
                    ax_press.tick_params(axis='y', labelcolor='blue')
                    
                    # Combine legends
                    lines1, labels1 = ax_temp.get_legend_handles_labels()
                    lines2, labels2 = ax_press.get_legend_handles_labels()
                    if lines1 or lines2:
                        ax_temp.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))
                    
                    ax_temp.grid(True, alpha=0.3)
                    plt.title(f"HVAC System Performance Overview - {project_title}", fontsize=14, fontweight='bold')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig2)
                
            # Enhanced Download report as PDF
            st.subheader("📄 Generate Professional Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Generate PDF Report", type="primary"):
                    try:
                        # Generate PDF
                        pdf_buffer = generate_pdf_report(project_title, logo_file, issues, df)
                        
                        from datetime import datetime

                        report_text = generate_report(combined_df)  # Define this helper function
                        
                        st.download_button(
                            label="⬇️ Download Combined Report",
                            data=report_text,
                            file_name=f"HVAC_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain"
                        )
    
                        st.success("✅ PDF report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
                        st.info("PDF generation requires additional libraries. Falling back to text report.")
                        
                        # Fallback to text report
                        report_lines = [
                            f"{project_title}",
                            "="*len(project_title),
                            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            "",
                            "HVAC DIAGNOSTIC ANALYSIS REPORT",
                            "="*50,
                            "",
                            "SYSTEM DATA ANALYSIS FINDINGS:",
                            ""
                        ]
                        
                        if issues:
                            high_issues = [i for i in issues if i['severity'] == 'high']
                            medium_issues = [i for i in issues if i['severity'] == 'medium']
                            low_issues = [i for i in issues if i['severity'] == 'low']
                            
                            if high_issues:
                                report_lines.extend(["HIGH PRIORITY ISSUES:", "-"*20])
                                for issue in high_issues:
                                    report_lines.extend([
                                        f"ISSUE: {issue['message']}",
                                        f"EXPLANATION: {issue['explanation']}",
                                        f"RECOMMENDATIONS: {'; '.join(issue['suggestions'])}",
                                        ""
                                    ])
                            
                            if medium_issues:
                                report_lines.extend(["MEDIUM PRIORITY ISSUES:", "-"*22])
                                for issue in medium_issues:
                                    report_lines.extend([
                                        f"ISSUE: {issue['message']}",
                                        f"EXPLANATION: {issue['explanation']}",
                                        f"RECOMMENDATIONS: {'; '.join(issue['suggestions'])}",
                                        ""
                                    ])
                            
                            if low_issues:
                                report_lines.extend(["LOW PRIORITY ISSUES:", "-"*19])
                                for issue in low_issues:
                                    report_lines.extend([
                                        f"ISSUE: {issue['message']}",
                                        f"EXPLANATION: {issue['explanation']}",
                                        f"RECOMMENDATIONS: {'; '.join(issue['suggestions'])}",
                                        ""
                                    ])
                        else:
                            report_lines.append("✅ No immediate HVAC issues detected in data analysis.")
                        
                        report_lines.extend([
                            "",
                            "="*50,
                            f"Report generated by {project_title} Analysis System",
                            f"For technical support, please contact your HVAC service provider."
                        ])
                        
                        report = "\n".join(report_lines)
                        st.download_button(
                            "📄 Download Text Report (Fallback)", 
                            report, 
                            file_name=f"{project_title.replace(' ', '_')}_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain"
                        )
            
            with col2:
                st.info("📋 **PDF Report Includes:**\n- Executive Summary\n- Detailed Issue Analysis\n- Recommendations\n- Data Statistics\n- Professional Formatting")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please make sure your CSV files are properly formatted and contain valid data.")
    
else:
    st.info("👆 Please upload CSV or XLSX files to begin HVAC data analysis")
    st.markdown("### 📋 **Expected Data Format**")
    st.markdown("""
    Your CSV and XLSX files should contain columns with names that include:
    - **Date/Time** information (e.g., 'Date', 'Timestamp')
    - **Suction Pressure** data (e.g., 'Suction Pressure', 'Suction PSI')
    - **Discharge Pressure** data (e.g., 'Discharge Pressure', 'Head Pressure')
    - **Temperature** readings (e.g., 'Suction Temp', 'Supply Air Temp', 'Discharge Temp')
    
    The system will automatically detect and analyze these parameters based on column names.
    """)
