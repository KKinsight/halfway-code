import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import StringIO, BytesIO
from datetime import datetime
import base64

# Only import reportlab if available
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.warning("ReportLab not available. PDF generation will be limited to text reports.")

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
        'coolingSetpoints': [],
        'heatingSetpoints': [],
        'indoorRH': [],
        'outdoorRH': [],
        'indoorTemps': [],
        'date': None,
        'time': None,
        'datetime': None
    }
    
    for i, h in enumerate(headers):
        h_clean = str(h).strip()
        header_lower = h_clean.lower()
       
        # Date/Time detection first
        if any(keyword in header_lower for keyword in ['timestamp', 'datetime']):
            mapping['datetime'] = i
        elif any(keyword in header_lower for keyword in ['date']) and mapping['date'] is None:
            mapping['date'] = i
        elif any(keyword in header_lower for keyword in ['time']) and mapping['time'] is None:
            mapping['time'] = i
        
        # Relative Humidity detection
        elif any(keyword in header_lower for keyword in ['rel hum', 'rel. hum', 'relative humidity', 'rh']):
            if any(kw in header_lower for kw in ['oa rh', 'outdoor', 'outside', 'outside air rh']):
                mapping['outdoorRH'].append(i)
            else:
                mapping['indoorRH'].append(i)
        
        # Indoor Temperature detection
        elif any(keyword in header_lower for keyword in ['indoor temp', 'indoor temperature', 'room temp', 'spacetemp','space temp','space-temp']):
            mapping['indoorTemps'].append(i)
        
        # Enhanced temperature detection
        elif any(keyword in header_lower for keyword in ['1suctmp1','suctmp', 'suc tmp', 'suction tmp', 'suction_tmp', 'suction temp', 'suction-temp']):
            mapping['suctionTemps'].append(i)
        
        elif any(keyword in header_lower for keyword in ['sat', 'supply air', 'supply_air', 'discharge temp']):
            mapping['supplyAirTemps'].append(i)
        
        elif any(keyword in header_lower for keyword in ['dischg', 'dis chg', 'discharge']) and 'temp' in header_lower:
            mapping['dischargeTemps'].append(i)
        
        elif any(keyword in header_lower for keyword in ['oat', 'outdoor', 'outside']) and ('temp' in header_lower or 'air' in header_lower):
            mapping['outdoorAirTemps'].append(i)

                # Enhanced pressure detection
        elif any(keyword in header_lower for keyword in ['1sucpr1','suction', 'sucpr','suc pr', 'suction pr', 'suction_pr']) or \
             (('suc' in header_lower or 'suction' in header_lower) and ('pr' in header_lower or 'pressure' in header_lower)):
            mapping['suctionPressures'].append(i)
        
        elif any(keyword in header_lower for keyword in ['1dischg1','dischg', 'dis chg', 'discharge pr', 'head pr', 'headpr', '1cond1', '1headpr1']) or \
             (('discharge' in header_lower or 'head' in header_lower or 'cond' in header_lower) and ('pr' in header_lower or 'pressure' in header_lower)):
            mapping['dischargePressures'].append(i)
        
        # Setpoint detection
        elif any(keyword in header_lower for keyword in ['csp', 'cool', 'cooling']) and ('sp' in header_lower or 'setpoint' in header_lower):
            mapping['coolingSetpoints'].append(i)
        
        elif any(keyword in header_lower for keyword in ['hsp', 'heat', 'heating']) and ('sp' in header_lower or 'setpoint' in header_lower):
            mapping['heatingSetpoints'].append(i)
    
    return mapping

def create_datetime_column(df, mapping):
    """Create a datetime column from date/time or datetime columns, with support for '31-May' format"""
    try:
        if mapping['datetime'] is not None:
            df['parsed_datetime'] = pd.to_datetime(df.iloc[:, mapping['datetime']], errors='coerce')
        elif mapping['date'] is not None and mapping['time'] is not None:
            date_col = df.iloc[:, mapping['date']].astype(str).str.strip()
            time_col = df.iloc[:, mapping['time']].astype(str).str.strip()

            # Convert '31-May' to '2024-05-31'
            def convert_date(date_str):
                if pd.isna(date_str) or date_str == 'nan':
                    return None
                if '-' in date_str and len(date_str.split('-')) == 2:
                    parts = date_str.split('-')
                    if parts[0].isdigit():
                        day = parts[0]
                        month = parts[1]
                        # Convert month name to number
                        month_map = {
                            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
                            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
                            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
                        }
                        month_num = month_map.get(month.lower()[:3], month)
                        return f"2024-{month_num}-{day.zfill(2)}"
                return date_str

            date_col = date_col.apply(convert_date)
            datetime_str = date_col + ' ' + time_col
            df['parsed_datetime'] = pd.to_datetime(datetime_str, errors='coerce')
        elif mapping['date'] is not None:
            date_col = df.iloc[:, mapping['date']].astype(str).str.strip()
            date_col = date_col.apply(lambda x: convert_date(x) if callable(convert_date) else x)
            df['parsed_datetime'] = pd.to_datetime(date_col, errors='coerce')
        else:
            df['parsed_datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
        return df
    except Exception as e:
        st.warning(f"Could not parse datetime: {e}. Using sequential index.")
        df['parsed_datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
        return df

def check_comfort_conditions(df, headers, mapping):
    """Check indoor comfort conditions"""
    results = []
    
    # Check relative humidity
    for idx in mapping.get('indoorRH', []):
        if headers[idx] == '1SprHtSP':
            continue
        humidity_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(humidity_data) > 0:
            above_60 = (humidity_data > 60).sum()
            percent_over = (above_60 / len(humidity_data)) * 100
            avg_humidity = humidity_data.mean()
            results.append({
                'type': 'Indoor Relative Humidity',
                'column': headers[idx],
                'average': avg_humidity,
                'percent_over': percent_over,
                'compliant': percent_over == 0
            })

    # Check indoor temperature
    for idx in mapping.get('indoorTemps', []):
        temp_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(temp_data) > 0:
            below_70 = (temp_data < 70).sum()
            above_75 = (temp_data > 75).sum()
            percent_outside = ((below_70 + above_75) / len(temp_data)) * 100
            avg_temp = temp_data.mean()
            results.append({
                'type': 'Indoor Temperature',
                'column': headers[idx],
                'average': avg_temp,
                'percent_outside': percent_outside,
                'compliant': percent_outside == 0
            })
    
    return results

def analyze_hvac_data_enhanced(df, headers, mapping):
    """Enhanced HVAC analysis with improved detection logic"""
    issues = []
    
    # Check suction pressures
    for idx in mapping['suctionPressures']:
        col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(col_data) > 0:
            avg_pressure = col_data.mean()
            if avg_pressure > 200:
                issues.append({
                    'message': f'High suction pressure detected in {headers[idx]} (Avg: {avg_pressure:.1f} PSI)',
                    'severity': 'high',
                    'explanation': 'High suction pressure may indicate system overcharge or restricted airflow',
                    'suggestions': ['Check refrigerant levels', 'Inspect air filters', 'Verify ductwork'],
                    'issue_type': 'refrigerant_system'
                })
            elif avg_pressure < 50:
                issues.append({
                    'message': f'Low suction pressure detected in {headers[idx]} (Avg: {avg_pressure:.1f} PSI)',
                    'severity': 'medium',
                    'explanation': 'Low suction pressure may indicate refrigerant leak or expansion valve issues',
                    'suggestions': ['Check for refrigerant leaks', 'Inspect expansion valve', 'Verify system charge'],
                    'issue_type': 'refrigerant_system'
                })
    
    # Check discharge pressures
    for idx in mapping['dischargePressures']:
        col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(col_data) > 0:
            avg_pressure = col_data.mean()
            if avg_pressure > 400:
                issues.append({
                    "severity": "high", 
                    "message": f"High discharge pressure detected in {headers[idx]} (Avg: {avg_pressure:.1f} PSI)",
                    "explanation": "High discharge pressure indicates condenser problems, overcharge, or airflow restrictions.",
                    "suggestions": ["Clean condenser coil", "Check condenser fan operation", "Verify proper airflow", "Check for overcharge"],
                    "issue_type": "condenser_system"
                })
            elif avg_pressure < 150:
                issues.append({
                    "severity": "medium",
                    "message": f"Low discharge pressure detected in {headers[idx]} (Avg: {avg_pressure:.1f} PSI)",
                    "explanation": "Low discharge pressure may indicate undercharge, compressor wear, or valve problems.",
                    "suggestions": ["Check refrigerant charge", "Test compressor valves", "Inspect for internal leaks", "Verify compressor operation"],
                    "issue_type": "compressor_system"
                })
    
    # Check suction temperatures
    for idx in mapping['suctionTemps']:
        col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(col_data) > 0:
            avg_temp = col_data.mean()
            if avg_temp > 65:
                issues.append({
                    "severity": "medium",
                    "message": f"High suction temperature in {headers[idx]} (Avg: {avg_temp:.1f}°F)",
                    "explanation": "High suction temperature indicates low refrigerant charge or expansion valve problems.",
                    "suggestions": ["Check superheat settings", "Verify refrigerant charge", "Inspect expansion valve", "Check for restrictions"],
                    "issue_type": "refrigerant_system"
                })
            elif avg_temp < 35:
                issues.append({
                    "severity": "high",
                    "message": f"Low suction temperature risk in {headers[idx]} (Avg: {avg_temp:.1f}°F)",
                    "explanation": "Very low suction temperature risks liquid refrigerant returning to compressor.",
                    "suggestions": ["Check superheat immediately", "Verify proper airflow", "Inspect expansion valve", "Check for flooding"],
                    "issue_type": "refrigerant_system"
                })
    
    # Check supply air and discharge temperatures
    for idx in mapping['supplyAirTemps'] + mapping['dischargeTemps']:
        col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(col_data) > 0:
            avg_temp = col_data.mean()
            if avg_temp > 120:
                issues.append({
                    "severity": "high",
                    "message": f"High discharge temperature in {headers[idx]} (Avg: {avg_temp:.1f}°F)",
                    "explanation": "High discharge temperature indicates compressor stress, poor heat rejection, or overcharge.",
                    "suggestions": ["Check condenser operation", "Verify proper airflow", "Check refrigerant charge", "Inspect compressor condition"],
                    "issue_type": "compressor_system"
                })
            elif avg_temp < 50:
                issues.append({
                    "severity": "medium",
                    "message": f"Very low supply air temperature in {headers[idx]} (Avg: {avg_temp:.1f}°F)",
                    "explanation": "Extremely low supply air temperature may indicate overcooling or control issues.",
                    "suggestions": ["Check thermostat settings", "Verify cooling load", "Inspect damper operation", "Check for overcooling"],
                    "issue_type": "control_system"
                })
    
    return issues

def generate_pdf_report(project_title, logo_file, issues, df_summary=None):
    """Generate a comprehensive PDF report"""
    if not REPORTLAB_AVAILABLE:
        return None
        
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
    story.append(Paragraph("HVAC Diagnostic Analysis Report", heading_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    if issues:
        high_count = len([i for i in issues if i['severity'] == 'high'])
        medium_count = len([i for i in issues if i['severity'] == 'medium'])
        low_count = len([i for i in issues if i['severity'] == 'low'])
        
        summary_text = f"""
        This report analyzes HVAC system performance data and identifies {len(issues)} total issues requiring attention:
        <br/>• {high_count} High Priority Issues (require immediate attention)
        <br/>• {medium_count} Medium Priority Issues (should be addressed soon)
        <br/>• {low_count} Low Priority Issues (monitor and plan maintenance)
        """
        story.append(Paragraph(summary_text, normal_style))
    else:
        story.append(Paragraph("System analysis shows no immediate issues detected. All parameters appear to be within normal operating ranges.", normal_style))
    
    story.append(Spacer(1, 20))
    
    # Detailed Findings
    story.append(Paragraph("Detailed Findings", heading_style))
    
    if issues:
        # Group issues by severity
        high_issues = [i for i in issues if i['severity'] == 'high']
        medium_issues = [i for i in issues if i['severity'] == 'medium']
        low_issues = [i for i in issues if i['severity'] == 'low']
        
        # High Priority Issues
        if high_issues:
            story.append(Paragraph("🔴 HIGH PRIORITY ISSUES", subheading_style))
            for i, issue in enumerate(high_issues, 1):
                story.append(Paragraph(f"<b>{i}. {issue['message']}</b>", normal_style))
                story.append(Paragraph(f"<b>Explanation:</b> {issue['explanation']}", normal_style))
                
                recommendations = "<br/>".join([f"• {rec}" for rec in issue['suggestions']])
                story.append(Paragraph(f"<b>Recommended Actions:</b><br/>{recommendations}", normal_style))
                story.append(Spacer(1, 12))
        
        # Medium Priority Issues
        if medium_issues:
            story.append(Paragraph("🟡 MEDIUM PRIORITY ISSUES", subheading_style))
            for i, issue in enumerate(medium_issues, 1):
                story.append(Paragraph(f"<b>{i}. {issue['message']}</b>", normal_style))
                story.append(Paragraph(f"<b>Explanation:</b> {issue['explanation']}", normal_style))
                
                recommendations = "<br/>".join([f"• {rec}" for rec in issue['suggestions']])
                story.append(Paragraph(f"<b>Recommended Actions:</b><br/>{recommendations}", normal_style))
                story.append(Spacer(1, 12))
        
        # Low Priority Issues
        if low_issues:
            story.append(Paragraph("🔵 LOW PRIORITY ISSUES", subheading_style))
            for i, issue in enumerate(low_issues, 1):
                story.append(Paragraph(f"<b>{i}. {issue['message']}</b>", normal_style))
                story.append(Paragraph(f"<b>Explanation:</b> {issue['explanation']}", normal_style))
                
                recommendations = "<br/>".join([f"• {rec}" for rec in issue['suggestions']])
                story.append(Paragraph(f"<b>Recommended Actions:</b><br/>{recommendations}", normal_style))
                story.append(Spacer(1, 12))
    
    # Add data summary if provided
    if df_summary is not None:
        story.append(Spacer(1, 20))
        story.append(Paragraph("Data Summary Statistics", heading_style))
        
        try:
            numeric_df = df_summary.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                stats_data = [['Parameter', 'Mean', 'Min', 'Max', 'Std Dev']]
                for col in numeric_df.columns[:10]:
                    stats_data.append([
                        col,
                        f"{numeric_df[col].mean():.2f}",
                        f"{numeric_df[col].min():.2f}",
                        f"{numeric_df[col].max():.2f}",
                        f"{numeric_df[col].std():.2f}"
                    ])

            numeric_df = df_summary.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                stats_data = [['Parameter', 'Mean', 'Min', 'Max', 'Std Dev']]
                for col in numeric_df.columns[:10]:
                    mean = numeric_df[col].mean()
                    min_val = numeric_df[col].min()
                    max_val = numeric_df[col].max()
                    std_dev = numeric_df[col].std()
                    if not (mean == 0 and min_val == 0 and max_val == 0 and std_dev == 0):
                        stats_data.append([
                            col,
                                f"{mean:.2f}",
                                f"{min_val:.2f}",
                                f"{max_val:.2f}",
                                f"{std_dev:.2f}"
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
    This automated diagnostic report is based on pattern analysis of HVAC system data. 
    All recommendations should be verified by qualified HVAC technicians before implementation. 
    Regular maintenance and professional inspections are essential for optimal system performance.
    """, normal_style))
    
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated by {project_title} Analysis System", normal_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def read_csv_with_encoding(uploaded_file):
    """Read CSV with proper encoding handling"""
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings_to_try:
        try:
            uploaded_file.seek(0)
            content = uploaded_file.read().decode(encoding)
            df = pd.read_csv(StringIO(content))
            return df, content
        except Exception as e:
            continue
    
    # If all encodings fail, try with error handling
    uploaded_file.seek(0)
    content = uploaded_file.read().decode('utf-8', errors='replace')
    df = pd.read_csv(StringIO(content))
    return df, content

def get_legend_label(header):
    """Map short header names to descriptive labels for plotting."""
    header_lower = str(header).strip().lower()
    if 'sat' == header_lower:
        return 'Supply Air Temp'
    elif 'oat' == header_lower:
        return 'Outdoor Air Temp'
    elif 'oa rh' in header_lower or 'oa_rh' in header_lower:
        return 'Outside Air Relative Humidity'
    elif '1sucpr1' in header_lower:
        return 'Suction Pressure 1'
    elif '1dischg1' in header_lower:
        return 'Discharge Pressure'
    elif '1suctmp1' in header_lower:
        return 'Suction Temp 1'
    elif '1headpr1' in header_lower:
        return 'Head Pressure 1'
    elif '1cond1' in header_lower:
        return 'Condenser Pressure 1'
    else:
        return header  # Fallback to original header

def create_time_series_plots(df, headers, mapping):
    """Create temperature vs time and pressure vs time plots"""
    plots = []
    
    # Temperature vs Time Plot
    temp_indices = (mapping['suctionTemps'] + mapping['supplyAirTemps'] + 
                   mapping['dischargeTemps'] + mapping['outdoorAirTemps'] + 
                   mapping['indoorTemps'])
    
    if temp_indices and 'parsed_datetime' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for idx_num, idx in enumerate(temp_indices[:6]):  # Limit to 6 columns for readability
            temp_data = pd.to_numeric(df.iloc[:, idx], errors='coerce')
            valid_mask = ~temp_data.isna() & ~df['parsed_datetime'].isna()
            if valid_mask.sum() > 0:
                ax.plot(df.loc[valid_mask, 'parsed_datetime'], 
                       temp_data[valid_mask], 
                       label=get_legend_label(headers[idx]),
                       marker='o', 
                       markersize=2,
                       linewidth=1,
                       color=colors[idx_num % len(colors)])
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature (°F)')
        ax.set_title('Temperature vs Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        if len(df) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df)//10)))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots.append(('Temperature vs Time', fig))
    
    # Pressure vs Time Plot
    pressure_indices = mapping['suctionPressures'] + mapping['dischargePressures']
    if pressure_indices and 'parsed_datetime' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['darkblue', 'darkred', 'darkgreen', 'darkorange', 'purple', 'brown']
        
        for idx_num, idx in enumerate(pressure_indices[:6]):
            pressure_data = pd.to_numeric(df.iloc[:, idx], errors='coerce')
            valid_mask = ~pressure_data.isna() & ~df['parsed_datetime'].isna()
            if valid_mask.sum() > 0:
                ax.plot(df.loc[valid_mask, 'parsed_datetime'], 
                       pressure_data[valid_mask], 
                       label=get_legend_label(headers[idx]),
                       marker='o', 
                       markersize=2,
                       linewidth=1,
                       color=colors[idx_num % len(colors)])
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Pressure (PSI)')
        ax.set_title('Pressure vs Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        if len(df) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df)//10)))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots.append(('Pressure vs Time', fig))

    # Relative Humidity vs Time Plot
    indoor_rh_indices = mapping.get('indoorRH', [])
    outdoor_rh_indices = mapping.get('outdoorRH', [])
    colors = ['teal', 'magenta', 'olive', 'coral', 'gray', 'gold']
    
    if (indoor_rh_indices or outdoor_rh_indices) and 'parsed_datetime' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
    
        # Indoor RH
        for idx_num, idx in enumerate(indoor_rh_indices[:3]):
            rh_data = pd.to_numeric(df.iloc[:, idx], errors='coerce')
            label = get_legend_label(headers[idx]) + " (Indoor)"
            valid_mask = ~rh_data.isna() & ~df['parsed_datetime'].isna()
            ax.plot(df.loc[valid_mask, 'parsed_datetime'],
                    rh_data[valid_mask],
                    label=label,
                    color=colors[idx_num % len(colors)],
                    marker='o', linewidth=1, markersize=2)
    
        # Outdoor RH
        for idx_num, idx in enumerate(outdoor_rh_indices[:3]):
            rh_data = pd.to_numeric(df.iloc[:, idx], errors='coerce')
            label = get_legend_label(headers[idx]) + " (Outdoor)"
            valid_mask = ~rh_data.isna() & ~df['parsed_datetime'].isna()
            ax.plot(df.loc[valid_mask, 'parsed_datetime'],
                    rh_data[valid_mask],
                    label=label,
                    linestyle='--',
                    color=colors[(idx_num + 3) % len(colors)],
                    marker='x', linewidth=1, markersize=2)
    
        ax.set_xlabel("Time")
        ax.set_ylabel("Relative Humidity (%)")
        ax.set_title("Relative Humidity (Indoor vs Outdoor) vs Time")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df) // 10)))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots.append(('Relative Humidity vs Time', fig))
    
    return plots
    
# --- Streamlit App ---
st.set_page_config(page_title="Enhanced HVAC Data Analysis", layout="wide")

# --- Sidebar Configuration ---
st.sidebar.title("Configuration")
logo_file = st.sidebar.file_uploader("Upload Logo (Optional)", type=["png", "jpg", "jpeg"])

# --- Display Logo and Title ---
if logo_file:
    st.image(logo_file, width=200)

# Title and project input
project_title = st.text_input("Enter Project Title", "HVAC Diagnostic Report")
st.title(project_title)

# --- Single File Upload Section ---
st.markdown("## 📁 Upload HVAC Data Files")
uploaded_files = st.file_uploader(
    "Upload one or more CSV or Excel files containing HVAC data",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True
)

if uploaded_files:
    all_dataframes = []
    all_issues = []
    all_file_info = []
    
    # Process each file
    for uploaded_file in uploaded_files:
        try:
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            if file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
                st.success(f"✅ Excel file '{uploaded_file.name}' successfully read with {len(df)} rows")
            else:
                # Read and clean CSV by skipping the second row (units like °F)
                try:
                    uploaded_file.seek(0)
                    lines = uploaded_file.read().decode('latin-1').splitlines()
                    if len(lines) > 1:
                        lines.pop(1)
                    cleaned_csv = "\n".join(lines)
                    df = pd.read_csv(StringIO(cleaned_csv))
                    st.success(f"✅ Cleaned CSV file '{uploaded_file.name}' successfully read with {len(df)} rows")
                except Exception as e:
                    st.error(f"Failed to read and clean '{uploaded_file.name}': {e}")
                    continue
            
            # Clean the data - skip rows that are all NaN or contain header-like content
            df = df.dropna(how='all')  # Remove completely empty rows
            
            # If the first row contains units (like °F, PSI, etc.), remove it
            if len(df) > 0 and df.iloc[0].astype(str).str.contains('°F|PSI|%|WG', case=False, na=False).any():
                df = df.iloc[1:].reset_index(drop=True)
                st.info(f"Removed units row from {uploaded_file.name}")
            
            # Add source file identifier
            df['source_file'] = uploaded_file.name
            all_dataframes.append(df.copy())
            all_file_info.append({'name': uploaded_file.name, 'df': df})
            
            # Analyze each file
            headers = df.columns.tolist()
            mapping = parse_headers_enhanced(headers)

          # Analyze each file
            headers = df.columns.tolist()
            mapping = parse_headers_enhanced(headers)
            
            # Create datetime column
            df = create_datetime_column(df, mapping)
            
            # Show detected columns
            st.subheader(f"🔍 Detected Columns in {uploaded_file.name}")
            col1, col2 = st.columns(2)
            
            with col1:
                if mapping['suctionPressures']:
                    st.write(f"**Suction Pressures:** {[headers[i] for i in mapping['suctionPressures']]}")
                if mapping['dischargePressures']:
                    st.write(f"**Discharge Pressures:** {[headers[i] for i in mapping['dischargePressures']]}")
                if mapping['suctionTemps']:
                    st.write(f"**Suction Temps:** {[headers[i] for i in mapping['suctionTemps']]}")
            
            with col2:
                if mapping['supplyAirTemps']:
                    st.write(f"**Supply Air Temps:** {[headers[i] for i in mapping['supplyAirTemps']]}")
                if mapping['outdoorAirTemps']:
                    st.write(f"**Outdoor Air Temps:** {[headers[i] for i in mapping['outdoorAirTemps']]}")
                if mapping['indoorTemps']:
                    st.write(f"**Indoor Temps:** {[headers[i] for i in mapping['indoorTemps']]}")
            
            # Analyze issues for this file
            issues = analyze_hvac_data_enhanced(df, headers, mapping)
            all_issues.extend(issues)

        except Exception as e:
            st.error(f"File {uploaded_file.name} could not be processed: {e}")

    # Combine all dataframes for unified analysis
    if len(all_dataframes) == 1:
        combined_df = all_dataframes[0]
        combined_headers = list(combined_df.columns)
    elif len(all_dataframes) > 1:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_headers = list(combined_df.columns)
    else:
        combined_df = None
        combined_headers = []

    # Show summary statistics on the main page
    if combined_df is not None:
        st.markdown("## 📊 Data Summary Statistics")
        numeric_df = combined_df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            summary_stats = numeric_df.describe().T[['mean', 'min', 'max', 'std']]
            summary_stats.columns = ['Mean', 'Min', 'Max', 'Std Dev']
            summary_stats = summary_stats[~(summary_stats == 0).all(axis=1)]    # Exclude all-zero stats
            st.dataframe(summary_stats.style.format("{:.2f}"))
        else:
            st.info("No numeric data available for summary statistics.")
    
    # Unified Indoor Comfort Check
    if combined_df is not None:
        combined_mapping = parse_headers_enhanced(combined_headers)
        combined_df = create_datetime_column(combined_df, combined_mapping)
        comfort_results = check_comfort_conditions(combined_df, combined_headers, combined_mapping)
    
        if comfort_results:
            st.markdown("## 🏠 Indoor Comfort Check")
            for result in comfort_results:
                if result["type"] == "Indoor Relative Humidity":
                    msg = ('✅ Within ideal range (≤60%)' if result['compliant'] 
                        else f'⚠️ {result["percent_over"]:.1f}% of values above 60%')
                    st.write(f"**{result['column']}** (Avg: {result['average']:.1f}%) - {msg}")
                elif result["type"] == "Indoor Temperature":
                    msg = ('✅ Within ideal range (70-75°F)' if result['compliant']              
                        else f"⚠️ {result['percent_outside']:.1f}% of values outside 70-75°F range")
                    st.write(f"**{result['column']}** (Avg: {result['average']:.1f}°F) - {msg}")

    # Ensure parsed_datetime exists in combined_df
    if combined_df is not None and 'parsed_datetime' not in combined_df.columns:
        combined_mapping = parse_headers_enhanced(combined_headers)
        combined_df = create_datetime_column(combined_df, combined_mapping)
    
    # Add this block to define combined_mapping
    if combined_df is not None:
        combined_mapping = parse_headers_enhanced(combined_headers)
    else:
        combined_mapping = {}

    # Single set of time series plots using combined data
    st.markdown("## 📈 Time Series Analysis")
    combined_plots = create_time_series_plots(combined_df, combined_headers, combined_mapping)
    for plot_title, fig in combined_plots:
        st.pyplot(fig)
        plt.close(fig)  # Close figure to free memory

    # Single unified analysis results
    if all_issues:
        # Show summary counts
        high_count = len([i for i in all_issues if i['severity'] == 'high'])
        medium_count = len([i for i in all_issues if i['severity'] == 'medium'])
        low_count = len([i for i in all_issues if i['severity'] == 'low'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 High Priority", high_count)
        with col2:
            st.metric("🟡 Medium Priority", medium_count)
        with col3:
            st.metric("🔵 Low Priority", low_count)

        # Display all issues grouped by severity
        if high_count > 0:
            st.markdown("### 🔴 High Priority Issues")
            for issue in [i for i in all_issues if i['severity'] == 'high']:
                st.error(f"**{issue['message']}**")
                st.markdown(f"**Why this matters:** {issue['explanation']}")
                st.markdown("**Recommended actions:**")
                for suggestion in issue['suggestions']:
                    st.markdown(f"• {suggestion}")
                st.markdown("---")

        if medium_count > 0:
            st.markdown("### 🟡 Medium Priority Issues")
            for issue in [i for i in all_issues if i['severity'] == 'medium']:
                st.warning(f"**{issue['message']}**")
                st.markdown(f"**Why this matters:** {issue['explanation']}")
                st.markdown("**Recommended actions:**")
                for suggestion in issue['suggestions']:
                    st.markdown(f"• {suggestion}")
                st.markdown("---")

        if low_count > 0:
            st.markdown("### 🔵 Low Priority Issues")
            for issue in [i for i in all_issues if i['severity'] == 'low']:
                st.info(f"**{issue['message']}**")
                st.markdown(f"**Why this matters:** {issue['explanation']}")
                st.markdown("**Recommended actions:**")
                for suggestion in issue['suggestions']:
                    st.markdown(f"• {suggestion}")
                st.markdown("---")
    else:
        st.success("✅ No immediate HVAC issues detected in the combined data analysis.")

    # Single PDF Report Generation
    st.markdown("## 📄 Generate Unified Report")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Generate PDF Report", type="primary"):
            try:
                pdf_buffer = generate_pdf_report(
                    project_title=project_title,
                    logo_file=logo_file,
                    issues=all_issues,
                    df_summary=combined_df
                )
                
                if pdf_buffer:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"{project_title.replace(' ', '_')}_combined_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf"
                    )
                else:
                    raise Exception("PDF generation failed")
                    
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                st.info("PDF generation requires additional libraries. Falling back to text report.")
                
                # Fallback to text report
                report_lines = [
                    f"{project_title} - Project File Analysis",
                    "=" * (len(project_title) + 20),
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Files Analyzed: {', '.join([info['name'] for info in all_file_info])}",
                    f"Total Data Points: {len(combined_df)}",
                    "",
                    "HVAC DIAGNOSTIC ANALYSIS REPORT",
                    "=" * 50,
                    "",
                    "UNIFIED SYSTEM DATA ANALYSIS FINDINGS:",
                    ""
                ]
                
                if all_issues:
                    high_issues = [i for i in all_issues if i.get('severity') == 'high']
                    medium_issues = [i for i in all_issues if i.get('severity') == 'medium']
                    low_issues = [i for i in all_issues if i.get('severity') == 'low']
                    
                    if high_issues:
                        report_lines.extend(["HIGH PRIORITY ISSUES:", "-" * 20])
                        for issue in high_issues:
                            report_lines.extend([
                                f"ISSUE: {issue['message']}",
                                f"EXPLANATION: {issue['explanation']}",
                                f"RECOMMENDATIONS: {'; '.join(issue['suggestions'])}",
                                ""
                            ])
                    
                    if medium_issues:
                        report_lines.extend(["MEDIUM PRIORITY ISSUES:", "-" * 22])
                        for issue in medium_issues:
                            report_lines.extend([
                                f"ISSUE: {issue['message']}",
                                f"EXPLANATION: {issue['explanation']}",
                                f"RECOMMENDATIONS: {'; '.join(issue['suggestions'])}",
                                ""
                            ])
                    
                    if low_issues:
                        report_lines.extend(["LOW PRIORITY ISSUES:", "-" * 19])
                        for issue in low_issues:
                            report_lines.extend([
                                f"ISSUE: {issue['message']}",
                                f"EXPLANATION: {issue['explanation']}",
                                f"RECOMMENDATIONS: {'; '.join(issue['suggestions'])}",
                                ""
                            ])
                else:
                    report_lines.append("✅ No immediate HVAC issues detected in combined data analysis.")
                
                report_lines.extend([
                    "",
                    "DATA SOURCES:",
                    "-" * 13
                ])
                
                for info in all_file_info:
                    report_lines.append(f"• {info['name']} ({len(info['df'])} data points)")
                
                report_lines.extend([
                    "",
                    "=" * 50,
                    f"Report generated by {project_title} Analysis System",
                    "For technical support, please contact your HVAC service provider."
                ])
                
                report = "\n".join(report_lines)
                
                st.download_button(
                    "📄 Download Text Report",
                    report,
                    file_name=f"{project_title.replace(' ', '_')}_combined_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
    
    with col2:
        st.info(
            "📋 **PDF Report Includes:**\n"
            "- Executive Summary for All Data\n"
            "- Unified Issue Analysis\n"
            "- Consolidated Recommendations\n"
            "- Data Statistics\n"
            "- Source File Information\n"
            "- Professional Formatting"
        )

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

st.markdown("---")
st.markdown("*Enhanced HVAC Data Analysis System - Professional diagnostic reporting for HVAC systems*")
