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

def filter_meaningful_columns_strict(df, zero_threshold=0.95):
    """
    Filter out columns that are empty, contain only zeros, mostly zeros, or only meaningless values
    zero_threshold: if more than this percentage of values are zero, exclude the column
    """
    meaningful_columns = []

    for col in df.columns:
        if col in df.columns:
            # Convert column to numeric, coercing errors to NaN
            numeric_col = pd.to_numeric(df[col], errors='coerce')

            # Check if column has any meaningful data
            if not numeric_col.isna().all():  # Not all NaN
                # Remove NaN values for analysis
                clean_data = numeric_col.dropna()

                if len(clean_data) > 0:
                    # Check if ALL values are zero
                    if (clean_data == 0).all():
                        continue  # Skip columns with all zeros

                    # NEW: Check if mostly zeros (optional stricter filtering)
                    zero_percentage = (clean_data == 0).sum() / len(clean_data)
                    if zero_percentage > zero_threshold:
                        continue  # Skip columns that are mostly zeros

                    # Check if there's actual variation
                    if clean_data.nunique() > 1:
                        meaningful_columns.append(col)
                    elif clean_data.iloc[0] != 0:
                        meaningful_columns.append(col)
                else:

                    # Check for text columns that might contain meaningful non-numeric data
                    text_col = df[col].astype(str).str.lower().str.strip()
                    unique_values = text_col.unique()

                    # Filter out common empty/meaningless values
                    meaningless_values = {'nan', 'none', 'null', '', '0', 'o', 'na', 'n/a'}
                    meaningful_values = [val for val in unique_values if val not in meaningless_values]

                    if len(meaningful_values) > 0:
                        meaningful_columns.append(col)

    return meaningful_columns

def generate_enhanced_data_summary(df_summary):
    """Generate data summary with filtered meaningful columns"""
    if df_summary is None or df_summary.empty:
        return [['No data available for analysis']]

    try:
        # Filter for meaningful columns
        meaningful_cols = filter_meaningful_columns_strict(df_summary)
        meaningful_cols = [col for col in meaningful_cols if col != 'parsed_datetime']
        if not meaningful_cols:
            return [['No meaningful data columns found']]

        # Create dataframe with only meaningful columns
        meaningful_df = df_summary[meaningful_cols].copy()

        # Convert meaningful columns to numeric where possible
        numeric_data = {}
        for col in meaningful_cols:
            numeric_series = pd.to_numeric(meaningful_df[col], errors='coerce')
            if not numeric_series.isna().all():
                clean_data = numeric_series.dropna()
                if len(clean_data) > 0:
                    if clean_data.abs().max() > 0.1 and clean_data.nunique() > 1:
                        numeric_data[col] = clean_data

        if not numeric_data:
            return [['No meaningful numeric data available for statistical analysis']]

        # Generate statistics table
        stats_data = [['Parameter', 'Mean', 'Min', 'Max', 'Std Dev']]

        def format_value(val): 
            if pd.isna(val):
                return "None"
            if abs(val) >= 100:
                return f"{val:.1f}"
            elif abs(val) >= 1:
                return f"{val:.2f}"
            else:
                return f"{val:.3f}"

        for col in sorted(numeric_data.keys()):
            clean_data = numeric_data[col]
            if clean_data.empty:
                continue

            mean_val = clean_data.mean()
            min_val = clean_data.min()
            max_val = clean_data.max()
            std_val = clean_data.std()

            # Skip rows where all stats are NaN or None
            if all(pd.isna(val) for val in [mean_val, min_val, max_val, std_val]):
                continue

            stats_data.append([
                col[:30],
                format_value(mean_val),
                format_value(min_val),
                format_value(max_val),
                format_value(std_val),
            ])

        return stats_data if len(stats_data) > 1 else [['No meaningful statistics available after filtering']]

    except Exception as e:
        print(f"Error in generate_enhanced_data_summary: {str(e)}")
        return [['Error generating statistics:', str(e)]]

# Updated integration code for the PDF report function
def integrate_into_pdf_report():
    """
    Replace the existing data summary section in your generate_pdf_report function
    with this code block:
    """
    return '''
    # Add data summary if provided
    if df_summary is not None:
        story.append(Spacer(1, 20))
        story.append(Paragraph("Data Summary Statistics", heading_style))
        story.append(Spacer(1, 10))
        
        stats_data = generate_enhanced_data_summary(df_summary)
        
        if stats_data and len(stats_data) > 1:
            # Adjust column widths for better formatting
            table = Table(stats_data, colWidths=[2.8*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.7*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.lightgrey])
            ]))
            story.append(table)
            
            # Add a note about filtering
            story.append(Spacer(1, 10))
            story.append(Paragraph(
                "<i>Note: Only columns with meaningful variation (absolute values > 0.1) are included in this summary.</i>", 
                normal_style
            ))
        else:
            story.append(Paragraph(
                "No meaningful data available for statistical analysis. All columns appear to contain only zeros, empty values, or insignificant variations.", 
                normal_style
            ))
    '''

# Test function to verify filtering works correctly
def test_filtering_with_sample_data():
    """Test the filtering function with sample data similar to your CSV"""
    import pandas as pd
    import numpy as np

    # Create sample data similar to your CSV structure
    sample_data = {
        'Date': ['31-May', '31-May', '31-May'],
        'Time': ['1:46', '2:01', '2:16'],
        'Space': [70, 69.9, 69.7],  # Meaningful data
        'InRH': [52, 52, 51],       # Meaningful data
        'ZeroCol': [0, 0, 0],       # Should be filtered out
        'EmptyCol': ['', '', ''],   # Should be filtered out
        'NoneCol': [None, None, None],  # Should be filtered out
        'OCol': ['O', 'O', 'O'],    # Should be filtered out
        'TinyVarCol': [0.01, -0.01, 0.02],  # Should be filtered out (too small)
        'MeaningfulTemp': [75.3, 76.1, 74.9]  # Should be kept
    }

    df_test = pd.DataFrame(sample_data)
    meaningful_cols = filter_meaningful_columns(df_test)

    print("Test Results:")
    print(f"All columns: {list(df_test.columns)}")
    print(f"Meaningful columns: {meaningful_cols}")
    print(f"Filtered out: {set(df_test.columns) - set(meaningful_cols)}")

    # Test the summary generation
    summary_result = generate_enhanced_data_summary(df_test)
    print("\nSummary table:")
    for row in summary_result:
        print(row)

def check_comfort_conditions(df, headers, mapping):
    """Check indoor comfort conditions"""
    results = []

    # Check relative humidity
    for idx in mapping.get('indoorRH', []):
        col_name = headers[idx].lower()
        if 'sprheat' in col_name or 'sprhtsp' in col_name:
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
    """Enhanced HVAC analysis with 50+ comprehensive diagnostic checks"""
    issues = []

    # === BASIC PRESSURE DIAGNOSTICS ===
    # Check suction pressures
    for idx in mapping['suctionPressures']:
        col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(col_data) > 0:
            avg_pressure = col_data.mean()
            std_pressure = col_data.std()
            max_pressure = col_data.max()
            min_pressure = col_data.min()

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

            # Pressure variability analysis
            if std_pressure > 15:
                issues.append({
                    'message': f'High suction pressure variability in {headers[idx]} (StdDev: {std_pressure:.1f} PSI)',
                    'severity': 'medium',
                    'explanation': 'Unstable suction pressure indicates cycling issues or control problems',
                    'suggestions': ['Check expansion valve operation', 'Inspect refrigerant flow', 'Verify system controls'],
                    'issue_type': 'control_system'
                })

    # Check discharge pressures with advanced diagnostics
    for idx in mapping['dischargePressures']:
        col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(col_data) > 0:
            avg_pressure = col_data.mean()
            std_pressure = col_data.std()
            max_pressure = col_data.max()
            min_pressure = col_data.min()

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

            # Discharge pressure instability
            if std_pressure > 20:
                issues.append({
                    "severity": "medium",
                    "message": f"Unstable discharge pressure in {headers[idx]} (StdDev: {std_pressure:.1f} PSI)",
                    "explanation": "Pressure instability suggests condenser fan cycling or refrigerant flow issues.",
                    "suggestions": ["Check condenser fan control", "Inspect refrigerant metering", "Verify system capacity"],
                    "issue_type": "condenser_system"
                })

    # === TEMPERATURE DIAGNOSTICS ===
    # Enhanced suction temperature analysis
    for idx in mapping['suctionTemps']:
        col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(col_data) > 0:
            avg_temp = col_data.mean()
            std_temp = col_data.std()
            max_temp = col_data.max()
            min_temp = col_data.min()

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

            # Temperature stability analysis
            if std_temp > 8:
                issues.append({
                    "severity": "low",
                    "message": f"Suction temperature instability in {headers[idx]} (StdDev: {std_temp:.1f}°F)",
                    "explanation": "Temperature fluctuations suggest evaporator loading or refrigerant flow issues.",
                    "suggestions": ["Check evaporator coil", "Inspect refrigerant distribution", "Verify airflow patterns"],
                    "issue_type": "evaporator_system"
                })

    # Supply air temperature diagnostics
    for idx in mapping['supplyAirTemps'] + mapping['dischargeTemps']:
        col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(col_data) > 0:
            avg_temp = col_data.mean()
            std_temp = col_data.std()
            max_temp = col_data.max()
            min_temp = col_data.min()

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

            # Supply air temperature spread analysis
            temp_spread = max_temp - min_temp
            if temp_spread > 25:
                issues.append({
                    "severity": "medium",
                    "message": f"Wide supply air temperature range in {headers[idx]} ({temp_spread:.1f}°F spread)",
                    "explanation": "Large temperature variations suggest capacity control or load matching issues.",
                    "suggestions": ["Check capacity control", "Verify load calculations", "Inspect modulation systems", "Review control sequences"],
                    "issue_type": "capacity_control"
                })

    # === ADVANCED REFRIGERANT CYCLE ANALYSIS ===
    # Pressure ratio analysis
    if mapping['suctionPressures'] and mapping['dischargePressures']:
        for suction_idx in mapping['suctionPressures']:
            for discharge_idx in mapping['dischargePressures']:
                suction_data = pd.to_numeric(df.iloc[:, suction_idx], errors='coerce').dropna()
                discharge_data = pd.to_numeric(df.iloc[:, discharge_idx], errors='coerce').dropna()

                if len(suction_data) > 0 and len(discharge_data) > 0:
                    # Align data lengths
                    min_len = min(len(suction_data), len(discharge_data))
                    suction_aligned = suction_data.iloc[:min_len]
                    discharge_aligned = discharge_data.iloc[:min_len]

                    # Calculate pressure ratio
                    pressure_ratio = discharge_aligned / suction_aligned
                    avg_ratio = pressure_ratio.mean()

                    if avg_ratio > 4.5:
                        issues.append({
                            "severity": "high",
                            "message": f"High compression ratio detected (Ratio: {avg_ratio:.2f})",
                            "explanation": "High compression ratio indicates inefficient operation and potential compressor stress.",
                            "suggestions": ["Check condenser performance", "Verify refrigerant charge", "Inspect evaporator operation", "Consider load reduction"],
                            "issue_type": "compressor_efficiency"
                        })
                    elif avg_ratio < 2.0:
                        issues.append({
                            "severity": "medium",
                            "message": f"Low compression ratio detected (Ratio: {avg_ratio:.2f})",
                            "explanation": "Low compression ratio may indicate compressor wear or bypass issues.",
                            "suggestions": ["Test compressor valves", "Check internal leakage", "Verify compressor condition", "Inspect refrigerant circuit"],
                            "issue_type": "compressor_wear"
                        })

    # === HUMIDITY AND COMFORT DIAGNOSTICS ===
    # Indoor humidity analysis
    for idx in mapping.get('indoorRH', []):
        col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(col_data) > 0:
            avg_humidity = col_data.mean()
            max_humidity = col_data.max()
            min_humidity = col_data.min()
            std_humidity = col_data.std()

            if avg_humidity > 60:
                issues.append({
                    "severity": "medium",
                    "message": f"High indoor humidity in {headers[idx]} (Avg: {avg_humidity:.1f}%)",
                    "explanation": "High indoor humidity promotes mold growth and reduces comfort.",
                    "suggestions": ["Increase dehumidification", "Check drainage", "Verify ventilation rates", "Inspect ductwork for leaks"],
                    "issue_type": "humidity_control"
                })
            elif avg_humidity < 30:
                issues.append({
                    "severity": "low",
                    "message": f"Low indoor humidity in {headers[idx]} (Avg: {avg_humidity:.1f}%)",
                    "explanation": "Low humidity can cause discomfort and static electricity issues.",
                    "suggestions": ["Add humidification systems", "Check for excessive ventilation", "Inspect building envelope"],
                    "issue_type": "humidity_control"
                })

            # Humidity variability
            if std_humidity > 10:
                issues.append({
                    "severity": "low",
                    "message": f"Unstable indoor humidity in {headers[idx]} (StdDev: {std_humidity:.1f}%)",
                    "explanation": "Humidity swings indicate poor humidity control or cycling issues.",
                    "suggestions": ["Check humidity control systems", "Verify proper sizing", "Inspect control sequences"],
                    "issue_type": "humidity_control"
                })

    # === ENERGY EFFICIENCY DIAGNOSTICS ===
    # Temperature differential analysis
    if mapping.get('supplyAirTemps') and mapping.get('indoorTemps'):
        for supply_idx in mapping['supplyAirTemps']:
            for return_idx in mapping['indoorTemps']:
                supply_data = pd.to_numeric(df.iloc[:, supply_idx], errors='coerce').dropna()
                return_data = pd.to_numeric(df.iloc[:, return_idx], errors='coerce').dropna()

                if len(supply_data) > 0 and len(return_data) > 0:
                    min_len = min(len(supply_data), len(return_data))
                    temp_diff = return_data.iloc[:min_len] - supply_data.iloc[:min_len]
                    avg_diff = temp_diff.mean()

                    if avg_diff < 15:
                        issues.append({
                            "severity": "medium",
                            "message": f"Low temperature differential detected (ΔT: {avg_diff:.1f}°F)",
                            "explanation": "Low temperature differential indicates reduced system capacity or airflow issues.",
                            "suggestions": ["Check airflow rates", "Inspect coil performance", "Verify refrigerant charge", "Check for restrictions"],
                            "issue_type": "system_capacity"
                        })
                    elif avg_diff > 25:
                        issues.append({
                            "severity": "medium",
                            "message": f"High temperature differential detected (ΔT: {avg_diff:.1f}°F)",
                            "explanation": "High temperature differential may indicate insufficient airflow or oversized equipment.",
                            "suggestions": ["Check fan operation", "Inspect ductwork", "Verify equipment sizing", "Check filter restrictions"],
                            "issue_type": "airflow_restriction"
                        })

    # === SYSTEM CYCLING AND CONTROL ISSUES ===
    # Detect short cycling by analyzing data patterns
    for pressure_group in [mapping['suctionPressures'], mapping['dischargePressures']]:
        for idx in pressure_group:
            col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
            if len(col_data) > 10:  # Need sufficient data points
                # Look for rapid changes indicating cycling
                changes = col_data.diff().abs()
                rapid_changes = (changes > changes.std() * 2).sum()
                change_rate = rapid_changes / len(col_data)

                if change_rate > 0.2:  # More than 20% of readings show rapid changes
                    issues.append({
                        "severity": "medium",
                        "message": f"Potential short cycling detected in {headers[idx]}",
                        "explanation": "Frequent pressure changes suggest system is cycling on and off too frequently.",
                        "suggestions": ["Check thermostat differential", "Verify system sizing", "Inspect control systems", "Check for refrigerant issues"],
                        "issue_type": "short_cycling"
                    })

    # === ADDITIONAL NICHE DIAGNOSTICS ===

    # 1. Subcooling analysis (if both discharge temp and pressure available)
    if mapping.get('dischargeTemps') and mapping.get('dischargePressures'):
        for temp_idx in mapping['dischargeTemps']:
            for press_idx in mapping['dischargePressures']:
                temp_data = pd.to_numeric(df.iloc[:, temp_idx], errors='coerce').dropna()
                press_data = pd.to_numeric(df.iloc[:, press_idx], errors='coerce').dropna()

                if len(temp_data) > 0 and len(press_data) > 0:
                    avg_temp = temp_data.mean()
                    avg_press = press_data.mean()

                    # Estimate subcooling (simplified calculation)
                    estimated_saturation_temp = 32 + (avg_press - 14.7) * 0.25  # Rough approximation
                    subcooling = estimated_saturation_temp - avg_temp

                    if subcooling < 5:
                        issues.append({
                            "severity": "medium",
                            "message": f"Low subcooling detected (Est: {subcooling:.1f}°F)",
                            "explanation": "Low subcooling indicates potential undercharge or heat rejection issues.",
                            "suggestions": ["Check refrigerant charge", "Inspect condenser performance", "Verify proper airflow"],
                            "issue_type": "subcooling_low"
                        })
                    elif subcooling > 20:
                        issues.append({
                            "severity": "low",
                            "message": f"High subcooling detected (Est: {subcooling:.1f}°F)",
                            "explanation": "High subcooling may indicate overcharge or condenser oversizing.",
                            "suggestions": ["Check for overcharge", "Verify condenser operation", "Consider load conditions"],
                            "issue_type": "subcooling_high"
                        })

    # 2. Setpoint deviation analysis
    if mapping.get('coolingSetpoints') and mapping.get('indoorTemps'):
        for setpoint_idx in mapping['coolingSetpoints']:
            for temp_idx in mapping['indoorTemps']:
                setpoint_data = pd.to_numeric(df.iloc[:, setpoint_idx], errors='coerce').dropna()
                temp_data = pd.to_numeric(df.iloc[:, temp_idx], errors='coerce').dropna()

                if len(setpoint_data) > 0 and len(temp_data) > 0:
                    min_len = min(len(setpoint_data), len(temp_data))
                    deviation = temp_data.iloc[:min_len] - setpoint_data.iloc[:min_len]
                    avg_deviation = deviation.mean()

                    if avg_deviation > 3:
                        issues.append({
                            "severity": "medium",
                            "message": f"Consistent temperature overshoot (Avg: +{avg_deviation:.1f}°F above setpoint)",
                            "explanation": "Temperature consistently above setpoint indicates insufficient cooling capacity.",
                            "suggestions": ["Check system capacity", "Verify refrigerant charge", "Inspect airflow", "Consider load analysis"],
                            "issue_type": "capacity_insufficient"
                        })
                    elif avg_deviation < -2:
                        issues.append({
                            "severity": "low",
                            "message": f"Consistent temperature undershoot (Avg: {avg_deviation:.1f}°F below setpoint)",
                            "explanation": "Temperature consistently below setpoint may indicate oversized equipment or control issues.",
                            "suggestions": ["Check control settings", "Verify equipment sizing", "Inspect thermostat operation"],
                            "issue_type": "overcooling"
                        })

    # 3. Refrigerant migration detection
    for idx in mapping['suctionPressures']:
        col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(col_data) > 0:
            # Look for pressure changes during off cycles
            pressure_changes = col_data.diff()
            large_drops = (pressure_changes < -10).sum()

            if large_drops > len(col_data) * 0.1:  # More than 10% of readings show large drops
                issues.append({
                    "severity": "low",
                    "message": f"Potential refrigerant migration detected in {headers[idx]}",
                    "explanation": "Pressure drops during off-cycles may indicate refrigerant migration to evaporator.",
                    "suggestions": ["Check crankcase heater", "Inspect liquid line solenoid", "Verify proper shutdown procedures"],
                    "issue_type": "refrigerant_migration"
                })

    # 4. Liquid line restrictions
    if mapping.get('suctionPressures') and mapping.get('dischargePressures'):
        for suct_idx in mapping['suctionPressures']:
            for disch_idx in mapping['dischargePressures']:
                suct_data = pd.to_numeric(df.iloc[:, suct_idx], errors='coerce').dropna()
                disch_data = pd.to_numeric(df.iloc[:, disch_idx], errors='coerce').dropna()

                if len(suct_data) > 0 and len(disch_data) > 0:
                    # Look for pressure differential patterns
                    min_len = min(len(suct_data), len(disch_data))
                    pressure_diff = disch_data.iloc[:min_len] - suct_data.iloc[:min_len]
                    std_diff = pressure_diff.std()

                    if std_diff > 30:  # High variability in pressure differential
                        issues.append({
                            "severity": "medium",
                            "message": f"Variable pressure differential suggests flow restrictions",
                            "explanation": "Inconsistent pressure differential may indicate intermittent flow restrictions.",
                            "suggestions": ["Check liquid line filters", "Inspect expansion valve", "Verify proper refrigerant flow", "Check for moisture/debris"],
                            "issue_type": "flow_restriction"
                        })

    # 5. Compressor efficiency analysis
    for idx in mapping['dischargePressures']:
        col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(col_data) > 0:
            # Look for gradual pressure decline (compressor wear)
            if len(col_data) > 50:  # Need sufficient data points
                first_half = col_data.iloc[:len(col_data)//2].mean()
                second_half = col_data.iloc[len(col_data)//2:].mean()
                pressure_decline = first_half - second_half

                if pressure_decline > 15:
                    issues.append({
                        "severity": "medium",
                        "message": f"Gradual pressure decline detected in {headers[idx]}",
                        "explanation": "Decreasing discharge pressure over time may indicate compressor wear or refrigerant loss.",
                        "suggestions": ["Monitor compressor performance", "Check for refrigerant leaks", "Consider compressor evaluation"],
                        "issue_type": "compressor_degradation"
                    })

    # 6. Evaporator icing risk
    for idx in mapping['suctionTemps']:
        col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
        if len(col_data) > 0:
            freezing_risk = (col_data < 32).sum()
            if freezing_risk > 0:
                issues.append({
                    "severity": "high",
                    "message": f"Evaporator icing risk detected in {headers[idx]} ({freezing_risk} readings below 32°F)",
                    "explanation": "Suction temperatures below freezing indicate potential evaporator icing.",
                    "suggestions": ["Check airflow immediately", "Inspect air filters", "Verify proper refrigerant charge", "Check defrost operation"],
                    "issue_type": "icing_risk"
                })

    # 8. Dew point analysis for moisture control
        for idx in mapping.get('spaceDewPoints', []):
            col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
            if len(col_data) > 0:
                avg_dewpoint = col_data.mean()
                max_dewpoint = col_data.max()

                if avg_dewpoint > 60:
                    issues.append({
                        "severity": "high",
                        "message": f"High dew point detected in {headers[idx]} (Avg: {avg_dewpoint:.1f}°F)",
                        "explanation": "High dew point indicates excessive moisture that can cause condensation and mold issues.",
                        "suggestions": ["Increase dehumidification capacity", "Check building envelope", "Verify ventilation rates", "Inspect for moisture sources"],
                        "issue_type": "moisture_control"
                    })

                if max_dewpoint > 65:
                    issues.append({
                        "severity": "high",
                        "message": f"Critical dew point levels detected in {headers[idx]} (Max: {max_dewpoint:.1f}°F)",
                        "explanation": "Critical dew point levels risk condensation on cool surfaces.",
                        "suggestions": ["Immediate dehumidification action", "Check for water intrusion", "Inspect HVAC drainage"],
                        "issue_type": "condensation_risk"
                    })

        # 9. Compressor fan performance analysis
        for idx in mapping.get('compressorFanSpeeds', []):
            col_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
            if len(col_data) > 0:
                avg_speed = col_data.mean()
                std_speed = col_data.std()
                min_speed = col_data.min()

                # Fan speed variability
                if std_speed > avg_speed * 0.15:  # More than 15% variation
                    issues.append({
                        "severity": "medium",
                        "message": f"Compressor fan speed instability in {headers[idx]} (StdDev: {std_speed:.1f})",
                        "explanation": "Unstable fan speed indicates potential motor, control, or power supply issues.",
                        "suggestions": ["Check fan motor condition", "Inspect electrical connections", "Verify VFD operation", "Check control signals"],
                        "issue_type": "fan_instability"
                    })

                # Low fan speed detection
                if min_speed < avg_speed * 0.5 and min_speed > 0:
                    issues.append({
                        "severity": "medium",
                        "message": f"Low compressor fan speeds detected in {headers[idx]} (Min: {min_speed:.1f})",
                        "explanation": "Intermittent low fan speeds may indicate motor problems or control issues.",
                        "suggestions": ["Inspect fan motor bearings", "Check motor overload protection", "Verify control programming"],
                        "issue_type": "fan_performance"
                    })

        # 10. Superheat calculation and analysis
        if mapping.get('suctionTemps') and mapping.get('suctionPressures'):
            for temp_idx in mapping['suctionTemps']:
                for press_idx in mapping['suctionPressures']:
                    temp_data = pd.to_numeric(df.iloc[:, temp_idx], errors='coerce').dropna()
                    press_data = pd.to_numeric(df.iloc[:, press_idx], errors='coerce').dropna()

                    if len(temp_data) > 0 and len(press_data) > 0:
                        min_len = min(len(temp_data), len(press_data))
                        # Simplified saturation temperature calculation
                        sat_temp = 32 + (press_data.iloc[:min_len] - 14.7) * 0.3  # Rough R-410A approximation
                        superheat = temp_data.iloc[:min_len] - sat_temp
                        avg_superheat = superheat.mean()

                        if avg_superheat < 5:
                            issues.append({
                                "severity": "high",
                                "message": f"Low superheat detected (Avg: {avg_superheat:.1f}°F)",
                                "explanation": "Low superheat risks liquid refrigerant return to compressor, causing damage.",
                                "suggestions": ["Adjust expansion valve", "Check refrigerant charge", "Verify proper evaporator airflow"],
                                "issue_type": "superheat_low"
                            })
                        elif avg_superheat > 25:
                            issues.append({
                                "severity": "medium",
                                "message": f"High superheat detected (Avg: {avg_superheat:.1f}°F)",
                                "explanation": "High superheat indicates undercharge or restricted refrigerant flow.",
                                "suggestions": ["Check refrigerant charge", "Inspect expansion valve", "Verify proper metering"],
                                "issue_type": "superheat_high"
                            })

        # 11. Heat pump defrost cycle analysis
        for idx in mapping.get('outdoorAirTemps', []):
            outdoor_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
            if len(outdoor_data) > 0:
                freezing_conditions = (outdoor_data < 40).sum()  # Conditions where defrost may be needed

                if freezing_conditions > len(outdoor_data) * 0.3:  # More than 30% of time in defrost conditions
                    # Check if system is showing signs of frost buildup
                    for press_idx in mapping.get('suctionPressures', []):
                        press_data = pd.to_numeric(df.iloc[:, press_idx], errors='coerce').dropna()
                        if len(press_data) > 0:
                            # Low pressure during cold conditions may indicate frost
                            cold_weather_pressure = press_data[outdoor_data < 40].mean() if len(outdoor_data[outdoor_data < 40]) > 0 else None
                            if cold_weather_pressure and cold_weather_pressure < 60:
                                issues.append({
                                    "severity": "medium",
                                    "message": f"Potential defrost issues during cold weather (Pressure: {cold_weather_pressure:.1f} PSI)",
                                    "explanation": "Low suction pressure during cold weather suggests inadequate defrost operation.",
                                    "suggestions": ["Check defrost sensors", "Verify defrost control logic", "Inspect outdoor coil", "Check defrost termination"],
                                    "issue_type": "defrost_issues"
                                })

        # 12. Economizer operation analysis
        if mapping.get('outdoorAirTemps') and mapping.get('indoorTemps'):
            for outdoor_idx in mapping['outdoorAirTemps']:
                for indoor_idx in mapping['indoorTemps']:
                    outdoor_data = pd.to_numeric(df.iloc[:, outdoor_idx], errors='coerce').dropna()
                    indoor_data = pd.to_numeric(df.iloc[:, indoor_idx], errors='coerce').dropna()

                    if len(outdoor_data) > 0 and len(indoor_data) > 0:
                        min_len = min(len(outdoor_data), len(indoor_data))

                        # Find times when economizer should be active (outdoor < indoor - 2°F)
                        economizer_opportunity = (outdoor_data.iloc[:min_len] < indoor_data.iloc[:min_len] - 2).sum()

                        if economizer_opportunity > min_len * 0.2:  # More than 20% of time
                            issues.append({
                                "severity": "low",
                                "message": f"Economizer opportunities detected ({economizer_opportunity} instances)",
                                "explanation": "Outdoor conditions favorable for free cooling were available but may not have been utilized.",
                                "suggestions": ["Verify economizer operation", "Check damper controls", "Inspect outdoor air sensors", "Review economizer logic"],
                                "issue_type": "economizer_optimization"
                            })

        # 13. Supply air reset analysis
        for idx in mapping.get('supplyAirSetpoints', []):
            setpoint_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
            if len(setpoint_data) > 0:
                setpoint_range = setpoint_data.max() - setpoint_data.min()

                if setpoint_range < 5:
                    issues.append({
                        "severity": "low",
                        "message": f"Limited supply air reset detected in {headers[idx]} (Range: {setpoint_range:.1f}°F)",
                        "explanation": "Fixed supply air temperature may indicate unused energy savings opportunity.",
                        "suggestions": ["Implement supply air reset", "Check reset control logic", "Verify outdoor air temperature sensor"],
                        "issue_type": "supply_air_reset"
                    })

        # 14. Simultaneous heating and cooling detection
        if mapping.get('heatingSetpoints') and mapping.get('coolingSetpoints'):
            for heat_idx in mapping['heatingSetpoints']:
                for cool_idx in mapping['coolingSetpoints']:
                    heat_data = pd.to_numeric(df.iloc[:, heat_idx], errors='coerce').dropna()
                    cool_data = pd.to_numeric(df.iloc[:, cool_idx], errors='coerce').dropna()

                    if len(heat_data) > 0 and len(cool_data) > 0:
                        min_len = min(len(heat_data), len(cool_data))
                        deadband = cool_data.iloc[:min_len] - heat_data.iloc[:min_len]
                        avg_deadband = deadband.mean()

                        if avg_deadband < 3:
                            issues.append({
                                "severity": "medium",
                                "message": f"Narrow temperature deadband detected (Avg: {avg_deadband:.1f}°F)",
                                "explanation": "Narrow deadband between heating and cooling may cause simultaneous operation and energy waste.",
                                "suggestions": ["Increase deadband to 3-5°F", "Check control sequences", "Verify sensor calibration"],
                                "issue_type": "deadband_narrow"
                            })

        # 15. Power quality and electrical issues
        if mapping.get('powerConsumption', []):
            for idx in mapping['powerConsumption']:
                power_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
                if len(power_data) > 0:
                    power_variation = power_data.std() / power_data.mean()

                    if power_variation > 0.3:  # High power variation
                        issues.append({
                            "severity": "medium",
                            "message": f"High power consumption variation in {headers[idx]} (CV: {power_variation:.2f})",
                            "explanation": "Unstable power consumption may indicate electrical issues or equipment cycling problems.",
                            "suggestions": ["Check electrical connections", "Inspect contactors and relays", "Monitor voltage stability", "Verify load balance"],
                            "issue_type": "power_instability"
                        })

        # 16. Ventilation adequacy analysis
        if mapping.get('outdoorAirFlow', []):
            for idx in mapping['outdoorAirFlow']:
                oa_flow_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
                if len(oa_flow_data) > 0:
                    avg_oa_flow = oa_flow_data.mean()
                    min_oa_flow = oa_flow_data.min()

                    # Check for inadequate outdoor air
                    if min_oa_flow < avg_oa_flow * 0.5:
                        issues.append({
                            "severity": "medium",
                            "message": f"Inadequate outdoor air flow detected in {headers[idx]} (Min: {min_oa_flow:.1f})",
                            "explanation": "Low outdoor air flow compromises indoor air quality and may violate codes.",
                            "suggestions": ["Check outdoor air dampers", "Inspect air handling unit", "Verify minimum ventilation settings", "Check for blockages"],
                            "issue_type": "ventilation_inadequate"
                        })

        # 17. Filter loading analysis
        if mapping.get('filterPressureDrop', []):
            for idx in mapping['filterPressureDrop']:
                filter_dp_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
                if len(filter_dp_data) > 0:
                    max_filter_dp = filter_dp_data.max()
                    avg_filter_dp = filter_dp_data.mean()

                    if max_filter_dp > 2.0:  # High filter pressure drop
                        issues.append({
                            "severity": "medium",
                            "message": f"High filter pressure drop detected in {headers[idx]} (Max: {max_filter_dp:.2f} in WC)",
                            "explanation": "High filter pressure drop reduces airflow and increases energy consumption.",
                            "suggestions": ["Replace air filters", "Check filter installation", "Consider higher capacity filters", "Implement filter monitoring"],
                            "issue_type": "filter_loading"
                        })

        # 18. Thermal comfort analysis
        if mapping.get('indoorTemps') and mapping.get('indoorRH'):
            for temp_idx in mapping['indoorTemps']:
                for rh_idx in mapping['indoorRH']:
                    temp_data = pd.to_numeric(df.iloc[:, temp_idx], errors='coerce').dropna()
                    rh_data = pd.to_numeric(df.iloc[:, rh_idx], errors='coerce').dropna()

                    if len(temp_data) > 0 and len(rh_data) > 0:
                        min_len = min(len(temp_data), len(rh_data))

                        # Simple comfort analysis (ASHRAE comfort zone approximation)
                        comfort_violations = 0
                        for i in range(min_len):
                            temp = temp_data.iloc[i]
                            rh = rh_data.iloc[i]

                            # Basic comfort zone check (68-78°F, 30-60% RH)
                            if temp < 68 or temp > 78 or rh < 30 or rh > 60:
                                comfort_violations += 1

                        comfort_compliance = (min_len - comfort_violations) / min_len

                        if comfort_compliance < 0.8:  # Less than 80% compliance
                            issues.append({
                                "severity": "low",
                                "message": f"Poor thermal comfort compliance ({comfort_compliance:.1%})",
                                "explanation": "Conditions frequently outside ASHRAE comfort zone may cause occupant complaints.",
                                "suggestions": ["Review setpoint strategies", "Check zone control", "Verify sensor calibration", "Consider comfort surveys"],
                                "issue_type": "thermal_comfort"
                            })

        # 19. Energy efficiency trending
        if mapping.get('powerConsumption') and mapping.get('outdoorAirTemps'):
            for power_idx in mapping['powerConsumption']:
                for temp_idx in mapping['outdoorAirTemps']:
                    power_data = pd.to_numeric(df.iloc[:, power_idx], errors='coerce').dropna()
                    temp_data = pd.to_numeric(df.iloc[:, temp_idx], errors='coerce').dropna()

                    if len(power_data) > 20 and len(temp_data) > 20:
                        min_len = min(len(power_data), len(temp_data))

                        # Look for efficiency degradation over time
                        first_quarter = power_data.iloc[:min_len//4].mean()
                        last_quarter = power_data.iloc[3*min_len//4:].mean()

                        efficiency_change = (last_quarter - first_quarter) / first_quarter

                        if efficiency_change > 0.15:  # More than 15% increase in power consumption
                            issues.append({
                                "severity": "medium",
                                "message": f"Energy efficiency degradation detected ({efficiency_change:.1%} increase)",
                                "explanation": "Increasing power consumption over time suggests equipment degradation or fouling.",
                                "suggestions": ["Schedule maintenance inspection", "Check coil cleanliness", "Verify refrigerant charge", "Inspect mechanical components"],
                                "issue_type": "efficiency_degradation"
                            })

        # 20. Space pressure analysis
        if mapping.get('spacePressure', []):
            for idx in mapping['spacePressure']:
                pressure_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
                if len(pressure_data) > 0:
                    avg_pressure = pressure_data.mean()
                    std_pressure = pressure_data.std()

                    if abs(avg_pressure) > 0.05:  # More than 0.05" WC
                        issues.append({
                            "severity": "medium",
                            "message": f"Space pressure imbalance detected in {headers[idx]} (Avg: {avg_pressure:.3f} in WC)",
                            "explanation": "Significant space pressure indicates ventilation imbalance affecting comfort and energy.",
                            "suggestions": ["Balance supply and return air", "Check building envelope", "Verify exhaust systems", "Inspect damper operation"],
                            "issue_type": "pressure_imbalance"
                        })

                    if std_pressure > 0.02:  # Unstable pressure
                        issues.append({
                            "severity": "low",
                            "message": f"Unstable space pressure in {headers[idx]} (StdDev: {std_pressure:.3f} in WC)",
                            "explanation": "Pressure instability suggests control or airflow issues.",
                            "suggestions": ["Check pressure control systems", "Inspect damper operation", "Verify airflow measurements"],
                            "issue_type": "pressure_instability"
                        })

        # 21. Refrigerant leak detection
        for idx in mapping.get('suctionPressures', []):
            pressure_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
            if len(pressure_data) > 50:  # Need substantial data
                # Look for gradual pressure decline over time
                trend_slope = np.polyfit(range(len(pressure_data)), pressure_data, 1)[0]

                if trend_slope < -0.1:  # Declining trend
                    issues.append({
                        "severity": "high",
                        "message": f"Potential refrigerant leak detected in {headers[idx]} (Declining trend: {trend_slope:.2f} PSI/reading)",
                        "explanation": "Gradual pressure decline suggests refrigerant loss through leakage.",
                        "suggestions": ["Perform leak detection", "Check all refrigerant connections", "Inspect coil integrity", "Monitor refrigerant levels"],
                        "issue_type": "refrigerant_leak"
                    })

        # 22. Condensate drain issues
        if mapping.get('condensatePanLevel', []):
            for idx in mapping['condensatePanLevel']:
                level_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
                if len(level_data) > 0:
                    max_level = level_data.max()
                    avg_level = level_data.mean()

                    if max_level > 0.8:  # High water level (assuming normalized 0-1 scale)
                        issues.append({
                            "severity": "high",
                            "message": f"High condensate pan level detected in {headers[idx]} (Max: {max_level:.2f})",
                            "explanation": "High condensate levels risk overflow and water damage.",
                            "suggestions": ["Clear condensate drain", "Check drain pump operation", "Inspect drain lines", "Verify proper slope"],
                            "issue_type": "condensate_drainage"
                        })

        # 23. Variable frequency drive analysis
        if mapping.get('vfdOutput', []):
            for idx in mapping['vfdOutput']:
                vfd_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
                if len(vfd_data) > 0:
                    avg_output = vfd_data.mean()
                    max_output = vfd_data.max()
                    min_output = vfd_data.min()

                    if avg_output > 90:  # Consistently high VFD output
                        issues.append({
                            "severity": "medium",
                            "message": f"High VFD output detected in {headers[idx]} (Avg: {avg_output:.1f}%)",
                            "explanation": "Consistently high VFD output suggests undersized equipment or excessive load.",
                            "suggestions": ["Check system load requirements", "Verify equipment sizing", "Inspect for restrictions", "Consider capacity upgrades"],
                            "issue_type": "vfd_high_output"
                        })

                    if max_output - min_output < 20 and avg_output > 50:  # Limited modulation
                        issues.append({
                            "severity": "low",
                            "message": f"Limited VFD modulation in {headers[idx]} (Range: {max_output - min_output:.1f}%)",
                            "explanation": "Limited VFD modulation reduces energy savings potential.",
                            "suggestions": ["Review control programming", "Check minimum speed settings", "Verify load diversity", "Consider control optimization"],
                            "issue_type": "vfd_limited_modulation"
                        })

        # 24. Indoor air quality indicators
        if mapping.get('co2Levels', []):
            for idx in mapping['co2Levels']:
                co2_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
                if len(co2_data) > 0:
                    max_co2 = co2_data.max()
                    avg_co2 = co2_data.mean()

                    if max_co2 > 1000:  # High CO2 levels
                        issues.append({
                            "severity": "medium",
                            "message": f"High CO2 levels detected in {headers[idx]} (Max: {max_co2:.0f} ppm)",
                            "explanation": "High CO2 levels indicate inadequate ventilation and poor indoor air quality.",
                            "suggestions": ["Increase outdoor air ventilation", "Check occupancy levels", "Verify ventilation control", "Consider demand-controlled ventilation"],
                            "issue_type": "indoor_air_quality"
                        })

        # 25. Equipment runtime optimization
        for temp_group in [mapping.get('indoorTemps', []), mapping.get('supplyAirTemps', [])]:
            for idx in temp_group:
                temp_data = pd.to_numeric(df.iloc[:, idx], errors='coerce').dropna()
                if len(temp_data) > 100:  # Need substantial data for runtime analysis
                    # Detect equipment cycling frequency
                    temp_changes = temp_data.diff().abs()
                    significant_changes = (temp_changes > temp_changes.std()).sum()
                    cycle_frequency = significant_changes / len(temp_data)

                    if cycle_frequency > 0.15:  # More than 15% of readings show significant changes
                        issues.append({
                            "severity": "low",
                            "message": f"Frequent equipment cycling detected in {headers[idx]} (Frequency: {cycle_frequency:.2f})",
                            "explanation": "Frequent cycling reduces equipment life and energy efficiency.",
                            "suggestions": ["Adjust control differentials", "Check equipment sizing", "Review thermostat settings", "Consider staging optimization"],
                            "issue_type": "frequent_cycling"
                        })

        # Deduplicate similar issues
        seen_issues = {}
        deduplicated_issues = []

        for issue in issues:
            # Create a key based on issue type and main message content
            key_parts = [issue['issue_type'], issue['severity']]

            # For certain issue types, make the key more specific to avoid over-deduplication
            if issue['issue_type'] in ['thermal_comfort', 'humidity_control', 'pressure_imbalance']:
                # Extract the main diagnostic value from message
                import re
                values = re.findall(r'\d+\.?\d*', issue['message'])
                if values:
                    key_parts.append(values[0])  # Use first numerical value

            key = tuple(key_parts)

            if key not in seen_issues:
                seen_issues[key] = len(deduplicated_issues)
                deduplicated_issues.append(issue)
            else:
                # Merge suggestions from duplicate
                existing_idx = seen_issues[key]
                existing_suggestions = set(deduplicated_issues[existing_idx]['suggestions'])
                new_suggestions = set(issue['suggestions'])
                deduplicated_issues[existing_idx]['suggestions'] = list(existing_suggestions | new_suggestions)

        issues = deduplicated_issues

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

            stats_data = generate_enhanced_data_summary(df_summary)

            if stats_data and len(stats_data) > 1:
                table = Table(stats_data, colWidths=[2.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.6*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
                ]))
                story.append(table)
            else:
                story.append(Paragraph("No meaningful data available for statistical analysis.", normal_style))

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

# Add this function to apply filtering to your dataframes before analysis
def filter_dataframe_for_analysis(df, mapping, zero_threshold=0.95):
    """
    Filter the dataframe to only include meaningful columns for analysis and plotting
    Returns filtered dataframe and updated mapping
    """
    # Get meaningful columns (excluding datetime and source_file)
    exclude_prefixes = ['SAT-StPt-Clg', 'SAT-StPt-Dehum', 'SAT-StPt-Htg']
    exclude_cols = ['parsed_datetime', 'source_file'] + [
        col for col in df.columns if any(col.startswith(prefix) for prefix in exclude_prefixes)
    ]
    analysis_df = df.drop(columns=[col for col in exclude_cols if col in df.columns])

    meaningful_cols = filter_meaningful_columns_strict(analysis_df, zero_threshold)

    # Create filtered dataframe with meaningful columns plus datetime
    filtered_cols = meaningful_cols.copy()
    if 'parsed_datetime' in df.columns:
        filtered_cols.append('parsed_datetime')
    if 'source_file' in df.columns:
        filtered_cols.append('source_file')

    filtered_df = df[filtered_cols].copy()

    # Update mapping to only include meaningful columns
    updated_mapping = {}
    original_headers = df.columns.tolist()

    for category, indices in mapping.items():
        if category in ['date', 'time', 'datetime']:
            updated_mapping[category] = indices
        else:
            updated_indices = []
            if isinstance(indices, list):
                for idx in indices:
                    if idx < len(original_headers) and original_headers[idx] in meaningful_cols:
                        # Find new index in filtered dataframe
                        try:
                            new_idx = filtered_df.columns.get_loc(original_headers[idx])
                            updated_indices.append(new_idx)
                        except KeyError:
                            continue
            updated_mapping[category] = updated_indices

    return filtered_df, updated_mapping

# Updated create_time_series_plots function
def create_time_series_plots_filtered(df, headers, mapping):
    """Create temperature vs time and pressure vs time plots with filtered data"""
    plots = []

    # Filter the dataframe first
    filtered_df, filtered_mapping = filter_dataframe_for_analysis(df, mapping)
    filtered_headers = filtered_df.columns.tolist()

    # Temperature vs Time Plot
    temp_indices = (filtered_mapping['suctionTemps'] + filtered_mapping['supplyAirTemps'] +
                   filtered_mapping['dischargeTemps'] + filtered_mapping['outdoorAirTemps'] +
                   filtered_mapping['indoorTemps'])

    if temp_indices and 'parsed_datetime' in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

        plotted_any = False
        for idx_num, idx in enumerate(temp_indices[:6]):  # Limit to 6 columns for readability
            if idx < len(filtered_headers):
                temp_data = pd.to_numeric(filtered_df.iloc[:, idx], errors='coerce')
                valid_mask = ~temp_data.isna() & ~filtered_df['parsed_datetime'].isna()

                if valid_mask.sum() > 0:
                    ax.plot(filtered_df.loc[valid_mask, 'parsed_datetime'],
                           temp_data[valid_mask],
                           label=get_legend_label(filtered_headers[idx]),
                           marker='o',
                           markersize=2,
                           linewidth=1,
                           color=colors[idx_num % len(colors)])
                    plotted_any = True

        if plotted_any:
            ax.set_xlabel('Time')
            ax.set_ylabel('Temperature (°F)')
            ax.set_title('Temperature vs Time (Filtered Data)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

            # Format x-axis
            if len(filtered_df) > 0:
                from matplotlib.dates import AutoDateLocator
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                ax.xaxis.set_major_locator(AutoDateLocator(maxticks=24))
                plt.xticks(rotation=45)

            plt.tight_layout()
            plots.append(('Temperature vs Time (Filtered)', fig))
        else:
            plt.close(fig)

    # Pressure vs Time Plot
    pressure_indices = filtered_mapping['suctionPressures'] + filtered_mapping['dischargePressures']

    if pressure_indices and 'parsed_datetime' in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['darkblue', 'darkred', 'darkgreen', 'darkorange', 'purple', 'brown']

        plotted_any = False
        for idx_num, idx in enumerate(pressure_indices[:6]):
            if idx < len(filtered_headers):
                pressure_data = pd.to_numeric(filtered_df.iloc[:, idx], errors='coerce')
                valid_mask = ~pressure_data.isna() & ~filtered_df['parsed_datetime'].isna()

                if valid_mask.sum() > 0:
                    ax.plot(filtered_df.loc[valid_mask, 'parsed_datetime'],
                           pressure_data[valid_mask],
                           label=get_legend_label(filtered_headers[idx]),
                           marker='o',
                           markersize=2,
                           linewidth=1,
                           color=colors[idx_num % len(colors)])
                    plotted_any = True

        if plotted_any:
            ax.set_xlabel('Time')
            ax.set_ylabel('Pressure (PSI)')
            ax.set_title('Pressure vs Time (Filtered Data)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

            # Format x-axis
            if len(filtered_df) > 0:
                from matplotlib.dates import AutoDateLocator
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                ax.xaxis.set_major_locator(AutoDateLocator(maxticks=24))
                plt.xticks(rotation=45)

            plt.tight_layout()
            plots.append(('Pressure vs Time (Filtered)', fig))
        else:
            plt.close(fig)

    # Relative Humidity vs Time Plot

    indoor_rh_indices = [
        idx for idx in filtered_mapping.get('indoorRH', [])
        if 'sprheat' not in filtered_headers[idx].lower() 
            and 'sprhtsp' not in filtered_headers[idx].lower()
    ]
    outdoor_rh_indices = filtered_mapping.get('outdoorRH', [])
    colors = ['teal', 'magenta', 'olive', 'coral', 'gray', 'gold']

    if (indoor_rh_indices or outdoor_rh_indices) and 'parsed_datetime' in filtered_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        plotted_any = False

        # Indoor RH
        for idx_num, idx in enumerate(indoor_rh_indices[:3]):
            if idx < len(filtered_headers):
                rh_data = pd.to_numeric(filtered_df.iloc[:, idx], errors='coerce')
                label = get_legend_label(filtered_headers[idx]) + " (Indoor)"
                valid_mask = ~rh_data.isna() & ~filtered_df['parsed_datetime'].isna()

                if valid_mask.sum() > 0:
                    ax.plot(filtered_df.loc[valid_mask, 'parsed_datetime'],
                           rh_data[valid_mask],
                           label=label,
                           color=colors[idx_num % len(colors)],
                           marker='o', linewidth=1, markersize=2)
                    plotted_any = True

        # Outdoor RH
        for idx_num, idx in enumerate(outdoor_rh_indices[:3]):
            if idx < len(filtered_headers):
                rh_data = pd.to_numeric(filtered_df.iloc[:, idx], errors='coerce')
                label = get_legend_label(filtered_headers[idx]) + " (Outdoor)"
                valid_mask = ~rh_data.isna() & ~filtered_df['parsed_datetime'].isna()

                if valid_mask.sum() > 0:
                    ax.plot(filtered_df.loc[valid_mask, 'parsed_datetime'],
                           rh_data[valid_mask],
                           label=label,
                           linestyle='--',
                           color=colors[(idx_num + 3) % len(colors)],
                           marker='x', linewidth=1, markersize=2)
                    plotted_any = True

        if plotted_any:
            ax.set_xlabel("Time")
            ax.set_ylabel("Relative Humidity (%)")
            ax.set_title("Relative Humidity vs Time (Filtered Data)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)

            # Format x-axis
            if len(filtered_df) > 0:
                from matplotlib.dates import AutoDateLocator
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                ax.xaxis.set_major_locator(AutoDateLocator(maxticks=24))
                plt.xticks(rotation=45)

            plt.tight_layout()
            plots.append(('Relative Humidity vs Time (Filtered)', fig))
        else:
            plt.close(fig)

    return plots

# Updated analysis function to use filtered data
def analyze_hvac_data_enhanced_filtered(df, headers, mapping):
    """Enhanced HVAC analysis with filtered data to exclude zero/meaningless columns"""

    # Filter the dataframe first
    filtered_df, filtered_mapping = filter_dataframe_for_analysis(df, mapping)
    filtered_headers = filtered_df.columns.tolist()

    # Run the original analysis on filtered data
    issues = analyze_hvac_data_enhanced(filtered_df, filtered_headers, filtered_mapping)

    return issues

# Updated comfort check function
def check_comfort_conditions_filtered(df, headers, mapping):
    """Check indoor comfort conditions with filtered data"""

    # Filter the dataframe first
    filtered_df, filtered_mapping = filter_dataframe_for_analysis(df, mapping)
    filtered_headers = filtered_df.columns.tolist()

    # Run the original comfort check on filtered data
    results = check_comfort_conditions(filtered_df, filtered_headers, filtered_mapping)

    return results

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
            issues = analyze_hvac_data_enhanced_filtered(df, headers, mapping)
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
        summary_data = generate_enhanced_data_summary(combined_df)
        if summary_data and len(summary_data) > 1:
            st.markdown("### 📊 Filtered Data Summary")
            st.dataframe(pd.DataFrame(summary_data[1:], columns=summary_data[0]))
        else:
            st.info("No meaningful data available for summary statistics.")

    # Unified Indoor Comfort Check
    if combined_df is not None:
        combined_mapping = parse_headers_enhanced(combined_headers)
        combined_df = create_datetime_column(combined_df, combined_mapping)
        comfort_results = check_comfort_conditions_filtered(combined_df, combined_headers, combined_mapping)

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
    combined_plots = create_time_series_plots_filtered(combined_df, combined_headers, combined_mapping)
    for plot_title, fig in combined_plots:
        st.pyplot(fig)
        plt.close(fig)  # Close figure to free memory

    # Single unified analysis results
    st.markdown("## 📋 HVAC Issues Analysis")
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
