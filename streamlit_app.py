# --- Streamlit HVAC Analyzer (Unified Version) ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from io import StringIO
import matplotlib.dates as mdates

# Assume the following functions are defined elsewhere or imported:
# - parse_headers_enhanced
# - format_date_enhanced
# - analyze_hvac_data_enhanced
# - check_comfort_conditions
# - generate_diagnostic_reference
# - generate_pdf_report
# - read_csv_with_encoding (with multiple encodings fallback)

st.set_page_config(page_title="Unified HVAC Analyzer", layout="wide")

# Sidebar Logo and Title
logo_file = st.sidebar.file_uploader("Upload Logo (optional)", type=["png", "jpg", "jpeg"])
project_title = st.text_input("Enter Project Title", "HVAC Diagnostic Report")
st.title(project_title)

# Upload Files
uploaded_files = st.file_uploader("Upload CSV or Excel Files", type=['csv', 'xlsx', 'xls'], accept_multiple_files=True)

if uploaded_files:
    all_dataframes = []

    for uploaded_file in uploaded_files:
        try:
            ext = uploaded_file.name.split('.')[-1].lower()
            if ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                df, _ = read_csv_with_encoding(uploaded_file)

            df['source_file'] = uploaded_file.name
            all_dataframes.append(df)
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    headers = combined_df.columns.tolist()
    mapping = parse_headers_enhanced(headers)

    # Build __datetime__ column
    if mapping['date'] is not None and mapping['time'] is not None:
        combined_df['__datetime__'] = combined_df.apply(
            lambda row: format_date_enhanced(row.iloc[mapping['date']], row.iloc[mapping['time']]), axis=1)
    elif mapping['date'] is not None:
        combined_df['__datetime__'] = pd.to_datetime(combined_df.iloc[:, mapping['date']], errors='coerce')
    elif mapping['time'] is not None:
        combined_df['__datetime__'] = pd.to_datetime(combined_df.iloc[:, mapping['time']], errors='coerce')

    df_plot = combined_df[combined_df['__datetime__'].notna()].copy()

    issues = analyze_hvac_data_enhanced(combined_df, headers, mapping)
    comfort_results = check_comfort_conditions(combined_df, headers, mapping)
    diagnostic_ref = generate_diagnostic_reference(issues)

    st.subheader("📊 Combined Data Preview")
    st.dataframe(combined_df.head())

    if not df_plot.empty:
        st.subheader("📈 Combined Time-Series Graphs")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        def format_ax(ax):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%I %p'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True)

        for idx in mapping['suctionPressures']:
            ax1.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'),
                     label=headers[idx], color='blue', linewidth=2)
        for idx in mapping['dischargePressures']:
            ax1.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'),
                     label=headers[idx], color='navy', linewidth=2)
        ax1.set_title("System Pressures")
        ax1.set_ylabel("Pressure (PSI)")
        ax1.legend()
        format_ax(ax1)

        for idx in mapping['suctionTemps'] + mapping['supplyAirTemps'] + mapping['dischargeTemps']:
            ax2.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'),
                     label=headers[idx], linewidth=2)
        ax2.set_title("System Temperatures")
        ax2.set_ylabel("Temperature (°F)")
        ax2.legend()
        format_ax(ax2)

        for idx in mapping['outdoorAirTemps']:
            ax3.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'),
                     label=headers[idx], color='green', linewidth=2)
        ax3.set_title("Outdoor Air Temperature")
        ax3.set_ylabel("Temperature (°F)")
        ax3.legend()
        format_ax(ax3)

        for idx in mapping['coolingSetpoints'] + mapping['heatingSetpoints']:
            ax4.plot(df_plot['__datetime__'], pd.to_numeric(df_plot.iloc[:, idx], errors='coerce'),
                     label=headers[idx], linewidth=2)
        ax4.set_title("Temperature Setpoints")
        ax4.set_ylabel("Temperature (°F)")
        ax4.legend()
        format_ax(ax4)

        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("🏠 Indoor Comfort Conditions")
    if comfort_results:
        for result in comfort_results:
            label = result["column"]
            if result["type"] == "Relative Humidity":
                st.write(f"**{label}** Avg: {result['average']:.1f}% — " +
                         ("✅ OK" if result['compliant'] else f"⚠️ {result['percent_over']:.1f}% over 60%"))
            elif result["type"] == "Indoor Temperature":
                st.write(f"**{label}** Avg: {result['average']:.1f}°F — " +
                         ("✅ OK" if result['compliant'] else f"⚠️ {result['percent_outside']:.1f}% outside 70–75°F"))
    else:
        st.info("No comfort-related data found.")

    st.subheader("🚨 Detected HVAC Issues")
    if issues:
        for issue in issues:
            level = issue['severity']
            (st.error if level == 'high' else st.warning if level == 'medium' else st.info)(f"**{issue['message']}**")
            st.markdown(f"**Why:** {issue['explanation']}")
            st.markdown("**Fix:**")
            for fix in issue['suggestions']:
                st.markdown(f"- {fix}")
            if 'outlier_count' in issue:
                st.markdown(f"**Outliers:** {issue['outlier_count']}")
            st.markdown("---")
    else:
        st.success("✅ No major HVAC issues found.")

    st.subheader("📘 Diagnostic Reference")
    for category, problems in diagnostic_ref.items():
        st.markdown(f"### {category}")
        for title, detail in problems.items():
            with st.expander(title):
                st.write(f"**Symptoms:** {detail['symptoms']}")
                st.write(f"**Causes:** {detail['causes']}")
                st.write("**Diagnostics:**")
                for d in detail['diagnostics']:
                    st.markdown(f"- {d}")
                st.write("**Solutions:**")
                for s in detail['solutions']:
                    st.markdown(f"- {s}")

    st.subheader("📄 Export Report")
    if st.button("Generate PDF Report"):
        try:
            pdf_buffer = generate_pdf_report(project_title, logo_file, issues, combined_df)
            st.download_button("📥 Download PDF", pdf_buffer,
                               file_name=f"HVAC_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                               mime="application/pdf")
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")
else:
    st.info("👆 Please upload one or more CSV/XLSX files to begin analysis.")
