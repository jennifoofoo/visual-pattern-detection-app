import streamlit as st


def show_header():
    st.title("Visual Pattern Detection in Dotted Charts")
    st.write(
        """
        This application allows you to upload event logs (CSV or XES), view the data, and (coming soon) apply pattern detection in visualizations.
        """
    )


def show_upload_section():
    uploaded_file = st.file_uploader(
        "Upload your event log (CSV, XES, or XES.GZ)",
        type=None,  # Allow all; we validate in the loader to support .xes.gz reliably on macOS
        help="Supported: .csv, .xes, .xes.gz. You can also drag & drop."
    )
    return uploaded_file
