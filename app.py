import streamlit as st
from ui.layout import show_header, show_upload_section
from core.data_loader import load_event_log
from core.visualization.dotted_chart import make_dotted_chart

show_header()
file = show_upload_section()

if file is not None:
    try:
        df = load_event_log(file)
        st.success("File uploaded successfully!")
        st.dataframe(df.head(10))

        st.divider()
        view = st.selectbox("Select Dotted Chart view", ["Time", "Case", "Resource", "Performance"], index=0)
        try:
            fig = make_dotted_chart(df, view=view)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"View not available: {e}")

    except ImportError as e:
        st.error(f"{e}")
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Please upload a .csv or .xes file to begin.")
