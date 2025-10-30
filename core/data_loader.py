import pandas as pd


def load_event_log(file):
    """
    Load an event log from a CSV, XES, or XES.GZ file-like object.
    Returns a preprocessed DataFrame with normalized columns.
    Standard columns (when available): case_id, activity, timestamp, resource
    """
    filename = file.name.lower() if hasattr(file, 'name') else ''
    try:
        # 1) Read raw
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith('.xes') or filename.endswith('.xes.gz'):
            try:
                import pm4py
                from pm4py.objects.log.importer.xes import importer as xes_importer
                from pm4py.objects.conversion.log import converter as log_converter
                from pm4py.objects.log.util import dataframe_utils
                import tempfile
                import gzip
                import os

                # Write uploaded content to a temp file (handle .xes.gz by decompressing)
                if filename.endswith('.xes.gz'):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.xes') as tmp_out:
                        with gzip.GzipFile(fileobj=file, mode='rb') as gz:
                            tmp_out.write(gz.read())
                        temp_path = tmp_out.name
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.xes') as tmp_out:
                        tmp_out.write(file.read())
                        temp_path = tmp_out.name

                try:
                    log = xes_importer.apply(temp_path)
                    df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
                    df = dataframe_utils.convert_timestamp_columns_in_df(df)
                finally:
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
            except ImportError:
                raise ImportError("pm4py is required to load .xes or .xes.gz files. Please install pm4py.")
        else:
            raise ValueError("Unsupported file type. Please upload a .csv, .xes, or .xes.gz file.")

        # 2) Normalize columns -> lowercase
        df.columns = [c.lower() for c in df.columns]

        # 3) Rename common PM4Py-style columns to standard names
        rename_map = {
            'concept:name': 'activity',
            'org:resource': 'resource',
            'time:timestamp': 'timestamp',
            'case:concept:name': 'case_id',
            'caseid': 'case_id',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # 4) Fallbacks if standard columns still missing
        if 'timestamp' not in df.columns:
            for alt in ['time', 'event_time', 'start_time', 'end_time', 'event_timestamp']:
                if alt in df.columns:
                    df = df.rename(columns={alt: 'timestamp'})
                    break

        if 'case_id' not in df.columns:
            for alt in ['case', 'trace_id', 'case identifier']:
                if alt in df.columns:
                    df = df.rename(columns={alt: 'case_id'})
                    break

        if 'activity' not in df.columns:
            for alt in ['activity_name', 'event', 'name']:
                if alt in df.columns:
                    df = df.rename(columns={alt: 'activity'})
                    break

        if 'resource' not in df.columns:
            for alt in ['user', 'resource_name']:
                if alt in df.columns:
                    df = df.rename(columns={alt: 'resource'})
                    break

        # 5) Parse timestamp if present and sort
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp').reset_index(drop=True)

        return df
    except Exception as e:
        raise e
