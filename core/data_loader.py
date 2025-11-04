from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd


def load_xes_log(xes_path):
    log = xes_importer.apply(xes_path)
    events = []
    for trace in log:
        case_id = trace.attributes.get('concept:name', None)
        if len(trace) == 0:
            continue
        start_time = trace[0].get('time:timestamp', None)
        end_time = trace[-1].get('time:timestamp', None)
        total_events = len(trace)
        total_logical_time = None
        if start_time and end_time:
            # not sure if this is logical time or 
            total_logical_time = (end_time - start_time).total_seconds()
        for idx, event in enumerate(trace):
            timestamp = event.get('time:timestamp', None)
            logical_time = None
            logical_relative = None
            relative_ratio = None
            if start_time and timestamp:
                logical_time = (timestamp - start_time).total_seconds()
                if total_logical_time and total_logical_time > 0:
                    logical_relative = logical_time / total_logical_time
            if total_events > 1:
                relative_ratio = idx / (total_events - 1)
            events.append({
                'case_id': case_id,
                'activity': event.get('concept:name', None),
                'timestamp': timestamp,
                'logical_time': logical_time,
                'event_index': idx,
                'relative_ratio': relative_ratio,
                'logical_relative': logical_relative
            })
    return pd.DataFrame(events)

if __name__ == '__main__':
    load_xes_log("data/Hospital_log.xes")
