from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd

'''
Reminder:

Times Representations from the paper:
0. Actual Time: normal timelime
1. Relative Time: first Events are positioned at time 0
2. Relative Ratio: 1. + last Events are positioned at max(time) 
3. Logical Time: Sequence of all Events across all traces
4. Logical Relative: 3. + first Events are positioned at time 0

XES Event Logs Data that we probably care about:
Traces:
    Case_ID = one actual trace      -> max(case_id) = amount of traces
    start_time = timestamp of first event in trace
    end_time = timestamp of last event in trace
    total_events = amount of events
Events:
    'case_id'
    'event_index'
    'activity'
    # time representations
    'actual_time'
    'relative_time'
    'relative_ratio'
    'logical_time'
    'logical_relative'



'''
def load_xes_log(xes_path):
    """
    extracts event features
    possible ToDo: add trace dataframe so we have meta data about traces

    :param xes_path: Path to XES log file.
    :return: DataFrame containing Event Data:
        'case_id', 'event_index' , 'activity', 
        'actual_time', 'relative_time', 'relative_ratio', 
        'logical_time', 'logical_relative'
    """
    log = xes_importer.apply(xes_path)
    events = []

    # Initialize a GLOBAL event counter for Logical Time (3)
    global_event_index = 0

    for trace in log:
        # region Trace attributes
        # trace overall data/attributes
        case_id = trace.attributes.get('concept:name', None)
        if len(trace) == 0:
            continue
        start_time = trace[0].get('time:timestamp', None)
        end_time = trace[-1].get('time:timestamp', None)
        
        # for logical relative ratio if needed
        # total_events = len(trace)
        
        # event duration
        duration_in_seconds = None
        if start_time and end_time:
            duration_in_seconds = (end_time - start_time).total_seconds()
        # endregion 

        # region Event attributes
        # event attributes of a trace
        # is idx relative sequence? --> logical relative?
        for idx, event in enumerate(trace):
            # time stamps
            # 0. timestamp - actual time
            actual_time = event.get('time:timestamp', None)
            # 1. Relative Time: Event time - Trace start time (in seconds)
            relative_time = None
            if start_time and actual_time:
                # Time difference in seconds from the trace's start time
                relative_time = (actual_time - start_time).total_seconds()
            # 2. Relative Ratio: Relative Time / Trace Duration (Normalized time [0, 1] using actual time)
            # This is the more common interpretation for 'Relative Ratio' in time-based charts.
            relative_ratio = None
            if relative_time is not None:
                if duration_in_seconds > 0:
                    # Safe division
                    relative_ratio = relative_time / duration_in_seconds
                elif duration_in_seconds == 0:
                    # If duration is 0, the event is the only event or all events are instantaneous.
                    # Since relative_time must also be 0 in this case, the ratio is 0.
                    relative_ratio = 0.0
            # 3. timestamp - Logical Time: Sequence of all Events across all traces
            logical_time = global_event_index
            global_event_index += 1 # Increment for the next event
            # 4. timestamp - logical_relative
            logical_relative = idx

            
            # # 5. timestamp - logical_relative_ratio 
            # logical_relative = None
            # if start_time and timestamp:
            #     # logical_time = (timestamp - start_time).total_seconds()
            #     if total_events and total_events > 0:
            #         logical_relative = logical_time / total_events
            
            # other attributes
            events.append({
                'case_id': case_id,
                'event_index': idx,
                # might be smart to add total_events for comparing this timestamp to overall
                'activity': event.get('concept:name', None),
                # time representations
                'actual_time': actual_time,
                'relative_time': relative_time,
                'relative_ratio': relative_ratio,
                'logical_time': logical_time,
                'logical_relative': logical_relative
            })
        # endregion
    # maybe ToDo: add trace dataframe so we have meta data about traces
    return pd.DataFrame(events)

if __name__ == '__main__':
    load_xes_log("data/Hospital_log.xes")
