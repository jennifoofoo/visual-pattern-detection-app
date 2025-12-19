import streamlit as st


# ========== HELPER FUNCTIONS FOR SUB-PATTERN SELECTION ==========

def get_parent_visibility(key_prefix: str) -> bool:
    """
    Get the visibility state of the parent pattern from sidebar.
    
    Args:
        key_prefix: Pattern key prefix (e.g. 'temporal_cluster', 'outlier_type', 'gap_transition')
    
    Returns:
        True if parent pattern is visible, False otherwise
    """
    prefix_to_sidebar = {
        'temporal_cluster': 'visible_temporal_cluster',
        'outlier_type': 'visible_outlier',
        'gap_transition': 'visible_gap'
    }
    
    sidebar_key = prefix_to_sidebar.get(key_prefix)
    if sidebar_key:
        return st.session_state.get(sidebar_key, True)
    return True


def sync_sidebar_checkbox(key_prefix: str, value: bool):
    """
    Synchronize sidebar checkbox with tab selection.
    
    Args:
        key_prefix: Pattern key prefix (e.g. 'temporal_cluster', 'outlier_type', 'gap_transition')
        value: True to enable, False to disable
    """
    print(f"🟡 SYNC: sync_sidebar_checkbox('{key_prefix}', {value})")
    
    # Map key_prefix to sidebar session state key
    prefix_to_sidebar = {
        'temporal_cluster': 'visible_temporal_cluster',
        'outlier_type': 'visible_outlier',
        'gap_transition': 'visible_gap'
    }
    
    sidebar_key = prefix_to_sidebar.get(key_prefix)
    if sidebar_key:
        old_value = st.session_state.get(sidebar_key)
        st.session_state[sidebar_key] = value
        print(f"🟡 SYNC: Changed {sidebar_key} from {old_value} to {value}")


def deselect_all_subpatterns(pattern_type: str):
    """
    Deselect all sub-patterns when sidebar checkbox is unchecked.
    
    Args:
        pattern_type: 'temporal', 'outlier', or 'gap'
    """
    if pattern_type == 'temporal':
        for key in list(st.session_state.keys()):
            if key.startswith('list_checkbox_temporal_cluster_'):
                st.session_state[key] = False
    elif pattern_type == 'outlier':
        for key in list(st.session_state.keys()):
            if key.startswith('dict_checkbox_outlier_type_'):
                st.session_state[key] = False
    elif pattern_type == 'gap':
        for key in list(st.session_state.keys()):
            if key.startswith('dict_checkbox_gap_transition_'):
                st.session_state[key] = False


def select_all_subpatterns(pattern_type: str):
    """
    Select all sub-patterns when sidebar checkbox is checked.
    
    Args:
        pattern_type: 'temporal', 'outlier', or 'gap'
    """
    if pattern_type == 'temporal':
        for key in list(st.session_state.keys()):
            if key.startswith('list_checkbox_temporal_cluster_'):
                st.session_state[key] = True
    elif pattern_type == 'outlier':
        for key in list(st.session_state.keys()):
            if key.startswith('dict_checkbox_outlier_type_'):
                st.session_state[key] = True
    elif pattern_type == 'gap':
        for key in list(st.session_state.keys()):
            if key.startswith('dict_checkbox_gap_transition_'):
                st.session_state[key] = True


def list_to_multicheckbox(item_list: list, title: str = "Select Items", key_prefix: str = "item") -> list:
    """
    Renders a Streamlit multi-checkbox interface based on a Python list.
    
    Args:
        item_list: The input list of items to be displayed as checkboxes.
        title: The title to display above the group of checkboxes.
        key_prefix: Prefix for unique checkbox keys.
    
    Returns:
        List containing only the items selected by the user.
    """
    if not item_list:
        st.info("The input list is empty. No checkboxes to display.")
        return []
    
    selected_items = []
    
    # Use a container for visual grouping
    with st.container(border=True):
        st.write(f"**{title}**")
        st.caption("⚠️ Changes will update the chart automatically")
        
        # Add "Select All" / "Deselect All" functionality
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Select All", key=f"{key_prefix}_select_all", use_container_width=True):
                for index in range(len(item_list)):
                    st.session_state[f"list_checkbox_{key_prefix}_{index}"] = True
                # Turn ON parent sidebar checkbox when selecting all
                sync_sidebar_checkbox(key_prefix, True)
                st.rerun()
        with col_b:
            if st.button("Deselect All", key=f"{key_prefix}_deselect_all", use_container_width=True):
                for index in range(len(item_list)):
                    st.session_state[f"list_checkbox_{key_prefix}_{index}"] = False
                # Turn OFF parent sidebar checkbox when deselecting all
                sync_sidebar_checkbox(key_prefix, False)
                st.rerun()
        
        st.markdown("---")
        
        # Check if parent pattern is visible in sidebar
        parent_visible = get_parent_visibility(key_prefix)
        
        # If parent is NOT visible, show message and don't render checkboxes
        if not parent_visible:
            st.info("Enable this pattern in the sidebar to select individual items.")
            return []
        
        for index, item in enumerate(item_list):
            # Simple key without version - parent visible means all are active
            state_key = f"list_checkbox_{key_prefix}_{index}"
            
            # Initialize if needed
            if state_key not in st.session_state:
                st.session_state[state_key] = True  # Default to checked when parent is visible
            
            # Render checkbox
            checked = st.checkbox(str(item), key=state_key)
            
            if checked:
                selected_items.append(item)

    return selected_items


def dict_to_multicheckbox(data_dict: dict, title: str = "Select Items", key_prefix: str = "dict_item") -> list:
    """
    Renders a Streamlit multi-checkbox interface based on a Python dictionary.
    
    Args:
        data_dict: The input dictionary where keys are display labels and values are actual identifiers.
        title: The title to display above the group of checkboxes.
        key_prefix: Prefix for unique checkbox keys.
    
    Returns:
        List containing only the selected dictionary values.
    """
    if not data_dict:
        st.info("The input dictionary is empty.")
        return []
    
    selected_items = []
    
    with st.container(border=True):
        st.write(f"**{title}**")
        st.caption("⚠️ Changes will update the chart automatically")
        
        # Add "Select All" / "Deselect All" functionality
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Select All", key=f"{key_prefix}_select_all", use_container_width=True):
                for key in data_dict.keys():
                    st.session_state[f"dict_checkbox_{key_prefix}_{key}"] = True
                # Sync with sidebar checkbox
                sync_sidebar_checkbox(key_prefix, True)
                # Trigger chart redisplay
                st.session_state['chart_needs_update'] = True
                st.rerun()
        with col_b:
            if st.button("Deselect All", key=f"{key_prefix}_deselect_all", use_container_width=True):
                for key in data_dict.keys():
                    st.session_state[f"dict_checkbox_{key_prefix}_{key}"] = False
                # Sync with sidebar checkbox
                sync_sidebar_checkbox(key_prefix, False)
                # Trigger chart redisplay
                st.session_state['chart_needs_update'] = True
                st.rerun()
        
        st.markdown("---")
        
        # Check if parent pattern is visible in sidebar
        parent_visible = get_parent_visibility(key_prefix)
        
        # If parent is NOT visible, show message and don't render checkboxes
        if not parent_visible:
            st.info("Enable this pattern in the sidebar to select individual items.")
            return []
        
        for key, value in data_dict.items():
            # Simple key without version - parent visible means all are active
            state_key = f"dict_checkbox_{key_prefix}_{key}"
            
            # Initialize if needed
            if state_key not in st.session_state:
                st.session_state[state_key] = True  # Default to checked when parent is visible
            
            # Render checkbox
            checked = st.checkbox(key, key=state_key)
            
            if checked:
                selected_items.append(value)

    return selected_items