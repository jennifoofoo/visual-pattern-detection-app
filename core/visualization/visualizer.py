import plotly.express as px

def plot_dotted_chart(df):
    fig = px.scatter(
        df,
        x='x',
        y='activity',
        color='case_id',
        title='Dotted Chart',
        labels={
            'x': df.attrs.get('x_label', 'Time'),
            'activity': 'Activity',
            'case_id': 'Case ID'
        }
    )
    return fig