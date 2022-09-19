from dash import Dash, dcc, html, Input, Output
import plotly.express as px

import pandas as pd
import sys
from pathlib import Path
import plotly.graph_objects as go

# %%
PROJECT_ROOT = Path.cwd().parent
DATA_FOLDER = PROJECT_ROOT / "data"
sys.path.append(str(PROJECT_ROOT))

# Constants
PATH_TO_DEMAND_FILE = DATA_FOLDER / "Demand_History.csv"
PATH_TO_SUPPLY_FILE = DATA_FOLDER / "exisiting_EV_infrastructure_2018.csv"


df_demand = pd.read_csv(PATH_TO_DEMAND_FILE)
df_supply = pd.read_csv(PATH_TO_SUPPLY_FILE)
df_supply_melted = df_supply.melt(id_vars=["supply_point_index", "x_coordinate", "y_coordinate"],
                                  var_name="type_variable",
                                  value_name="number_stations")
# %%
years = ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018"]
year_int = [int(y) for y in years]
min_demand = df_demand[years].min(numeric_only=True).min()
max_demand = df_demand[years].max(numeric_only=True).max()
# %%
# Prepare Supply Points Figure
fig_supply = px.scatter(df_supply_melted, x="x_coordinate", y="y_coordinate", color="type_variable",
                        size='number_stations',
                        title="Supply MAP")
fig_supply.update_layout(width=800, height=700)

# %%
# Build DataViz WebApp on Dash
app = Dash(__name__)

app.layout = html.Div([
    html.H2('Demand Points history over years'),
    dcc.Slider(
        min(year_int),
        max(year_int),
        step=None,
        value=min(year_int),
        marks={str(year): str(year) for year in year_int},
        id='year-slider',
    ),
    dcc.Graph(id='demand-points-graph-with-slider'),
    html.H2('Supply Points MAP'),
    dcc.Graph(id='supply-points-graph', figure=fig_supply),

])


@app.callback(
    Output('demand-points-graph-with-slider', 'figure'),
    Input('year-slider', 'value'))
def update_figure(selected_year):
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=df_demand["x_coordinate"],
        y=df_demand["y_coordinate"],
        z=df_demand[str(selected_year)] / max(df_demand[str(selected_year)]),
        # zmin=min_demand,
        # zmax=max_demand,
        hoverongaps=True,
        colorscale="aggrnyl"))

    fig.update_layout(width=700, height=700)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
