import plotly.express as px
import plotly.graph_objects as go
import json
import pandas as pd

def get_field_dict(data_path):
    fields = {}
    with open(data_path, 'r') as file:
        data = file.readlines()
        for instance in data:
            instance = json.loads(instance)
            if instance['type'] in fields:
                fields[instance['type']] += 1
            else:
                fields[instance['type']] = 1
    return fields

train_fields = get_field_dict('../data/casiMedicos/JSONL/en.train_casimedicos.jsonl')
dev_fields = get_field_dict('../data/casiMedicos/JSONL/en.dev_casimedicos.jsonl')
test_fields = get_field_dict('../data/casiMedicos/JSONL/en.test_casimedicos.jsonl')

train_fields_pd = pd.DataFrame(train_fields.items(), columns=['field', 'count'])

# Here we use a column with categorical data
fig = go.Figure()
fig.add_trace(go.Histogram(histfunc="sum", y=list(train_fields.values()), x=list(train_fields.keys()), name="train"))
fig.add_trace(go.Histogram(histfunc="sum", y=list(dev_fields.values()), x=list(dev_fields.keys()), name="dev"))
fig.add_trace(go.Histogram(histfunc="sum", y=list(test_fields.values()), x=list(test_fields.keys()), name="test"))
fig.update_layout(
    title="Field distribution in the dataset",
    xaxis_title="Field",
    yaxis_title="Count",
    legend_title="Split",
    font=dict(
        size=4
    ),
    width=2000,
    height=500
)
fig.write_image("../imgs/CasiMedicos_field_distribution.png", width=2000, height=500, scale=3.0)