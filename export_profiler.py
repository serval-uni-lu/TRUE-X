import pandas as pd
import uuid

primitive_ids = {}

primitive_digests = {}

problem_ids = {}

problem_digests = {}

manual_primitive_types = {}


def get_id(key, lookup):
    if key not in lookup.keys():
        lookup[key] = str(uuid.uuid4())
    return lookup[key]


def create_step(idx, primitive_type, primitive_value):
    return {
        "type": "PRIMITIVE",
        "primitive": {
            "id": get_id(primitive_value, primitive_ids),
            "version": "0.1.0",
            "python_path": f"primitive.primitive.{primitive_type}.{primitive_value}",
            "name": f"{primitive_value} {primitive_type}",
            "digest": get_id(primitive_value, primitive_digests)
        },
        "arguments": {
            "inputs": {
                "type": "CONTAINER",
                "data": f"steps.{idx}.produce"
            }
        },
        "outputs": [{
            "id": "produce"
        }],
        "hyperparams": {
        }
    }


def create_score(metric, value):
    return {
        "metric": {
            "metric": metric,
            "params": {}
        },
        "value": round(value,4)
    }


def create_pipeline_for_row(steps, scores, problem):
    return {

        "pipeline_id": str(uuid.uuid4()),
        "pipeline_digest": str(uuid.uuid4()),
        "pipeline_source": {
            "name": "Combination",
            "contact": ""
        },
        "inputs": [{
            "name": "input dataset"
        }],
        "outputs": [{
            "data": "",
            "name": "predictions of input dataset"
        }],
        "problem": {
            "digest": get_id(problem, problem_digests),
            "id": get_id(problem, problem_ids)
        },
        "start": "2019-08-07T00:26:42.479929Z",
        "end": "2019-08-07T00:26:43.746437Z",
        "steps": steps,
        "scores": scores
    }


def create_pipelines_from_csv(filepath, metric_column, primitive_columns, metric_scores, dataset="", models=[], explainers=[], selected_metrics=[]):
    df = pd.read_csv(filepath)
    df = df[df['Dataset'] == dataset]
    df = df[df['Explainer'].isin(explainers)]
    df = df[df['Model'].isin(models)]
    df = df[df['Metric'].isin(selected_metrics)]
    pipelines = {}
    min_max = {}
    for _, row in df.iterrows():
        pipeline_key = ""
        steps = []
        metric = row[metric_column]
        problem = 'classification' if row[metric_column] == 'Accuracy' else 'regression'
        for i in range(len(primitive_columns)):
            manual_primitive_types[f"{primitive_columns[i]}.{row[primitive_columns[i]]}"] = primitive_columns[i]
            pipeline_key += row[primitive_columns[i]] + ' '
            steps.append(create_step(i, primitive_columns[i], row[primitive_columns[i]]))

        scores = []
        for metric_score in metric_scores:
            scores.append(create_score(f"{metric}", row[metric_score]))
            if f"{metric}" in min_max.keys():
                if row[metric_score] > min_max[f"{metric}"]['max']:
                    min_max[f"{metric}"]['max'] = row[metric_score]
                if row[metric_score] < min_max[f"{metric}"]['min']:
                    min_max[f"{metric}"]['min'] = row[metric_score]
            else:
                min_max[f"{metric}"] = {
                    'max': row[metric_score],
                    'min': row[metric_score]
                }

        if pipeline_key in pipelines.keys():
            pipelines[pipeline_key]['scores'] += scores
        else:
            pipelines[pipeline_key] = create_pipeline_for_row(steps, scores, problem)

    for pipeline in pipelines.values():
        for score in pipeline['scores']:
            m = score['metric']['metric']
            score['normalized'] = round(0.0 if (min_max[m]['max'] - min_max[m]['min']) == 0.0 else (score['value'] - min_max[m]['min']) / (min_max[m]['max'] - min_max[m]['min']), 4)
            if score['normalized'] < 0.0:
                print(pipeline)
                print(score)
                print(min_max[m])


    return pipelines, manual_primitive_types
