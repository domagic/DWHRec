import os

import pandas as pd

from src.model.recommendation import recommendations_phase


# 自定义函数，接受两个参数：column 和 multiplier
def multiply_column(column, multiplier):
    return column * multiplier


if __name__ == '__main__':
    # 示例 DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    # 在列上应用自定义函数，并传递额外的参数
    result_series = df['A'].apply(multiply_column, args=(2,))
    print(result_series)

    project_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    dataset_name = '100k'
    control_args = {
        'name': 'DWHRec',
        'dataset': {'name': dataset_name, 'fold': '0', },
        'model': {'hypergraph': False, 'walk': False, 'embedding': False,
                  'recommendation': True},
        'ablation': {'tag-track': True, 'album-track': True, 'artist-track': False},
        'parameter': {'r': 3, 'k': 200, 's': 50, 'w': 5, 'n': 100, },
        'path': {'data_folder': os.path.join(project_folder, 'datasets', dataset_name),
                 'temp_folder': os.path.join(project_folder, 'temp', 'ablation', '100k', ','.join(
                     ['r=' + '3', 'k=' + '200', 's=' + '50', 'w=' + '5', 'e2=' + str(True), 'e3=' + str(True),
                      'e4=' + str(False)])),
                 'result_folder': os.path.join(project_folder, 'results', 'ablation', dataset_name, ','.join(
                     ['r=' + str(3), 'k=' + str(200), 's=' + str(50), 'w=' + str(5), 'e2=' + str(True),
                      'e3=' + str(True), 'e4=' + str(False)])), },
    }
    embeddings_data_folder = os.path.join(control_args['path']['temp_folder'], 'embedding')
    recommendations_phase(embeddings_data_folder, control_args)
