import os
import numpy as np
import plotly.graph_objects as go
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import plotly.express as px


def load_tensorboard_data(logdir):
    accumulator = EventAccumulator(logdir)
    accumulator.Reload()
    data = accumulator.Scalars('Val/Accuracy')
    return np.array([(s.step, s.value) for s in data])


def hex_to_rgba(hex_color, opacity):
    """Convert a hex color to an RGBA string with the specified opacity."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'


def plot(data, dataset, model):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    color_cycle = [colors[i % len(colors)] for i in range(len(data.keys()))]

    for i, exp_name in enumerate(data.keys()):
        steps, means, stds = data[exp_name]
        line_color = color_cycle[i]
        fill_color = hex_to_rgba(line_color, 0.2)

        fig.add_trace(go.Scatter(x=steps, y=means, mode='lines', name=f'{exp_name}',
                                 line=dict(color=line_color, width=2)))

        fig.add_trace(go.Scatter(x=steps, y=means + stds, mode='lines', line=dict(width=0),showlegend=False,
                                 legendgroup=f'{exp_name}', fillcolor=fill_color))
        fig.add_trace(go.Scatter(x=steps, y=means - stds, mode='lines', name=f'STD {exp_name}', line=dict(width=0),
                                 fill='tonexty', fillcolor=fill_color, showlegend=True, legendgroup=f'{exp_name}',
                                 visible="legendonly"))
    fig.update_layout(title=f'Val Accuracy {dataset}', xaxis_title='Epoch', yaxis_title='Accuracy')
    os.makedirs('figures', exist_ok=True)
    fig.write_html(f'figures/{dataset}_{model}_accuracy_curves.html')


def print_best_accuracy(data):
    for exp_name, (steps, means, stds) in data.items():
        max_accuracy_index = np.argmax(means)
        max_epoch = steps[max_accuracy_index]
        max_mean_accuracy = means[max_accuracy_index]
        std_at_max = stds[max_accuracy_index]
        print(
            f"{exp_name}: Epoch {max_epoch} - Max Mean Accuracy = {max_mean_accuracy * 100:.2f} ± {std_at_max * 100:.2f}")


def main():
    dataset = 'cifar100'
    model = 'preactresnet18'
    num_seeds = 3
    experiment_path = f'logs/{dataset}/{model}'
    exp_names = [
        f'{model}_ce=1.0_occe=0.0_e=150',
        f'{model}_ce=1.0_occe=1.0_e=150'
    ]

    data = {}
    for exp_name in exp_names:
        all_accuracies = []
        for seed in range(num_seeds):
            logdir = os.path.join(experiment_path, f'{exp_name}_s={seed}')
            accuracies = load_tensorboard_data(logdir)
            if len(accuracies) > 0:
                all_accuracies.append(accuracies[:, 1])
        if all_accuracies:
            all_accuracies = np.array(all_accuracies)
            means = np.mean(all_accuracies, axis=0)
            stds = np.std(all_accuracies, axis=0)
            steps = accuracies[:, 0]
            data[exp_name] = (steps, means, stds)

    plot(data, dataset, model)
    print_best_accuracy(data)

if __name__ =='__main__':
    main()
