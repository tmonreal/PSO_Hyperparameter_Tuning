from datasets import get_dataset_wine, get_dataset_digits, get_dataset_synthetic_multiclass
from models import KNNWrapper, DecisionTreeWrapper, MLPWrapper
from experiments import run_experiment

def main():
    # Elegir datasets
    datasets = [
        get_dataset_synthetic_multiclass(),
        get_dataset_wine(),
        get_dataset_digits(),
    ]

    # Elegir modelos
    models = [
        (KNNWrapper, "kNN"),
        (DecisionTreeWrapper, "DecisionTree"),
        (MLPWrapper, "MLP"),
    ]

    # Parámetro de penalización de complejidad 
    alpha_complexity = 0.02 # si es 0, es lo mismo que solo optimizar accuracy

    for (X, y, ds_name) in datasets:
        for model_cls, model_name in models:
            print(f"Corriendo experimento: {model_name} en {ds_name}")
            run_experiment(
                model_wrapper_cls=model_cls,
                X=X,
                y=y,
                model_name=model_name,
                dataset_name=ds_name,
                alpha_complexity=alpha_complexity,
                pso_config=dict(
                    num_particles=15,
                    max_iter=25,
                    w=0.7,
                    c1=1.5,
                    c2=1.5,
                    random_state=42,
                ),
                random_state=42,
            )


if __name__ == "__main__":
    main()