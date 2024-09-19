import flwr as fl 
from typing import Dict, Optional,Tuple
import tensorflow as tf 
from dqnagent import dqnAgent
from dqn import DQN
import os

model = DQN(3)


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    passes the current round number to the client
    """
    config = {
        "round": server_round,
    }
    return config

def get_eval_fn(model):
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # model.set_weights(parameters)  # Update model with the latest parameters
        # filename = f"round_{server_round}.h5"
        # server_model_path = os.path.join(os.getcwd(),'server_model',filename)
        # model.save_weights(server_model_path)
        print("+++++++++++ Inside eval function server")
        return 0.0, {"accuracy": 0.0}
    return evaluate


strategy=fl.server.strategy.FedAvg(
fraction_fit=1.0,
min_fit_clients=5,
min_available_clients=5,
on_fit_config_fn=fit_config,
evaluate_fn=get_eval_fn(model=model)

)

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=50),
     strategy=strategy,
     )
 