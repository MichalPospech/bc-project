source .env/bin/activate

python generate_train_graph.py "Training - PEPG - Slimevolley" "train-slime-pepg" -2 25 ./log/*slime-pepg*.hist.json
python generate_eval_graph.py "Evaluation - PEPG - Slimevolley" "eval-slime-pepg" -2 20 ./log/*slime-pepg*.log.json


python generate_train_graph.py "Training - OpenAI-ES - Slimevolley" "train-slime-open" -2 25 ./log/*slime-open*.hist.json
python generate_eval_graph.py "Evaluation - OpenAI-ES - Slimevolley" "eval-slime-open" -2 35 ./log/*slime-open*.log.json

python generate_train_graph.py "Training - CMA-ES - Slimevolley" "train-slime-cmaes" -2 25 ./log/*slime-cmaes*.hist.json
python generate_eval_graph.py "Evaluation - CMA-ES - Slimevolley" "eval-slime-cmaes" -2 35 ./log/*slime-cmaes*.log.json


python generate_train_graph.py "Training - NSRA-ES - Slimevolley" "train-slime-nsra" -2 25 ./log/*slime-nsraes-adam-30*.hist.json
python generate_eval_graph.py "Evaluation - NSRA-ES - Slimevolley" "eval-slime-nsra" -2 25 ./log/*slime-nsraes-adam-30*.log.json





python generate_train_graph.py "Training - PEPG - Cartpole" "train-cart-pepg" 0 1000 ./log/*cart-pepg*.hist.json
python generate_eval_graph.py "Evaluation - PEPG - Cartpole" "eval-cart-pepg" 0 1000  ./log/*cart-pepg*.log.json


python generate_train_graph.py "Training - OpenAI-ES - Cartpole" "train-cart-open" 0 1000  ./log/*cart-open*.hist.json
python generate_eval_graph.py "Evaluation - OpenAI-ES - Cartpole" "eval-cart-open" 0 1000  ./log/*cart-open*.log.json

python generate_train_graph.py "Training - CMA-ES - Cartpole" "train-cart-cmaes" 0 1000  ./log/*cart-cmaes*.hist.json
python generate_eval_graph.py "Evaluation - CMA-ES - Cartpole" "eval-cart-cmaes" 0 1000  ./log/*cart-cmaes*.log.json


python generate_train_graph.py "Training - NSRA-ES - Cartpole" "train-cart-nsra" 0 1000  ./log/*cart-nsraes-adam-30*.hist.json
python generate_eval_graph.py "Evaluation - NSRA-ES - Cartpole" "eval-cart-nsra" 0 1000  ./log/*cart-nsraes-adam-30*.log.json