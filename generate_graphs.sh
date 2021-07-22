source .env/bin/activate

python generate_train_graph.py "Training - PEPG - Slimevolley" "train-slime-pepg" -2 35  0 1000 ./log/*slime-pepg*.hist.json
python generate_eval_graph.py "Evaluation - PEPG - Slimevolley" "eval-slime-pepg" -2 35 25 1000 ./log/*slime-pepg*.log.json


python generate_train_graph.py "Training - OpenAI-ES - Slimevolley" "train-slime-open" -2 35 0 1000 ./log/*slime-open*.hist.json
python generate_eval_graph.py "Evaluation - OpenAI-ES - Slimevolley" "eval-slime-open" -2 35  25 1000 ./log/*slime-open*.log.json

python generate_train_graph.py "Training - CMA-ES - Slimevolley" "train-slime-cmaes" -2 35  0 1000 ./log/*slime-cmaes*.hist.json
python generate_eval_graph.py "Evaluation - CMA-ES - Slimevolley" "eval-slime-cmaes" -2 35  25 1000 ./log/*slime-cmaes*.log.json


python generate_train_graph.py "Training - NSRA-ES - Slimevolley" "train-slime-nsraes" -2 35 0 900 ./log/*slime-nsraes-adam-30*.hist.json
python generate_eval_graph.py "Evaluation - NSRA-ES - Slimevolley" "eval-slime-nsraes" -2 35 25 900  ./log/*slime-nsraes-adam-30*.log.json


python generate_train_graph.py "Training - NSRA-ES - Slimevolley (metapopulation size 20)" "train-slime-nsraes-m20" -2 35 0 800  ./log/*slime-nsraes-adam-m20*.hist.json
python generate_eval_graph.py "Evaluation - NSRA-ES - Slimevolley (metapopulation size 20)" "eval-slime-nsraes-m20" -2 35 25 800 ./log/*slime-nsraes-adam-m20*.log.json


python generate_train_graph.py "Training - NSRA-ES - Slimevolley (metapopulation size 10)" "train-slime-nsraes-m10" -2 35 0 900 ./log/*slime-nsraes-adam-m10*.hist.json
python generate_eval_graph.py "Evaluation - NSRA-ES - Slimevolley (metapopulation size 10)" "eval-slime-nsraes-m10" -2 35 25 900 ./log/*slime-nsraes-adam-m10*.log.json


python generate_train_graph.py "Training - NSR-ES - Slimevolley" "train-slime-nsres" -2 35 0 900 ./log/*slime-nsres*.hist.json
python generate_eval_graph.py "Evaluation - NSR-ES - Slimevolley" "eval-slime-nsres" -2 35   25 900 ./log/*slime-nsres*.log.json


python generate_train_graph.py "Training - NS-ES - Slimevolley" "train-slime-nses" -2 35 0 900 ./log/*slime-nses*.hist.json
python generate_eval_graph.py "Evaluation - NS-ES - Slimevolley" "eval-slime-nses" -2 35 25 900 ./log/*slime-nses*.log.json




python generate_train_graph.py "Training - PEPG - Cartpole" "train-cart-pepg" 0 1000  0 1000 ./log/*cart-pepg*.hist.json
python generate_eval_graph.py "Evaluation - PEPG - Cartpole" "eval-cart-pepg" 0 1000 25 1000 ./log/*cart-pepg*.log.json


python generate_train_graph.py "Training - OpenAI-ES - Cartpole" "train-cart-open" 0 1000  0 1000 ./log/*cart-open*.hist.json
python generate_eval_graph.py "Evaluation - OpenAI-ES - Cartpole" "eval-cart-open" 0 1000  25 1000 ./log/*cart-open*.log.json

python generate_train_graph.py "Training - CMA-ES - Cartpole" "train-cart-cmaes" 0 1000  0 1000 ./log/*cart-cmaes*.hist.json
python generate_eval_graph.py "Evaluation - CMA-ES - Cartpole" "eval-cart-cmaes" 0 1000 25 1000 ./log/*cart-cmaes*.log.json


python generate_train_graph.py "Training - NSRA-ES - Cartpole" "train-cart-nsraes" 0 1000 0 900 ./log/*cart-nsraes-adam-30*.hist.json
python generate_eval_graph.py "Evaluation - NSRA-ES - Cartpole" "eval-cart-nsraes" 0 1000 25 900 ./log/*cart-nsraes-adam-30*.log.json

python generate_train_graph.py "Training - NSRA-ES - Cartpole (metapopulation size 20)" "train-cart-nsraes-m20" 0 1000  0 800 ./log/*cart-nsraes-adam-m20*.hist.json
python generate_eval_graph.py "Evaluation - NSRA-ES - Cartpole (metapopulation size 20)" "eval-cart-nsraes-m20" 0 1000 25 800 ./log/*cart-nsraes-adam-m20*.log.json


python generate_train_graph.py "Training - NSRA-ES - Cartpole (metapopulation size 10)" "train-cart-nsraes-m10" 0 1000  0 900 ./log/*cart-nsraes-adam-m10*.hist.json
python generate_eval_graph.py "Evaluation - NSRA-ES - Cartpole (metapopulation size 10)" "eval-cart-nsraes-m10" 0 1000 25 900 ./log/*cart-nsraes-adam-m10*.log.json


python generate_train_graph.py "Training - NSR-ES - Cartpole" "train-cart-nsres" 0 1000 0 900 ./log/*cart-nsres*.hist.json
python generate_eval_graph.py "Evaluation - NSR-ES - Cartpole" "eval-cart-nsres" 0 1000 25 900 ./log/*cart-nsres*.log.json


python generate_train_graph.py "Training - NS-ES - Cartpole" "train-cart-nses" 0 1000 0 900 ./log/*cart-nses*.hist.json
python generate_eval_graph.py "Evaluation - NS-ES - Cartpole" "eval-cart-nses" 0 1000 25 900 ./log/*cart-nses*.log.json

