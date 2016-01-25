echo "--------------"
echo "DEV SET"
echo "Embed 10"
python ./code/rnnlm.py -vocab ./data/vocab.txt -model ./models/embed10/rnn_embed10.json -weights ./models/embed10/rnn_embed10.h5 -ppl < ./data/en.dev_gold
echo "Embed 50"
python ./code/rnnlm.py -vocab ./data/vocab.txt -model ./models/embed50/rnn_embed50.json -weights ./models/embed50/rnn_embed50.h5 -ppl < ./data/en.dev_gold
echo "Embed 100"
python ./code/rnnlm.py -vocab ./data/vocab.txt -model ./models/embed100/rnn_embed100.json -weights ./models/embed100/rnn_embed100.h5 -ppl < ./data/en.dev_gold
echo "Embed 500"
python ./code/rnnlm.py -vocab ./data/vocab.txt -model ./models/embed500/rnn_embed500.json -weights ./models/embed500/rnn_embed500.h5 -ppl < ./data/en.dev_gold
echo "Embed 1000"
python ./code/rnnlm.py -vocab ./data/vocab.txt -model ./models/embed1000/rnn_embed1000.json -weights ./models/embed1000/rnn_embed1000.h5 -ppl < ./data/en.dev_gold
echo "Embed 5000"
python ./code/rnnlm.py -vocab ./data/vocab.txt -model ./models/embed5000/rnn_embed5000.json -weights ./models/embed5000/rnn_embed5000.h5 -ppl < ./data/en.dev_gold
echo "Embed 10003"
python ./code/rnnlm.py -vocab ./data/vocab.txt -model ./models/embed10003/rnn_embed10003.json -weights ./models/embed10003/rnn_embed10003.h5 -ppl < ./data/en.dev_gold
echo "--------------"
echo "TEST SET"
python ./code/rnnlm.py -vocab ./data/vocab.txt -model ./models/embed10/rnn_embed10.json -weights ./models/embed10/rnn_embed10.h5 -ppl < ./data/en.test_gold
echo "Embed 50"
python ./code/rnnlm.py -vocab ./data/vocab.txt -model ./models/embed50/rnn_embed50.json -weights ./models/embed50/rnn_embed50.h5 -ppl < ./data/en.test_gold
echo "Embed 100"
python ./code/rnnlm.py -vocab ./data/vocab.txt -model ./models/embed100/rnn_embed100.json -weights ./models/embed100/rnn_embed100.h5 -ppl < ./data/en.test_gold
echo "Embed 500"
python ./code/rnnlm.py -vocab ./data/vocab.txt -model ./models/embed500/rnn_embed500.json -weights ./models/embed500/rnn_embed500.h5 -ppl < ./data/en.test_gold
echo "Embed 1000"
python ./code/rnnlm.py -vocab ./data/vocab.txt -model ./models/embed1000/rnn_embed1000.json -weights ./models/embed1000/rnn_embed1000.h5 -ppl < ./data/en.test_gold
echo "Embed 5000"
python ./code/rnnlm.py -vocab ./data/vocab.txt -model ./models/embed5000/rnn_embed5000.json -weights ./models/embed5000/rnn_embed5000.h5 -ppl < ./data/en.test_gold
echo "Embed 10003"
python ./code/rnnlm.py -vocab ./data/vocab.txt -model ./models/embed10003/rnn_embed10003.json -weights ./models/embed10003/rnn_embed10003.h5 -ppl < ./data/en.test_gold
