# Towards end-to-end Cyberthreat Detection from Twitter using Multi-Task Learning 

This repository holds the data, source code and resulting model weights for the paper "Towards end-to-end Cyberthreat Detection from Twitter using Multi-Task Learning" to be presented at IJCNN 2020.

# Train models by running
```
python bin_classifier.py
python ner_classifier.py
python mt_classifier.py
```
# Example usage of a pretrained model
```
python mt_classifier.py -conf ckpts/mt/rnn_20210805172841/conf.json -load ckpts/mt/rnn_20210805172841/ckpt_step-265.pth -input <input_file>.json 
```

# Requierments
torch == 1.5.1 </br>
pandas == 1.0.5 </br>
pytorch-crf == 0.7.2 </br>

# Citation

```
@INPROCEEDINGS{9207159,
  author={N. {Dionísio} and F. {Alves} and P. M. {Ferreira} and A. {Bessani}},
  booktitle={2020 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Towards end-to-end Cyberthreat Detection from Twitter using Multi-Task Learning}, 
  year={2020},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/IJCNN48605.2020.9207159}}
```

# References

CRF module : https://github.com/kmkurn/pytorch-crf

@inproceedings{paszke2017automatic,
 title={Automatic Differentiation in {PyTorch}},
 author={Paszke, Adam and Gross, Sam and Chintala, Soumith and Chanan, Gregory and Yang, Edward and DeVito, Zachary and Lin, Zeming and Desmaison, Alban and Antiga, Luca and Lerer, Adam},
 booktitle={NIPS Autodiff Workshop},
 year={2017}
}
